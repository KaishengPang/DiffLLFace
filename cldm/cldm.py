import einops
import torch
import torch as th
import torch.nn as nn
import pickle
import numpy as np
import cv2
import os

light_model_count = 0

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
)

from einops import rearrange, repeat
from torchvision.utils import make_grid, save_image
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, exists, instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from newprior.model.mynet import LFSRNet_prior, LFSRNet_onlyconv

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(dtype=x.dtype)
            emb = self.time_embed(t_emb)
            h = x.type(x.dtype)
            for module in self.input_blocks:
                # print("h", h.dtype, "emb", emb.dtype, "context", context.dtype)
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class ControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        # self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, model_channels, 3, padding=1))
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return TimestepEmbedSequential(zero_module(conv_nd(self.dims, channels, channels, 1, padding=0)))
    
    @staticmethod
    def adjust_brightness_in_frequency_domain(hint, brightness_factor):
        """
        Adjust brightness in frequency domain for a single image using PyTorch.
        """
        # Ensure hint is a 4D tensor [batch_size, channels, height, width]
        if hint.dim() != 4:
            raise ValueError(f"Expected hint to be a 4D tensor, but got {hint.dim()}D tensor with shape {hint.shape}")

        batch_size, channels, height, width = hint.shape

        # Convert hint to [0, 255] range
        hint_np = (hint * 255).type(torch.uint8)

        # Initialize output tensor
        img_back_torch = torch.zeros_like(hint)

        # Process each image in the batch
        for i in range(batch_size):
            img = hint_np[i]  # Get the i-th image
            if channels == 3:
                # Multi-channel image (e.g., RGB)
                f_r = torch.fft.fft2(img[0])  # Red channel
                f_g = torch.fft.fft2(img[1])  # Green channel
                f_b = torch.fft.fft2(img[2])  # Blue channel

                fshift_r = torch.fft.fftshift(f_r)
                fshift_g = torch.fft.fftshift(f_g)
                fshift_b = torch.fft.fftshift(f_b)

                fshift_r = fshift_r * brightness_factor
                fshift_g = fshift_g * brightness_factor
                fshift_b = fshift_b * brightness_factor

                f_ishift_r = torch.fft.ifftshift(fshift_r)
                f_ishift_g = torch.fft.ifftshift(fshift_g)
                f_ishift_b = torch.fft.ifftshift(fshift_b)

                img_r_back = torch.fft.ifft2(f_ishift_r)
                img_g_back = torch.fft.ifft2(f_ishift_g)
                img_b_back = torch.fft.ifft2(f_ishift_b)

                img_r_back = torch.abs(img_r_back).clamp(0, 255).type(torch.uint8)
                img_g_back = torch.abs(img_g_back).clamp(0, 255).type(torch.uint8)
                img_b_back = torch.abs(img_b_back).clamp(0, 255).type(torch.uint8)

                img_back = torch.stack([img_r_back, img_g_back, img_b_back], dim=0)
            else:
                raise ValueError(f"Unsupported number of channels: {channels}")

            img_back_torch[i] = img_back

        # Normalize back to [0, 1]
        img_back_torch = img_back_torch.float() / 255.0

        return img_back_torch

    def add_prior(self):

        self.prior_conv = LFSRNet_prior(n_resblocks=1, in_channels=3)

        # self.prior_conv = LFSRNet_onlyconv()

        self.input_hint_block[0] = conv_nd(2, 10, 16, 3, padding=1)

        # self.input_hint_block[0] = conv_nd(2, 3, 16, 3, padding=1)
        
        # self.input_hint_block[0] = conv_nd(2, 1, 16, 3, padding=1)


    def forward(self, x, hint, hint_new, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        hint1 = self.prior_conv(hint)
        hint2 = self.adjust_brightness_in_frequency_domain(hint, 3.5)
        # os.makedirs("hint2", exist_ok=True)
        hint2 = hint2.half()
        hint_new = hint_new.half()
        # save_image(hint2, "hint2/hint2_image.png", normalize=True)
        hint = torch.cat((hint1, hint2, hint_new), dim=1)
        #hint就是先验的输入！！！
        
        guided_hint = self.input_hint_block(hint, emb, context)

        outs = []

        h = x.type(x.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs

class ControlLDM(LatentDiffusion):
    #ControlNet----子类，融合了ControlNet、LDM的新类
    def __init__(self, control_stage_config, control_key, only_mid_control, estimation_config=None, refinement_config=None, encoder_config=None, *args, **kwargs):
        # 从kwargs中去除这些不被父类接受的参数
        if 'estimation_config' in kwargs:
            del kwargs['estimation_config']
        if 'refinement_config' in kwargs:
            del kwargs['refinement_config']
        if 'encoder_config' in kwargs:
            del kwargs['encoder_config']
        if 'control_stage_config' in kwargs:
            del kwargs['control_stage_config']
            
        super().__init__(*args, **kwargs)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        self.control_scales = [1.0] * 13
        
        # 保存这些配置以供后续使用
        self.estimation_config = estimation_config
        self.refinement_config = refinement_config
        self.encoder_config = encoder_config
        self.control_stage_config = control_stage_config
        
            
        with open("empty_embedding.pkl", 'rb') as f:
            self.cond_txt_empty = pickle.load(f)

    @torch.no_grad()
    def add_new_layers(self):
        self.control_model.add_prior()

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        #x是"jpg"输入
        x = x.to(dtype=self.dtype())
        control = batch[self.control_key]
        #control是"hint"输入
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        # control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).to(dtype=self.dtype())
        return x, dict(c_crossattn=[""] * len(control), c_concat=[control])
    # 在调用 cv2.normalize 之前检查数据
    # def check_and_normalize(self, average_map_np):
    #     # 检查是否存在 NaN 或 inf
    #     if np.isnan(average_map_np).any() or np.isinf(average_map_np).any():
    #         print("Warning: average_map_np contains NaN or inf values. Replacing with zeros.")
    #         average_map_np = np.nan_to_num(average_map_np, nan=0.0, posinf=0.0, neginf=0.0)
        
    #     # 打印调试信息
    #     print("average_map_np shape:", average_map_np.shape)
    #     print("average_map_np dtype:", average_map_np.dtype)
    #     print("average_map_np min:", np.min(average_map_np))
    #     print("average_map_np max:", np.max(average_map_np))
    #     print("average_map_np mean:", np.mean(average_map_np))
    #     # 将数据类型转换为 float32
    #     average_map_np = average_map_np.astype(np.float32)

    #     # 将数据范围归一化到 [0, 1]
    #     average_map_np = (average_map_np - np.min(average_map_np)) / (np.max(average_map_np) - np.min(average_map_np))
    #     # 归一化到 [0, 255]
    #     average_map_np = cv2.normalize(average_map_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #     return average_map_np

    def apply_model(self, x_noisy, t, cond):
        # global light_model_count
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model 

        # cond_txt = torch.cat(cond['c_crossattn'], 1)
        cond_txt = self.cond_txt_empty.detach().to(x_noisy.device, dtype=self.dtype())
        cond_txt = cond_txt.repeat(x_noisy.shape[0], 1, 1)
        x_noisy = x_noisy.to(dtype=cond_txt.dtype)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:  
            c_concat_tensor = torch.cat(cond['c_concat'], dim=1)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, hint=c_concat_tensor, control_scales=self.control_scales, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        # return self.get_learned_conditioning([""] * N)
        cond_txt = self.cond_txt_empty.detach()
        return cond_txt.repeat(N, 1, 1)
    
    @torch.no_grad()
    def change_first_stage(self, checkpoint_file):
        del self.first_stage_model
        from my_vae.autoencoder import AutoencoderKL
        self.first_stage_model = AutoencoderKL(load_checkpoint=False)

        state_dict = torch.load(checkpoint_file, map_location=torch.device("cpu"))["state_dict"]
        new_state_dict = {}
        for s in state_dict:
            if "my_vae" in s:
                new_state_dict[s.replace("my_vae.", "")] = state_dict[s]
        self.first_stage_model.load_state_dict(new_state_dict)
        print("Successfully load new auto-encoder")

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        # c_cat, c = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        c_cat = c["c_concat"][0]
        c = self.get_unconditional_conditioning(N)
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        #z进行了encoder和decoder，return回来的结果
        log["control"] = c_cat * 2.0 - 1.0
        # log["control"] = c_cat[:,:,:3]
        # log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg.to(self.dtype()))
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.diffusion_model.control_model.parameters())
        params += list(self.model.diffusion_model.estimation_model.parameters())
        params += list(self.model.diffusion_model.refinement_model.parameters())
        params += list(self.model.diffusion_model.encoder.parameters())
        params += list(self.model.diffusion_model.middle_block[2].spade.parameters())
        params += list(self.first_stage_model.decoder.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = DeepSpeedCPUAdam(filter(lambda p: p.requires_grad, params), lr=lr)
        # opt = torch.optim.AdamW(params, lr=lr)
        
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            # self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            # self.cond_stage_model = self.cond_stage_model.cuda()

