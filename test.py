from cldm.hack import disable_verbosity, enable_sliced_attention
disable_verbosity()

import cv2
import einops
import numpy as np
import torch
import random
import glob
from PIL import Image
import os
import argparse
import time  

from torchvision.utils import save_image
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default='./checkpoints/pretrained.bin', type=str)
parser.add_argument("--same_folder", default='output_DiffLLFace', type=str)
parser.add_argument("--input_folder", default='./dataset/CelebA/test/LLR_x16', type=str)

parser.add_argument("--use_float16", default=False, type=bool)
parser.add_argument("--save_memory", default=False, type=bool) # Cannot use. Has bugs

if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint_file = args.checkpoint

    print("====== Load parameters ======")
    model = create_model('./models/cldm_v15.yaml').cpu()


    new_state_dict = {}
    trained_state_dict = load_state_dict(checkpoint_file, location='cpu')
    keys_to_process = list(trained_state_dict.keys())
    for key in keys_to_process:
        value = trained_state_dict[key]
        if '_forward_module' in key:
            new_state_dict[key.replace('_forward_module.', '')] = value
        else:
            new_state_dict[key] = value
    
    init_state_dict = load_state_dict('./models/control_sd15.ckpt', location='cpu')
    for key, value in init_state_dict.items():
        if key not in new_state_dict and "cond_stage_model.transformer" not in key:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict, strict=False)
    print("====== Finish loading parameters ======")

    if args.use_float16:
        model = model.cuda().to(dtype=torch.float16)
    else:
        model = model.cuda()
    diffusion_sampler = DDIMSampler(model)
    # 
    os.makedirs("sampling_steps", exist_ok=True)

    def process(input_image, prompt="", num_samples=1, image_resolution=256, diffusion_steps=50, guess_mode=False, strength=1.0, scale=9.0, seed=42, eta=0.0):
        with torch.no_grad():
            detected_map = resize_image(HWC3(input_image), image_resolution)
            H, W, C = detected_map.shape

            if args.use_float16:
                control = torch.from_numpy(detected_map.copy()).cuda().to(dtype=torch.float16) / 255.0
            else:
                control = torch.from_numpy(detected_map.copy()).cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            cond_txt = model.get_unconditional_conditioning(num_samples)
            cond = {"c_concat": [control], "c_crossattn": [cond_txt]}

            un_cond_txt = model.get_unconditional_conditioning(num_samples)
            un_cond = {"c_concat": [control], "c_crossattn": [un_cond_txt]}  
            
            shape = (4, H // 8, W // 8)

            if args.save_memory:
                model.low_vram_shift(is_diffusing=True)
            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

            samples, intermediates = diffusion_sampler.sample(
                diffusion_steps, 
                num_samples,
                shape, 
                cond, 
                verbose=False, 
                eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond, 
                dmp_order=3
            )

            if args.save_memory:
                model.low_vram_shift(is_diffusing=False)
            
            if args.use_float16:
                x_samples = model.decode_first_stage(samples.to(dtype=torch.float16))
            else:
                x_samples = model.decode_first_stage(samples)
                
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return results

    img_list = glob.glob(f"{args.input_folder}/*.*")
    print(f"Find {len(img_list)} files in {args.input_folder}")

    for img_path in img_list:
        save_name = os.path.split(img_path)[1]
        save_name = os.path.splitext(save_name)[0] + ".png"
        save_path = os.path.join(args.same_folder, save_name)

        if os.path.exists(save_path):
            print(f"Exists {save_path}, skip.")
            continue

        input_image = cv2.imread(img_path)
        if input_image is not None:
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        output = process(input_image, num_samples=1)[0]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{save_name} time: {elapsed_time:.2f} 秒")

        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        os.makedirs(args.same_folder, exist_ok=True)
        cv2.imwrite(save_path, output)