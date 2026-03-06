from cldm.hack import disable_verbosity, enable_sliced_attention
disable_verbosity()

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from dataset import create_webdataset
import webdataset as wds
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import torch

pl.seed_everything(42)
# Configs

resume_path = ''
GT_images = './dataset/CelebA/train/HR/*.*'
LLR_images = './dataset/CelebA/train/LLR_x16/*.*'
batch_size = 4
number_of_gpu = 1
learning_rate = 1e-4
logger_freq = 1000
name = f"CelebA-DiffLLFace"

sd_locked = True
only_mid_control = False

try:
    model = create_model('./models/cldm_v15.yaml').cpu()

except Exception as e:
    print(f"Error creating model: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

init_state_dict = load_state_dict('./models/control_sd15.ckpt', location='cpu')
init_new_state_dict = {}
for s in init_state_dict:
    if "model.diffusion_model.control_model.input_hint_block.0.weight" not in s:
        init_new_state_dict[s] = init_state_dict[s]

if resume_path != "":
    new_state_dict = load_state_dict(resume_path, location='cpu')
    for s in new_state_dict:
        if 'light_model.' in s:
            init_new_state_dict[s] = new_state_dict[s]

model.load_state_dict(init_new_state_dict, strict=False)

if resume_path != "":
    state_dict = load_state_dict(resume_path, location='cpu')
    new_state_dict = {}
    for sd_name, sd_param in state_dict.items():
        if 'control_model' in sd_name:
            new_state_dict[sd_name.replace('control_model.', '')] = sd_param

    model.control_model.load_state_dict(new_state_dict)

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = create_webdataset(
    jpg_dir = GT_images,
    hint_dir = LLR_images
)

dataloader = wds.WebLoader(
    dataset          =   dataset,
    batch_size       =   batch_size,
    num_workers      =   2,
    pin_memory       =   False,
    prefetch_factor  =   2,
)

logger = ImageLogger(batch_frequency=logger_freq)

checkpoint_callback = ModelCheckpoint(
    dirpath                   =     'checkpoints',
    filename                  =     name + '-{epoch:02d}-{step}',
    monitor                   =     'step',
    save_last                 =     False,
    save_top_k                =     -1,
    verbose                   =     True,
    every_n_train_steps       =     20000,   # How frequent to save checkpoint
    save_on_train_epoch_end   =     True,
)

strategy = DeepSpeedStrategy(
    stage                     =     2, 
    offload_optimizer         =     True, 
    cpu_checkpointing         =     True
)

trainer = pl.Trainer(devices                   =     number_of_gpu,
                     strategy                  =     strategy,
                     precision                 =     32,
                     sync_batchnorm            =     True,
                     accelerator               =     'gpu',
                     callbacks                 =     [logger, checkpoint_callback])

# Train
trainer.fit(model, dataloader)
