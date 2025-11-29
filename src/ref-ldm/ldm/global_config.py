# TODO: All variables in this file must be uppercase！！！
import torch

import os

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import load_model


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL_CONFIG_PATH = 'configs/refldm.yaml'
MODEL_CKPT_PATH = 'ckpts/refldm.ckpt'
VAE_CKPT_PATH = 'ckpts/vqgan.ckpt'

print("[global_config] begin load environment...")
if not os.path.exists(MODEL_CKPT_PATH) or not os.path.exists(VAE_CKPT_PATH):
    print("\n[ERROR] No model files found")
    print(f"Please make sure '{MODEL_CKPT_PATH}' and '{VAE_CKPT_PATH}' exist.")
    print("You can use the script(download_ckpts.py) to download the model file or download it manually.")

MODEL = load_model(MODEL_CONFIG_PATH, MODEL_CKPT_PATH, VAE_CKPT_PATH, DEVICE)
SAMPLER = DDIMSampler(MODEL, print_tqdm=False, schedule='uniform_trailing')
print("[global_config] environment loaded. Let's go!")
