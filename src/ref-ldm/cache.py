# TODO: Not used, to be removed
import os
import sys
import argparse
import hashlib
import json
import shutil
import subprocess

# The original script to run for a cache miss
INFERENCE_SCRIPT = 'inference.py'
# Directory to store cached results
CACHE_DIR = '.inference_cache'

def parse_args():
    parser = argparse.ArgumentParser()
    # Image paths
    parser.add_argument('--output_path', type=str, default='result.png')
    parser.add_argument('--lq_path', type=str, default='assets/demo/lq.png')
    parser.add_argument('--ref_paths', nargs='+', default=['assets/demo/ref0.png', 'assets/demo/ref1.png', 'assets/demo/ref2.png', 'assets/demo/ref3.png'])
    # Model paths
    parser.add_argument('--model_config_path', type=str, default='configs/refldm.yaml')
    parser.add_argument('--model_ckpt_path', type=str, default='ckpts/refldm.ckpt')
    parser.add_argument('--vae_ckpt_path', type=str, default='ckpts/vqgan.ckpt')
    # Inference settings
    parser.add_argument('--ddim_step', type=int, default=50)
    parser.add_argument('--ddim_schedule', type=str, default='uniform_trailing')
    parser.add_argument('--cfg_scale', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=2024)
    args = parser.parse_args()
    return args

def get_args_hash(args):
    args_dict = vars(args)

    keys_to_exclude = ['output_path']
    hashable_dict = {k: v for k, v in args_dict.items() if k not in keys_to_exclude}

    canonical_string = json.dumps(hashable_dict, sort_keys=True)

    hasher = hashlib.sha256(canonical_string.encode('utf-8'))
    return hasher.hexdigest()


def main_cached():
    user_args = parse_args()

    cache_key = get_args_hash(user_args)
    cached_image_filename = f"{cache_key}.png"
    cached_image_path = os.path.join(CACHE_DIR, cached_image_filename)

    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(cached_image_path):
        print(f"Cache hit! Loading result for key: {cache_key[:10]}...")
        shutil.copy(cached_image_path, user_args.output_path)
        print(f"   Result copied to: {user_args.output_path}")
    else:
        print(f"Cache miss. Running inference for key: {cache_key[:10]}...")

        command = [sys.executable, INFERENCE_SCRIPT]

        for key, value in vars(user_args).items():
            # Override the output path to adapt cache
            if key == 'output_path':
                command.extend([f'--{key}', cached_image_path])
                continue

            if isinstance(value, list):
                command.append(f'--{key}')
                command.extend(value)
            else:
                command.extend([f'--{key}', str(value)])

        print(f"   Executing: {' '.join(command)}")

        try:
            subprocess.run(command, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\n[ERROR] The inference script failed to run: {e}", file=sys.stderr)
            sys.exit(1)

        print("\n   Inference complete. Result is now cached.")

        shutil.copy(cached_image_path, user_args.output_path)
        print(f"   Result copied to: {user_args.output_path}")


if __name__ == '__main__':
    main_cached()
