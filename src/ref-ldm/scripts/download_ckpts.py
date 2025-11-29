# Created by @Xiongzile
import os
import sys
import requests
from tqdm import tqdm

def download():
    ckpt_info = {
        "ckpts/refldm.ckpt": "https://github.com/ChiWeiHsiao/ref-ldm/releases/download/1.0.0/refldm.ckpt",
        "ckpts/vqgan.ckpt": "https://github.com/ChiWeiHsiao/ref-ldm/releases/download/1.0.0/vqgan.ckpt"
    }

    print(">>> Checking and preparing model checkpoints...")

    for ckpt_path, url in ckpt_info.items():
        if not os.path.exists(ckpt_path):
            while True:
                user_input = input(f"--> File not found: {ckpt_path}\n    Download it now? [Y/n]: ").strip().lower()

                if user_input in ["y", "yes", ""]:
                    print(f"    Proceeding with download from: {url}")
                    download_file(url, ckpt_path)
                    break
                elif user_input in ["n", "no"]:
                    print("    Operation cancelled. Cannot continue without required files.", file=sys.stderr)
                    sys.exit(1)
                else:
                    print("    Invalid input. Please enter 'y' or 'n'.")

    print(">>> All required files are ready.")


def download_file(url, destination):
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024

            with tqdm(total=total_size, unit='iB', unit_scale=True,
                      desc=f"    Downloading {os.path.basename(destination)}") as progress_bar:
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        progress_bar.update(len(chunk))
                        f.write(chunk)

            if total_size != 0 and progress_bar.n != total_size:
                raise IOError("Download incomplete. File might be corrupted.")

        print(f"    File downloaded successfully to: {destination}")

    except (requests.exceptions.RequestException, IOError) as e:
        print(f"\n    [ERROR] Download failed: {e}", file=sys.stderr)
        if os.path.exists(destination):
            os.remove(destination)
        sys.exit(1)


if __name__ == '__main__':
    download()
