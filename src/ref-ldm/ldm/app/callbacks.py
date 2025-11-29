import gradio as gr
import torch
from torchvision.transforms.functional import to_pil_image

import time
from PIL import Image

from ldm.global_config import DEVICE, MODEL, SAMPLER
from ldm.util import read_and_normalize_image


def process_and_restore(lq_image, ref_images, progress=gr.Progress(track_tqdm=True)):
    """The main function to run inference when the button is clicked."""
    # Input validation
    if lq_image is None:
        raise gr.Error("请上传一张低质量图片 (LQ Image)！")
    if not ref_images or len(ref_images) > 4:
        raise gr.Error("请上传 1 到 4 张参考图片 (Reference Images)！")

    progress(0, desc="准备输入数据...")
    time.sleep(0.2)

    try:
        # Prepare condition
        lq_tensor = read_and_normalize_image(lq_image)
        ref_tensors = [read_and_normalize_image(Image.open(p.name)) for p in ref_images]
        ref_tensor_cat = torch.concat(ref_tensors, axis=-1)

        c = {
            'lq_image': lq_tensor.unsqueeze(0).to(DEVICE),
            'ref_image': ref_tensor_cat.unsqueeze(0).to(DEVICE),
        }

        progress(0.2, desc="编码条件特征...")
        # Encode condition from image to latent
        with torch.no_grad():
            c = MODEL.get_learned_conditioning(c)

        # CFG null condition = no reference
        uc = {k: c[k].detach().clone() for k in c.keys()}
        uc['ref_image'] *= 0

    except Exception as e:
        raise gr.Error(f"图片预处理失败: {e}")

    # Sample initial latent xT
    latent_shape = [MODEL.model.diffusion_model.out_channels, MODEL.image_size, MODEL.image_size]
    torch.manual_seed(2024)  # Use a fixed seed for reproducibility
    xT = torch.randn([1, *latent_shape], device=DEVICE)

    progress(0.4, desc="开始扩散去噪 (DDIM Sampling)...")
    # Diffusion denoising
    with torch.no_grad(), MODEL.ema_scope():
        output_latent, _ = SAMPLER.sample(
            S=50,  # DDIM steps
            unconditional_guidance_scale=1.5,  # CFG scale
            conditioning=c,
            unconditional_conditioning=uc,
            shape=latent_shape,
            x_T=xT,
            batch_size=1,
            verbose=False,
        )

    progress(0.9, desc="解码生成结果...")
    # Decode output latent to image
    with torch.no_grad():
        output_image = MODEL.decode_first_stage(output_latent)

    # Post-process and save result
    output_image = ((output_image + 1.0) / 2.0).clamp(0.0, 1.0)
    output_image = output_image.squeeze(0).cpu()
    final_pil_image = to_pil_image(output_image)

    progress(1.0, desc="处理完成！")

    # Prepare comparison gallery
    comparison_gallery = [lq_image] + [Image.open(p.name) for p in ref_images] + [final_pil_image]

    return final_pil_image, comparison_gallery


def update_preview(lq_img, ref_list):
    """Updates the preview gallery when images are uploaded."""
    if lq_img is None and not ref_list:
        return None

    preview_list = []
    if lq_img:
        preview_list.append(lq_img)
    if ref_list:
        for ref_file in ref_list:
            try:
                preview_list.append(Image.open(ref_file.name))
            except Exception:
                # Handle cases where a file might not be a valid image
                pass
    return preview_list