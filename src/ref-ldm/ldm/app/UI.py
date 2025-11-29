import gradio as gr

from dataclasses import dataclass
CSS = """
.gradio-container { font-family: 'Microsoft YaHei', 'SimSun', sans-serif; }
.gr-button-primary { background: linear-gradient(to bottom right, #4A90E2, #0056b3); border-color: #0056b3; }
#preview_gallery .h-full { object-fit: cover; }
#result_gallery .h-full { object-fit: contain; }
"""

@dataclass
class UILayout:
    lq_input_img: gr.Image
    ref_input_files: gr.File
    process_btn: gr.Button
    preview_gallery: gr.Gallery
    result_img: gr.Image
    result_gallery: gr.Gallery
def create_ui_layout() -> UILayout:
    gr.Markdown("## Ref-LDM: 有参考的人脸图像修复")
    gr.Markdown("上传一张低质量（模糊、有噪点等）的人脸图片和 1-4 张高质量的参考人脸图片，模型将修复低质量图片。")
    with gr.Row(variant='panel'):
        # --- Left Column (Inputs) ---
        with gr.Column(scale=1):
            gr.Markdown("### 1. 输入控制")
            with gr.Accordion("上传图片", open=True):
                lq_input_img = gr.Image(
                    label="上传低质量图片 (LQ)",
                    type="pil",
                    image_mode="RGB"
                )
                ref_input_files = gr.File(
                    label="上传参考图片 (Ref, 最多4张)",
                    file_count="multiple",
                    file_types=["image"]
                )
            process_btn = gr.Button("开始处理", variant="primary")
        # --- Right Column (Outputs) ---
        with gr.Column(scale=2):
            gr.Markdown("### 2. 结果展示")
            with gr.Tab("输入预览"):
                preview_gallery = gr.Gallery(
                    label="输入图片预览",
                    columns=5,
                    height="auto",
                    elem_id="preview_gallery"
                )
                gr.Markdown("上方第一张为低质量图，其余为参考图。")
            with gr.Tab("修复结果"):
                result_img = gr.Image(label="修复后图片", interactive=False)
                result_gallery = gr.Gallery(
                    label="效果对比 (LQ / Refs / Result)",
                    columns=5,
                    height="auto",
                    elem_id="result_gallery"
                )
                gr.Markdown("上方为最终修复结果，下方为修复前后及参考图的对比。")
    return UILayout(
        lq_input_img=lq_input_img,
        ref_input_files=ref_input_files,
        process_btn=process_btn,
        preview_gallery=preview_gallery,
        result_img=result_img,
        result_gallery=result_gallery,
    )