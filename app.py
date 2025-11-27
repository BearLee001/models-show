# gradio_frontend.py (适配 Gradio 3.50.2)
import gradio as gr
import requests
import tempfile
import os
from PIL import Image
import io

# 后端服务配置
CODEFORMER_SERVICE_URL = "http://localhost:8001/restore"


def restore_face(image, weight):
    if image is None:
        return None, "请先上传图片"

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            if hasattr(image, 'shape'):  # 判断是否为 numpy array
                pil_image = Image.fromarray(image)
                pil_image.save(tmp_file.name, "PNG")
            else:
                image.save(tmp_file.name, "PNG")

            temp_path = tmp_file.name

        files = {'file': open(temp_path, 'rb')}
        data = {'weight': weight}

        response = requests.post(
            CODEFORMER_SERVICE_URL,
            files=files,
            data=data,
            timeout=60
        )

        if response.status_code == 200:
            restored_image = Image.open(io.BytesIO(response.content))

            files['file'].close()
            os.unlink(temp_path)

            return restored_image, "修复完成!"
        else:
            files['file'].close()
            os.unlink(temp_path)

            error_msg = f"服务返回错误: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f" - {error_detail.get('error', '未知错误')}"
            except:
                error_msg += f" - {response.text}"

            return None, error_msg

    except requests.exceptions.Timeout:
        return None, "请求超时，请稍后重试"
    except requests.exceptions.ConnectionError:
        return None, f"无法连接到服务，请确保 CodeFormer 服务正在运行在 {CODEFORMER_SERVICE_URL}"
    except Exception as e:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None, f"处理过程中发生错误: {str(e)}"


def create_demo():

    with gr.Blocks(
            title="AI 人脸修复平台 - CodeFormer",
            css="""
        .gradio-container {
            max-width: 1000px !important;
        }
        .output-image {
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        """
    ) as demo:
        gr.Markdown(
            """
            # 🎭 AI 人脸修复平台
            ### 使用 CodeFormer 技术修复和增强人脸图片

            上传图片后，调整修复强度参数，点击"开始修复"按钮即可。
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📤 上传图片")
                    input_image = gr.Image(
                        label="选择图片",
                        type="pil",
                        height=300
                    )

                with gr.Group():
                    gr.Markdown("### ⚙️ 参数设置")
                    weight_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="修复强度 (Weight)",
                        info="较小值(0.1-0.3): 更自然但改变较多 | 较大值(0.7-1.0): 保留更多原图特征"
                    )

                restore_btn = gr.Button(
                    "🚀 开始修复",
                    variant="primary"
                )

                status_text = gr.Textbox(
                    label="状态",
                    placeholder="等待处理...",
                    interactive=False
                )

            with gr.Column(scale=1):
                gr.Markdown("### 📥 修复结果")
                output_image = gr.Image(
                    label="修复后的图片",
                    type="pil",
                    height=400
                )

                with gr.Row():
                    download_btn = gr.Button("💾 下载结果")
                    clear_btn = gr.Button("🗑️ 清空", variant="secondary")

        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            **使用步骤:**
            1. 上传一张包含人脸的图片（支持 PNG、JPG、JPEG 格式）
            2. 调整修复强度参数（推荐值 0.5-0.7）
            3. 点击"开始修复"按钮
            4. 等待处理完成，查看并下载结果

            **参数说明:**
            - **修复强度 (Weight)**: 控制修复程度
              - 较低值 (0.1-0.3): 修复效果更明显，可能改变更多原图特征
              - 较高值 (0.7-1.0): 保留更多原图细节，修复效果较轻微
              - 推荐值 (0.5): 平衡修复效果和保真度

            **注意事项:**
            - 确保 CodeFormer 后端服务正在运行
            - 处理时间根据图片大小和服务器性能可能需 10-30 秒
            - 建议图片尺寸不要过大（最好在 1024x1024 像素以内）
            """)

        restore_btn.click(
            fn=restore_face,
            inputs=[input_image, weight_slider],
            outputs=[output_image, status_text]
        )

        clear_btn.click(
            fn=lambda: [None, None, "已清空"],
            inputs=[],
            outputs=[input_image, output_image, status_text]
        )

        def download_result(image):
            if image is not None:
                temp_dir = "downloads"
                os.makedirs(temp_dir, exist_ok=True)
                download_path = f"{temp_dir}/restored_result.png"
                image.save(download_path)
                return download_path
            return None

        download_btn.click(
            fn=download_result,
            inputs=[output_image],
            outputs=gr.File(label="下载修复结果")
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )