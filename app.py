# gradio_frontend.py
import gradio as gr
import requests
import os
from PIL import Image

# 后端服务配置
CODEFORMER_SERVICE_URL = "http://localhost:8001"


def restore_face(input_path, weight, output_dir=None):
    """
    调用后端服务进行人脸修复
    """
    # 准备请求数据
    data = {
        "input_path": input_path,
        "weight": weight
    }
    print(data)

    if output_dir:
        data["output_path"] = output_dir

    try:
        response = requests.post(
            f"{CODEFORMER_SERVICE_URL}/restore",
            json=data
        )

        result = response.json()

        if result.get("status") == "success":
            # 获取结果文件
            output_file = result["output_files"]["main_result"]
            file_response = requests.get(f"{CODEFORMER_SERVICE_URL}/result/{output_file}")

            if file_response.status_code == 200:
                # 保存结果图片
                result_path = f"temp_result_{os.path.basename(input_path)}"
                with open(result_path, "wb") as f:
                    f.write(file_response.content)
                # 压缩图片到 400x400
                compressed_path = compress_image(result_path)
                return compressed_path, "修复完成!"
            else:
                return None, "无法获取结果文件"
        else:
            return None, f"处理失败: {result.get('error', '未知错误')}"

    except Exception as e:
        return None, f"请求失败: {str(e)}"


def list_available_images():
    """
    获取可用的输入图片列表
    """
    try:
        response = requests.get(f"{CODEFORMER_SERVICE_URL}/list_images")
        result = response.json()
        if "images" in result:
            return result["absolute_paths"]
        return []
    except:
        return []


def create_demo():
    """创建 Gradio 演示界面"""

    with gr.Blocks(title="AI 人脸修复平台") as demo:
        gr.Markdown("# 🎭 AI 人脸修复平台")

        with gr.Row():
            with gr.Column():
                # 图片选择
                available_images = list_available_images()
                print(available_images)
                image_select = gr.Dropdown(
                    choices=available_images,
                    label="选择输入图片",
                    value=available_images[0] if available_images else None
                )

                # 修复强度
                weight_slider = gr.Slider(
                    0.0, 1.0, 0.5, step=0.1,
                    label="修复强度",
                    info="较小值：更自然 | 较大值：保留原图特征"
                )

                # 输出目录（可选）
                output_dir = gr.Textbox(
                    label="输出目录（可选）",
                    placeholder="留空使用默认目录",
                    value=""
                )

                restore_btn = gr.Button("🚀 开始修复", variant="primary")
                status_text = gr.Textbox(label="状态", interactive=False)

            with gr.Column():
                output_image = gr.Image(label="修复结果", height=400, width=400)

        # 按钮事件
        restore_btn.click(
            fn=restore_face,
            inputs=[image_select, weight_slider, output_dir],
            outputs=[output_image, status_text]
        )

        # 刷新图片列表
        refresh_btn = gr.Button("🔄 刷新图片列表")

        @refresh_btn.click
        def refresh_images():
            new_images = list_available_images()
            new_choices = [os.path.basename(img) for img in new_images]
            return gr.Dropdown.update(choices=new_choices, value=new_choices[0] if new_choices else None)

    return demo


def compress_image(image_path):
    """
    将图片压缩到 400x400 像素
    """
    try:
        with Image.open(image_path) as img:
            img_resized = img.resize((400, 400), Image.Resampling.LANCZOS)

            compressed_path = f"compressed_{os.path.basename(image_path)}"
            img_resized.save(compressed_path, "PNG")

            return compressed_path
    except Exception as e:
        print(f"图片压缩失败: {e}")
        return image_path  # 如果压缩失败，返回原图


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)