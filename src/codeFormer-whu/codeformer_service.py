# codeformer_service.py
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import os
from fastapi.responses import FileResponse

app = FastAPI(title="CodeFormer 人脸修复服务")


class RestoreRequest(BaseModel):
    input_path: str  # 输入文件路径
    output_path: str = None  # 输出目录路径（可选）
    weight: float = 0.5  # 修复权重


@app.post("/restore")
async def restore_face(request: RestoreRequest):
    """
    人脸修复服务
    - input_path: 输入图片路径
    - output_path: 输出目录路径（可选，默认自动生成）
    - weight: 修复权重 0.0-1.0
    """
    if not os.path.exists(request.input_path):
        return {"error": "输入文件不存在", "file_path": request.input_path}

    cmd_args = [
        "python", "inference_codeformer.py",
        "-w", str(request.weight),
        "--input_path", request.input_path
    ]

    if request.output_path:
        cmd_args.extend(["--output_path", request.output_path])
        # 确保输出目录存在
        os.makedirs(request.output_path, exist_ok=True)
    print("begin execute {}", cmd_args)
    result = subprocess.run(cmd_args, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Subprocess finished successfully. Now searching for output file...")
        print(f"Searching with parameters: input_path='{request.input_path}', output_path='{request.output_path}', weight='{request.weight}'")
        output_files = find_output_files(request.input_path, request.output_path, request.weight)
        print(f"find_output_files result: {output_files}")
        if output_files and output_files.get("main_result"):
            # --- START of the fix ---
            # Convert all found paths to absolute paths before returning
            output_files["main_result"] = os.path.abspath(output_files["main_result"])
            if "related_files" in output_files:
                output_files["related_files"] = [os.path.abspath(f) for f in output_files["related_files"]]
            if "cropped_faces" in output_files:
                output_files["cropped_faces"] = [os.path.abspath(f) for f in output_files["cropped_faces"]]
            # --- END of the fix ---
            return {
                "status": "success",
                "output_files": output_files,
                "message": "修复完成"
            }
        else:
            return {"error": "处理成功但未找到输出文件"}
    else:
        # 强制打印捕获到的输出，这是调试的关键
        print("--- Subprocess Failed ---")
        print(f"Return Code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        print("-------------------------")
        return {
            "error": "处理失败",
            "details": result.stderr,
            "stdout": result.stdout, # 添加 stdout 以便调试
            "returncode": result.returncode
        }


def find_output_files(input_path, output_path, weight):
    import glob
    import os

    # 获取输入文件的基本信息
    input_filename = os.path.basename(input_path)  # 例如: "face1.jpg"
    input_name = os.path.splitext(input_filename)[0]  # 例如: "face1"

    # 确定输出目录
    if not output_path:
        result_dir = f"results/test_img_{weight}"
    else:
        result_dir = output_path

    output_files = {}

    # 根据输入文件名构建可能的输出文件模式
    possible_patterns = [
        # 最终结果（整张修复后的图片）
        os.path.join(result_dir, "final_results", f"{input_name}.png"),
        os.path.join(result_dir, "final_results", f"{input_name}*.png"),

        # 修复后的人脸
        os.path.join(result_dir, "restored_faces", f"{input_name}.png"),
        os.path.join(result_dir, "restored_faces", f"{input_name}_*.png"),
        os.path.join(result_dir, "restored_faces", f"{input_name}*.png"),

        # 直接在该目录下
        os.path.join(result_dir, f"{input_name}.png"),
        os.path.join(result_dir, f"{input_name}*.png"),

        # 通用匹配（如果上述都没找到）
        os.path.join(result_dir, "final_results", "*.png"),
        os.path.join(result_dir, "restored_faces", "*.png"),
        os.path.join(result_dir, "*.png")
    ]

    # 按优先级查找文件
    for pattern in possible_patterns:
        files = glob.glob(pattern)
        if files:
            # 如果有多个匹配，优先选择最精确的匹配
            if len(files) > 1:
                # 优先选择与输入文件名完全匹配的文件
                exact_match = [f for f in files if os.path.basename(f).startswith(input_name)]
                if exact_match:
                    output_files["main_result"] = exact_match[0]
                else:
                    output_files["main_result"] = files[0]
            else:
                output_files["main_result"] = files[0]
            break

    # 如果找到了文件，也查找相关的其他文件
    if "main_result" in output_files:
        result_dir = os.path.dirname(output_files["main_result"])

        # 查找同一目录下的所有相关文件
        related_files = glob.glob(os.path.join(result_dir, f"{input_name}*"))
        output_files["related_files"] = related_files

        # 查找裁剪的人脸文件（如果有）
        cropped_faces_dir = os.path.join(os.path.dirname(result_dir), "cropped_faces")
        if os.path.exists(cropped_faces_dir):
            cropped_files = glob.glob(os.path.join(cropped_faces_dir, f"{input_name}*"))
            output_files["cropped_faces"] = cropped_files

    return output_files


@app.get("/result/{file_path:path}")
async def get_result(file_path: str):
    """
    获取处理结果文件
    """
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="image/png",
            filename=os.path.basename(file_path)
        )
    else:
        return {"error": "文件不存在"}


@app.get("/list_images")
async def list_images(directory: str = "inputs/whole_imgs"):
    import glob
    import os

    # 获取绝对路径
    abs_directory = os.path.abspath(directory)

    if not os.path.exists(abs_directory):
        return {"error": "目录不存在", "requested_path": directory, "absolute_path": abs_directory}

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    images = []

    for ext in image_extensions:
        pattern = os.path.join(abs_directory, ext)
        images.extend(glob.glob(pattern))

    return {
        "directory": directory,
        "absolute_directory": abs_directory,
        "images": [os.path.basename(img) for img in images],
        "full_paths": images,  # 这里已经是绝对路径了
        "absolute_paths": [os.path.abspath(img) for img in images]  # 显式返回绝对路径
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)