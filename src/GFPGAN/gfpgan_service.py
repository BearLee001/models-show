# gfpgan_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="GFPGAN Face Restoration Service")

class RestoreRequest(BaseModel):
    input_path: str
    output_path: str = "results"
    version: str = "1.4"
    upscale: int = 2
    weight: float = 0.5

@app.post("/restore")
async def restore_face(request: RestoreRequest):
    """
    Restores faces in an image using GFPGAN.
    - input_path: Path to the input image.
    - output_path: Directory for the output files.
    - version: GFPGAN model version.
    - upscale: The final upsampling scale.
    - weight: Adjustable weights.
    """
    logger.info(f"Received restore request: {request.model_dump_json(indent=2)}")

    if not os.path.isfile(request.input_path):
        raise HTTPException(status_code=404, detail=f"Input file not found at {request.input_path}")

    # Ensure the output directory exists
    os.makedirs(request.output_path, exist_ok=True)

    cmd_args = [
        "python", "inference_gfpgan.py",
        "-i", request.input_path,
        "-o", request.output_path,
        "-v", request.version,
        "-s", str(request.upscale),
        "-w", str(request.weight),
        "--bg_upsampler", "realesrgan"
    ]

    logger.info(f"Executing command: {' '.join(cmd_args)}")
    
    try:
        process = subprocess.run(
            cmd_args, 
            capture_output=True, 
            text=True, 
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        logger.info(f"Inference script stdout: {process.stdout}")
        
        # Find the main restored image
        input_basename = os.path.splitext(os.path.basename(request.input_path))[0]
        restored_img_dir = os.path.join(request.output_path, "restored_imgs")
        
        # Search for the output file
        possible_files = glob.glob(os.path.join(restored_img_dir, f"{input_basename}.*"))
        
        if possible_files:
            output_file = possible_files[0]
            return {
                "status": "success",
                "output_file": os.path.abspath(output_file),
                "message": "Face restoration complete."
            }
        else:
            logger.error(f"Could not find restored image for {input_basename} in {restored_img_dir}")
            raise HTTPException(status_code=500, detail="Processing succeeded but the output file was not found.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Inference script failed with return code {e.returncode}")
        logger.error(f"Stderr: {e.stderr}")
        logger.error(f"Stdout: {e.stdout}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Face restoration failed.",
                "details": e.stderr,
                "stdout": e.stdout
            }
        )

if __name__ == "__main__":
    import uvicorn
    # Running on port 8003
    uvicorn.run(app, host="0.0.0.0", port=8003)
