# ref_ldm_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Ref-LDM Image Generation Service")

class GenerateRequest(BaseModel):
    lq_path: str
    ref_paths: List[str]
    output_path: str = "result.png"
    cfg_scale: float = 1.5
    ddim_step: int = 50
    seed: int = 2024

@app.post("/generate")
async def generate_image(request: GenerateRequest):
    """
    Generate an image using the Ref-LDM model.
    - lq_path: Path to the low-quality input image.
    - ref_paths: A list of paths to the reference images.
    - output_path: Optional path for the output file.
    - cfg_scale: Optional configuration scale.
    - ddim_step: Optional DDIM steps.
    - seed: Optional random seed.
    """
    logger.info(f"Received generation request: {request.model_dump_json(indent=2)}")

    if not os.path.exists(request.lq_path):
        raise HTTPException(status_code=404, detail=f"Low-quality image not found at {request.lq_path}")
    
    for path in request.ref_paths:
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Reference image not found at {path}")

    # Ensure the output directory exists
    output_dir = os.path.dirname(request.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cmd_args = [
        "python", "inference.py",
        "--lq_path", request.lq_path,
        "--output_path", request.output_path,
        "--cfg_scale", str(request.cfg_scale),
        "--ddim_step", str(request.ddim_step),
        "--seed", str(request.seed),
        "--ref_paths", *request.ref_paths
    ]
    
    logger.info(f"Executing command: {' '.join(cmd_args)}")
    
    try:
        # Running the command from the correct directory
        process = subprocess.run(
            cmd_args, 
            capture_output=True, 
            text=True, 
            check=True,  # This will raise CalledProcessError if returncode is non-zero
            cwd=os.path.dirname(os.path.abspath(__file__)) # Ensure script runs in its own dir
        )
        logger.info(f"Inference script stdout: {process.stdout}")
        
        if os.path.exists(request.output_path):
            return {
                "status": "success",
                "output_file": os.path.abspath(request.output_path),
                "message": "Image generation complete."
            }
        else:
            logger.error("Processing succeeded but the output file was not found.")
            raise HTTPException(status_code=500, detail="Processing succeeded but the output file was not found.")
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Inference script failed with return code {e.returncode}")
        logger.error(f"Stderr: {e.stderr}")
        logger.error(f"Stdout: {e.stdout}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Image generation failed.",
                "details": e.stderr,
                "stdout": e.stdout
            }
        )

if __name__ == "__main__":
    import uvicorn
    # Running on port 8002 to avoid conflict with codeformer_service
    uvicorn.run(app, host="0.0.0.0", port=8002)
