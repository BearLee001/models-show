import gradio as gr
import requests
import os
import shutil
import uuid
from typing import List, Optional

# Define the base URLs for the backend services
CODEFORMER_API_URL = "http://localhost:8001"
REFLDM_API_URL = "http://localhost:8002"
GFPGAN_API_URL = "http://localhost:8003"

# Create a temporary directory for uploads if it doesn't exist
TEMP_DIR = "gradio_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

def process_image(
    model: str,
    main_image: str,
    ref_images: Optional[List[str]],
    weight: float,
    cfg_scale: float,
    upscale: int,
    version: str,
):
    """
    Main function to process the image based on the selected model.
    """
    if not main_image:
        raise gr.Error("Please upload a main image.")

    # A unique path for this request's output
    output_dir = os.path.join(TEMP_DIR, str(uuid.uuid4()))
    os.makedirs(output_dir, exist_ok=True)

    try:
        if model == "CodeFormer":
            # --- CodeFormer Backend Call ---
            payload = {
                "input_path": main_image,
                "output_path": output_dir,
                "weight": weight,
            }
            response = requests.post(f"{CODEFORMER_API_URL}/restore", json=payload)
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "success":
                # Find the primary output file from the results
                output_files = result.get("output_files", {})
                if "main_result" in output_files:
                    return output_files["main_result"]
                else:
                    raise gr.Error("CodeFormer processing succeeded, but no main output file was found.")
            else:
                raise gr.Error(f"CodeFormer API Error: {result.get('error', 'Unknown error')}")


        elif model == "Ref-LDM":
            # --- Ref-LDM Backend Call ---
            if not ref_images:
                raise gr.Error("Ref-LDM requires at least one reference image.")
            
            # Convert TemporaryFileWrapper objects to file paths
            ref_image_paths = [img.name for img in ref_images]

            output_filename = os.path.join(output_dir, f"{uuid.uuid4()}.png")

            payload = {
                "lq_path": main_image,
                "ref_paths": ref_image_paths,
                "output_path": output_filename,
                "cfg_scale": cfg_scale,
            }
            response = requests.post(f"{REFLDM_API_URL}/generate", json=payload)
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "success":
                return result["output_file"]
            else:
                 raise gr.Error(f"Ref-LDM API Error: {result.get('detail', 'Unknown error')}")


        elif model == "GFPGAN":
            # --- GFPGAN Backend Call ---
            payload = {
                "input_path": main_image,
                "output_path": output_dir,
                "version": version,
                "upscale": upscale,
                "weight": weight,
            }
            response = requests.post(f"{GFPGAN_API_URL}/restore", json=payload)
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "success":
                return result["output_file"]
            else:
                raise gr.Error(f"GFPGAN API Error: {result.get('error', 'Unknown error')}")

        else:
            raise gr.Error(f"Unknown model: {model}")

    except requests.exceptions.RequestException as e:
        raise gr.Error(f"Failed to connect to the {model} backend. Is it running? Details: {e}")
    except Exception as e:
        # Catch any other unexpected errors
        raise gr.Error(f"An unexpected error occurred: {e}")


def on_model_change(model_choice: str):
    """
    Updates the visibility of UI components based on the selected model.
    """
    if model_choice == "Ref-LDM":
        return {
            ref_images_box: gr.update(visible=True),
            gfpgan_options: gr.update(visible=False),
            codeformer_options: gr.update(visible=False),
            refldm_options: gr.update(visible=True),
        }
    elif model_choice == "GFPGAN":
        return {
            ref_images_box: gr.update(visible=False),
            gfpgan_options: gr.update(visible=True),
            codeformer_options: gr.update(visible=False),
            refldm_options: gr.update(visible=False),
        }
    elif model_choice == "CodeFormer":
        return {
            ref_images_box: gr.update(visible=False),
            gfpgan_options: gr.update(visible=False),
            codeformer_options: gr.update(visible=True),
            refldm_options: gr.update(visible=False),
        }
    return {
        ref_images_box: gr.update(visible=False),
        gfpgan_options: gr.update(visible=False),
        codeformer_options: gr.update(visible=False),
        refldm_options: gr.update(visible=False),
    }

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Image Model Hub")
    gr.Markdown("Select a model, upload your image(s), and see the results.")

    with gr.Row():
        with gr.Column(scale=1):
            # Input Components
            model_selector = gr.Dropdown(
                ["CodeFormer", "Ref-LDM", "GFPGAN"],
                label="Choose a Model",
                value="CodeFormer",
            )
            input_image = gr.Image(type="filepath", label="Input Image")
            
            with gr.Box(visible=False) as ref_images_box:
                ref_images = gr.File(
                    label="Reference Images (for Ref-LDM)",
                    file_count="multiple",
                    file_types=["image"],
                )

            # Model-specific options
            with gr.Box(visible=True) as codeformer_options:
                gr.Markdown("### CodeFormer Options")
                cf_weight = gr.Slider(0, 1, value=0.5, label="Fidelity Weight (higher means stronger restoration)")
            
            with gr.Box(visible=False) as gfpgan_options:
                gr.Markdown("### GFPGAN Options")
                gfpgan_version = gr.Dropdown(["1.3", "1.4"], value="1.4", label="Model Version")
                gfpgan_upscale = gr.Slider(1, 4, value=2, step=1, label="Upscale Factor")
                gfpgan_weight = gr.Slider(0, 1, value=0.5, label="Fidelity Weight")

            with gr.Box(visible=False) as refldm_options:
                gr.Markdown("### Ref-LDM Options")
                refldm_cfg_scale = gr.Slider(1, 10, value=1.5, label="CFG Scale")

            submit_btn = gr.Button("Process Image", variant="primary")
        
        with gr.Column(scale=1):
            # Output Component
            output_image = gr.Image(type="filepath", label="Output Image")

    # Connect UI components to functions
    model_selector.change(
        fn=on_model_change,
        inputs=model_selector,
        outputs=[ref_images_box, gfpgan_options, codeformer_options, refldm_options],
    )

    submit_btn.click(
        fn=process_image,
        inputs=[
            model_selector,
            input_image,
            ref_images,
            cf_weight, # Corresponds to 'weight' param
            refldm_cfg_scale, # Corresponds to 'cfg_scale'
            gfpgan_upscale, # Corresponds to 'upscale'
            gfpgan_version, # Corresponds to 'version'
        ],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch(share=True)
