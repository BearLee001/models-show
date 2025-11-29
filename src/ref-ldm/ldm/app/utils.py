import os
from PIL import Image


def resize_images_to_square(input_dir, output_dir, size=512):
    """
    Resizes all images in the input directory to a square format and saves them
    to the output directory.

    The function first resizes the image so that its shorter side matches the
    target size, maintaining the aspect ratio. Then, it performs a center
    crop to make the image a square.

    Args:
        input_dir (str): The directory containing the original images.
        output_dir (str): The directory where the resized images will be saved.
        size (int): The target edge length of the square images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            try:
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)

                with Image.open(input_path) as img:
                    # Convert to RGB to handle formats like RGBA or P
                    img = img.convert('RGB')

                    # Calculate new dimensions to maintain aspect ratio
                    width, height = img.size
                    if width < height:
                        new_width = size
                        new_height = int(height * (size / width))
                    else:
                        new_height = size
                        new_width = int(width * (size / height))

                    # Resize based on the shorter side
                    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                    # Perform center crop
                    left = (new_width - size) / 2
                    top = (new_height - size) / 2
                    right = (new_width + size) / 2
                    bottom = (new_height + size) / 2

                    img_cropped = img_resized.crop((left, top, right, bottom))

                    img_cropped.save(output_path)
                    print(f"Successfully resized and saved {filename} to {output_path}")

            except Exception as e:
                print(f"Could not process {filename}. Reason: {e}")
