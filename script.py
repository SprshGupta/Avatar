import os
import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline
from safetensors.torch import load_file

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Image Enhancement using Stable Diffusion XL")
parser.add_argument("--prompt", type=str, required=True, help="Enhancement prompt for the model")
parser.add_argument("--image", type=str, required=True, help="Path to the input image")
args = parser.parse_args()

# Directories
output_dir = "./ops_data/"
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

# Load the model
model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

# Load the SafeTensor weights correctly
safetensor_path = "sd_xl_refiner_1.0_0.9vae.safetensors"
weights = load_file(safetensor_path)

if hasattr(pipe, "vae") and weights:
    pipe.vae.load_state_dict(weights, strict=False)
    print("Custom SafeTensor weights loaded successfully.")

# Use the provided prompt
prompt = args.prompt
input_image_path = args.image
output_image_path = os.path.join(output_dir, f"enhanced_{os.path.basename(input_image_path)}")

# Check if the file exists
if not os.path.exists(input_image_path):
    print(f"Error: File {input_image_path} does not exist.")
    exit(1)

# Load and resize input image using PIL
try:
    pil_image = Image.open(input_image_path).convert("RGB")
    pil_image = pil_image.resize((512, 512), Image.LANCZOS)
except Exception as e:
    print(f"Error loading image {input_image_path}: {e}.")
    exit(1)

# Process the image
output = pipe(
    prompt=prompt, 
    image=pil_image, 
    strength=0.15,
    guidance_scale=3.5
).images[0]

# Save the enhanced image
output.save(output_image_path)
print(f"Processed {input_image_path} -> {output_image_path}")

print("Image processed successfully!")