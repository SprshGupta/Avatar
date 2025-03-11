import gradio as gr
import os
import subprocess
import boto3
from urllib.parse import urlparse
import os
# Ensure required directories exist
input_dir = "./ins_data/"
driving_video_dir = "./driving_videos/"
output_image_dir = "./ops_data/"
output_video_dir = "./animations/"

os.makedirs(input_dir, exist_ok=True)
os.makedirs(driving_video_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_video_dir, exist_ok=True)

def download_from_s3(s3_uri, local_path):
    """Download file from S3 to local path"""
    parsed_url = urlparse(s3_uri)
    bucket_name = parsed_url.netloc
    key = parsed_url.path.lstrip('/')
    
    s3 = boto3.client('s3')
    s3.download_file(bucket_name, key, local_path)

def handle_input(file, s3_uri, save_dir):
    """Handle input file: Use uploaded file or download from S3"""
    if s3_uri:
        local_path = os.path.join(save_dir, os.path.basename(s3_uri))
        download_from_s3(s3_uri, local_path)
        return local_path
    elif file:
        local_path = os.path.join(save_dir, file.name)
        
        # 🔹 FIX: Ensure correct handling of `NamedString` objects
        if isinstance(file, str):  
            # If it's a string (NamedString object), treat it as a path
            return file  
        
        with open(local_path, "wb") as f:
            f.write(file.read())  # Correctly save the uploaded file
        
        return local_path
    return None

def run_pipeline(input_file, input_s3_uri, driving_file, driving_s3_uri, prompt):
    """Run image processing and inference"""

    input_path = handle_input(input_file, input_s3_uri, input_dir)
    if not input_path or not os.path.exists(input_path):
        return "Error: No input image provided or file not found.", None

    driving_path = handle_input(driving_file, driving_s3_uri, driving_video_dir)
    if not driving_path or not os.path.exists(driving_path):
        return "Error: No driving video provided or file not found.", None

    # Ensure prompt is not empty
    if not prompt:
        return "Error: Enhancement prompt cannot be empty.", None

    # Run the image processing script with the user-provided prompt
    subprocess.run(["python", "script.py", "--image", input_path, "--prompt", prompt], check=True)

    output_images = [f for f in os.listdir(output_image_dir) if f.startswith("enhanced_")]
    if not output_images:
        return "Error: No processed images found.", None
    
    output_image = os.path.join(output_image_dir, output_images[0])

    # Run the inference script with the added flag
    subprocess.run([
        "python", "./LivePortrait/inference.py", 
        "-s", output_image, "-d", driving_path, "--flag_crop_driving_video"
    ], check=True)

    output_videos = [f for f in os.listdir(output_video_dir) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
    if not output_videos:
        return "Error: Video generation failed.", None
    
    output_video = os.path.join(output_video_dir, output_videos[0])
################ To Delete Directorys #################
    # c1 = "sudo rm -rf ./animations/*"
    # c2 = "sudo rm -rf ./ops_data/*"
    # os.system(c1)
    # os.system(c2)
    # subprocess.run(c1, shell=True, check=True)
    # subprocess.run(c2, shell=True, check=True)

    return "Success: Video generated!", output_video

def gradio_interface(input_file, input_s3_uri, driving_file, driving_s3_uri, prompt):
    """Gradio interface function"""
    message, video_path = run_pipeline(input_file, input_s3_uri, driving_file, driving_s3_uri, prompt)

    if not video_path:
        return message, None, None
    
    return message, video_path, video_path
    # c1 = "sudo rm -rf ./animations/*"
    # c2 = "sudo rm -rf ./ops_data/*"
    # os.system(c1)
    # os.system(c2)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload Image (Optional)"),
        gr.Textbox(label="Or provide Image S3 URI (Optional)"),
        gr.File(label="Upload Driving Video (Optional)"),
        gr.Textbox(label="Or provide Driving Video S3 URI (Optional)"),
        gr.Textbox(label="Enter Enhancement Prompt", placeholder="Describe how to enhance the image")
    ],
    outputs=[
        gr.Textbox(label="Status"),
        gr.Video(label="Generated Video"),
        gr.File(label="Download Video")
    ],
    title="AI Avatar Generation",
    description="Upload an image (or provide an S3 URI), enter an enhancement prompt, and a driving video to generate an AI-driven animated video."
)

iface.launch(share=True)
