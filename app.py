import sys
sys.path.insert(0, "ComfyUI")
import os
import torch
import gc
import gradio as gr
from wan22_style import VideoProcessor
from huggingface_hub import hf_hub_download
import cv2
import shutil
import os


def convert_video_for_gradio(video_path):
    """
    Convert video to web-compatible format for Gradio display.
    This fixes the 'video not playable' issue with OpenCV-generated videos.
    """
    if video_path is None:
        return None
    
    try:
        
        # Create output in a Gradio-friendly location
        output_dir = "video"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        converted_path = os.path.join(output_dir, f"{base_name}_gradio_compatible.mp4")
        
        # Read the original video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return video_path  # Return original if can't process
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Try different codecs in order of preference
        codecs_to_try = ['mp4v', 'XVID', 'MJPG', 'X264']
        out = None
        
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(converted_path, fourcc, fps, (width, height))
                if out.isOpened():
                    print(f"Using codec: {codec}")
                    break
                else:
                    out.release()
                    out = None
            except:
                continue
        
        if out is None or not out.isOpened():
            print("No suitable codec found, copying original file")
            shutil.copy2(video_path, converted_path)
            return converted_path
        
        # Read and write frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        # Clean up
        cap.release()
        out.release()
        
        print(f"Video converted for Gradio: {converted_path}")
        return converted_path
        
    except Exception as e:
        print(f"Error converting video: {e}")
        # Fallback: just copy the original file
        try:
            output_dir = "video"
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            fallback_path = os.path.join(output_dir, f"{base_name}_fallback.mp4")
            shutil.copy2(video_path, fallback_path)
            return fallback_path
        except:
            return video_path  # Return original if all else fails

def video_output_wrapper(func):
    """
    Wrapper function that automatically converts video outputs to Gradio-compatible format.
    Use this to wrap any function that returns a video path.
    """
    def wrapper(*args, **kwargs):
        # Call the original function
        result = func(*args, **kwargs)
        
        # If result is a video path, convert it for Gradio
        if result is not None and isinstance(result, str) and result.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Converting output video for Gradio: {result}")
            converted_path = convert_video_for_gradio(result)
            return converted_path
        
        return result
    
    return wrapper

#@video_output_wrapper
#def generate_image(*args, **kwargs):
#    """
#    Example function that would return a video path.
#    Replace this with your actual video processing function.
#    """
#    # This is just a placeholder - replace with your actual processing
#    return "video/processed_rgbfast.mp4"
video_processor = VideoProcessor()

@video_output_wrapper
def process_video_wrapper(structure_video, prompt_video, prompt_negative, structure_option, prompt_style, num_frames, fps, seed):
    result = video_processor._process_single_video(
        video_file_path=structure_video,
        output_prefix="video/gradio_output",
        positive_prompt=prompt_video,
        negative_prompt=prompt_negative,
        preprocess_option=structure_option,
        flux_positive_prompt=prompt_style,
        num_frames=int(num_frames),
        fps=int(fps),
        seed=int(seed)
    )
    return result


if __name__ == "__main__":

    hf_hub_download(repo_id="Comfy-Org/Lumina_Image_2.0_Repackaged", filename="split_files/vae/ae.safetensors",local_dir="temp")
    os.system("mv temp/split_files/vae/ae.safetensors ComfyUI/models/vae/ae.safetensors")
    hf_hub_download(repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged", filename="split_files/vae/wan_2.1_vae.safetensors",local_dir="temp")
    os.system("mv temp/split_files/vae/wan_2.1_vae.safetensors ComfyUI/models/vae/wan_2.1_vae.safetensors")
    hf_hub_download(repo_id="openai/clip-vit-large-patch14", filename="model.safetensors",local_dir="temp")
    os.system("mv temp/model.safetensors ComfyUI/models/clip/clip-vit-large-patch14.safetensors")
    hf_hub_download(repo_id="comfyanonymous/flux_text_encoders", filename="t5xxl_fp8_e4m3fn_scaled.safetensors",local_dir="temp")
    os.system("mv temp/t5xxl_fp8_e4m3fn_scaled.safetensors ComfyUI/models/clip/t5xxl_fp8_e4m3fn_scaled.safetensors")
    hf_hub_download(repo_id="QuantStack/Wan2.2-Fun-A14B-Control-GGUF", filename="HighNoise/Wan2.2-Fun-A14B-Control_HighNoise-Q8_0.gguf",local_dir="temp")
    os.system("mv temp/HighNoise/Wan2.2-Fun-A14B-Control_HighNoise-Q8_0.gguf ComfyUI/models/unet/Wan2.2-Fun-A14B-Control_HighNoise-Q8_0.gguf")
    hf_hub_download(repo_id="QuantStack/Wan2.2-Fun-A14B-Control-GGUF", filename="LowNoise/Wan2.2-Fun-A14B-Control_LowNoise-Q8_0.gguf",local_dir="temp")
    os.system("mv temp/LowNoise/Wan2.2-Fun-A14B-Control_LowNoise-Q8_0.gguf ComfyUI/models/unet/Wan2.2-Fun-A14B-Control_LowNoise-Q8_0.gguf")
    hf_hub_download(repo_id="QuantStack/FLUX.1-Kontext-dev-GGUF", filename="flux1-kontext-dev-Q8_0.gguf",local_dir="temp")
    os.system("mv temp/flux1-kontext-dev-Q8_0.gguf ComfyUI/models/unet/flux1-kontext-dev-Q8_0.gguf")
    
    # Start your Gradio app
    with gr.Blocks() as app:
        # Add a title
        gr.Markdown("# Video Style Shaping with Flux and WAN2.2")
        gr.Markdown("## Click the convert button if getting the not playable error")

        with gr.Row():
            with gr.Column():
                # Add an input
                prompt_style = gr.Textbox(label="Style Prompt for Flux", placeholder="Enter your prompt here...")
                prompt_video = gr.Textbox(label="Content Prompt for WAN Video", placeholder="Enter your prompt here...")
                # Add a `Row` to include the groups side by side 
                with gr.Row():
                    # First group includes structure image and depth strength
                    with gr.Group():
                        structure_video = gr.Video(label="Structure Video (quality: 720 x 1280, decent quality: 480 x 832)")
                        convert_btn = gr.Button("Convert Video for Gradio", size="sm")
                        structure_option = gr.Radio(label="Preprocessing Option", choices=["Intensity", "Canny", "None"], value="Intensity")
                
                # Advanced Settings Section (collapsible)
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        with gr.Column():
                            # Video generation parameters
                            num_frames = gr.Slider(
                                label="Number of Frames", 
                                minimum=17, 
                                maximum=121, 
                                value=81, 
                                step=1,
                                info="Number of frames to generate"
                            )
                            fps = gr.Slider(
                                label="FPS", 
                                minimum=8, 
                                maximum=30, 
                                value=16, 
                                step=1,
                                info="Frames per second"
                            )
                            seed = gr.Number(
                                label="Seed", 
                                value=-1, 
                                precision=0,
                                info="Random seed (-1 for random)"
                            )
                            prompt_negative = gr.Textbox(
                                label="Negative Prompt for WAN Video", 
                                value="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，CG, game, cartoon, anime, render, 渲染，游戏，卡通",
                                lines=4
                            )
                
                # The generate button
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                # The output video
                output_video = gr.Video(label="Generated Video")

            # Convert video button (manual only to avoid infinite loops)
            convert_btn.click(
                fn=convert_video_for_gradio,
                inputs=[structure_video],
                outputs=[structure_video]
            )

            # When clicking the button, it will trigger the `generate_image` function, with the respective inputs
            # and the output an image
            generate_btn.click(
                fn=process_video_wrapper,
                inputs=[
                    structure_video,
                    prompt_video, 
                    prompt_negative,
                    structure_option,
                    prompt_style,
                    num_frames,
                    fps,
                    seed,
                ],
                outputs=[output_video]
            )
        app.launch(share=True)
