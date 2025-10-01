import os
import random
import pdb
import sys
import argparse
sys.path.insert(0, "ComfyUI")
from comfy.cli_args import args as comfy_args
from typing import Sequence, Mapping, Any, Union
from comfy.model_management import load_models_gpu, free_memory, unload_all_models
import torch
import gc
import time
import cv2
from PIL import Image
import numpy as np
import glob

def gaussian_blur_on_tensor(tensor: torch.Tensor, kernel_size: int = 25, sigma: float = 3.5) -> torch.Tensor:
    """
    Apply Gaussian blur to a tensor.
    """
    img = (tensor.cpu().numpy()*255.0).astype(np.uint8)[0]
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    img = img/255.0
    return torch.from_numpy(img).to(tensor.device).unsqueeze(0)

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directorty
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    sys.path.insert(0, find_path("ComfyUI"))
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    asyncio.run(init_extra_nodes())


add_comfyui_directory_to_sys_path()
add_extra_model_paths()
import_custom_nodes()


from nodes import NODE_CLASS_MAPPINGS

class VideoProcessor:
    """
    Efficient video processor that loads models once and reuses them for multiple inputs.
    """
    
    def __init__(self):
        """Initialize the processor with lazy loading."""
        self.models_loaded = False
        self.models = {}
        self.loaded_models = set()  # Track which models are currently loaded
        self._initialization_lock = False  # Prevent duplicate initialization
        # Don't load models immediately - load them when needed
    
    
    def _load_flux_models(self):
        """Load only Flux models when needed."""
        if 'flux_clip' not in self.loaded_models:
            print("Loading Flux CLIP models...")
            dual_clip_loader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
            self.models['flux_clip'] = dual_clip_loader.load_clip(
                clip_name1="t5xxl_fp8_e4m3fn_scaled.safetensors", 
                clip_name2="clip-vit-large-patch14/model.safetensors", 
                type="flux", 
                device="default"
            )
            self.loaded_models.add('flux_clip')
        
        if 'flux_unet' not in self.loaded_models:
            print("Loading Flux UNet model...")
            unet_loader_gguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            self.models['flux_unet'] = unet_loader_gguf.load_unet(unet_name="flux1-kontext-dev-Q8_0.gguf")
            self.loaded_models.add('flux_unet')
        
        if 'flux_vae' not in self.loaded_models:
            print("Loading Flux VAE model...")
            vae_loader = NODE_CLASS_MAPPINGS["VAELoader"]()
            self.models['flux_vae'] = vae_loader.load_vae(vae_name="split_files/vae/ae.safetensors")
            self.loaded_models.add('flux_vae')
        
        # Load utility models needed for Flux processing
        self._load_utility_models()
    
    def _load_wan_models(self):
        """Load only WAN models when needed."""
        if 'wan_clip' not in self.loaded_models:
            print("Loading WAN CLIP model...")
            clip_loader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
            self.models['wan_clip'] = clip_loader.load_clip(
                clip_name="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 
                type="wan", 
                device="default"
            )
            self.loaded_models.add('wan_clip')
        
        if 'wan_vae' not in self.loaded_models:
            print("Loading WAN VAE model...")
            vae_loader = NODE_CLASS_MAPPINGS["VAELoader"]()
            self.models['wan_vae'] = vae_loader.load_vae(vae_name="split_files/vae/wan_2.1_vae.safetensors")
            self.loaded_models.add('wan_vae')
    
    def _load_wan_high_noise_model(self):
        """Load WAN high noise model when needed."""
        if 'wan_unet_high_noise' not in self.loaded_models:
            print("Loading WAN high noise UNet model...")
            unet_loader_gguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            self.models['wan_unet_high_noise'] = unet_loader_gguf.load_unet(unet_name="HighNoise/Wan2.2-Fun-A14B-Control_HighNoise-Q8_0.gguf")
            self.loaded_models.add('wan_unet_high_noise')
        
        if 'wan_model_with_high_noise_lora' not in self.loaded_models:
            print("Loading WAN high noise LoRA...")
            lora_loader_model_only = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
            self.models['wan_model_with_high_noise_lora'] = lora_loader_model_only.load_lora_model_only(
                lora_name="split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", 
                strength_model=1, 
                model=get_value_at_index(self.models['wan_unet_high_noise'], 0)
            )
            self.loaded_models.add('wan_model_with_high_noise_lora')
    
    def _load_wan_low_noise_model(self):
        """Load WAN low noise model when needed."""
        if 'wan_unet_low_noise' not in self.loaded_models:
            print("Loading WAN low noise UNet model...")
            unet_loader_gguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            self.models['wan_unet_low_noise'] = unet_loader_gguf.load_unet(unet_name="LowNoise/Wan2.2-Fun-A14B-Control_LowNoise-Q8_0.gguf")
            self.loaded_models.add('wan_unet_low_noise')
        
        if 'wan_model_with_low_noise_lora' not in self.loaded_models:
            print("Loading WAN low noise LoRA...")
            lora_loader_model_only = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
            self.models['wan_model_with_low_noise_lora'] = lora_loader_model_only.load_lora_model_only(
                lora_name="split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", 
                strength_model=1, 
                model=get_value_at_index(self.models['wan_unet_low_noise'], 0)
            )
            self.loaded_models.add('wan_model_with_low_noise_lora')
    
    def _load_utility_models(self):
        """Load utility models that are needed for processing."""
        if 'utility_models' not in self.loaded_models:
            print("Loading utility models...")
            self.models['clip_text_encode'] = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            self.models['load_video'] = NODE_CLASS_MAPPINGS["LoadVideo"]()
            self.models['load_video_frame'] = NODE_CLASS_MAPPINGS["LoadVideoFrame"]()
            self.models['flux_kontext_image_scale'] = NODE_CLASS_MAPPINGS["FluxKontextImageScale"]()
            self.models['vae_encode'] = NODE_CLASS_MAPPINGS["VAEEncode"]()
            self.models['get_image_size'] = NODE_CLASS_MAPPINGS["GetImageSize"]()
            self.models['model_sampling_flux'] = NODE_CLASS_MAPPINGS["ModelSamplingFlux"]()
            self.models['flux_guidance'] = NODE_CLASS_MAPPINGS["FluxGuidance"]()
            self.models['reference_latent_node'] = NODE_CLASS_MAPPINGS["ReferenceLatent"]()
            self.models['basic_guider'] = NODE_CLASS_MAPPINGS["BasicGuider"]()
            self.models['basic_scheduler'] = NODE_CLASS_MAPPINGS["BasicScheduler"]()
            self.models['empty_sd3_latent_image'] = NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
            self.models['random_noise'] = NODE_CLASS_MAPPINGS["RandomNoise"]()
            self.models['k_sampler_select'] = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
            self.models['sampler_custom_advanced'] = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
            self.models['vae_decode'] = NODE_CLASS_MAPPINGS["VAEDecode"]()
            self.models['get_video_components'] = NODE_CLASS_MAPPINGS["GetVideoComponents"]()
            self.models['intensity_depth_estimation'] = NODE_CLASS_MAPPINGS["IntensityDepthEstimation"]()
            self.models['canny_opencv'] = NODE_CLASS_MAPPINGS["CannyOpenCV"]()
            self.models['model_sampling_sd3'] = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
            self.models['wan_22_fun_control_to_video'] = NODE_CLASS_MAPPINGS["Wan22FunControlToVideo"]()
            self.models['k_sampler_advanced'] = NODE_CLASS_MAPPINGS["KSamplerAdvanced"]()
            self.models['create_video'] = NODE_CLASS_MAPPINGS["CreateVideo"]()
            self.loaded_models.add('utility_models')

    #def process_video(self, video_file_path: str, output_prefix: str = "video", 
    #                 positive_prompt: str = None, negative_prompt: str = None,
    #                 style_prompt: str = None, fps: int = 16, num_frames: int = 81, 
    #                 seed: int = -1, preprocess_option: str = "Canny"):
    #    """
    #    Process a single video file using lazy-loaded models.
    #    
    #    Args:
    #        video_file_path: Path to the input video file
    #        output_prefix: Prefix for the output video file
    #        positive_prompt: Custom positive prompt (uses default if None)
    #        negative_prompt: Custom negative prompt (uses default if None)
    #        style_prompt: Style prompt that will be combined with positive_prompt (optional)
    #        fps: Output video FPS (default: 16)
    #        num_frames: Number of frames to generate (default: 81)
    #        seed: Random seed for reproducible results (default: -1 for random)
    #        preprocess_option: Preprocessing method for control (default: "Canny")
    #    """
    #    # With lazy loading, models will be loaded on-demand during processing
    #    
    #    # Use default prompts if not provided
    #    if positive_prompt is None:
    #        positive_prompt = ("Turn it into a photorealistic picture as if it's from a movie. "
    #                          "Keep the original lane markers. A photorealistic video as if it's a clip from a movie. "
    #                          "A video of a quiet, empty urban street on a gloomy, raining day. "
    #                          "The road is wide and wet, with visible puddles and worn textures, "
    #                          "giving the impression of recent rain. Faint blue lane markings run down the center of the street. "
    #                          "On the right side, a row of low-rise brick apartment buildings with multiple windows "
    #                          "and external air conditioning units is visible. A line of tall, thin evergreen trees "
    #                          "is planted along the sidewalk beside street lamps. On the left side, a river or waterfront "
    #                          "area can be seen, lined with benches, trash bins, and small concrete barriers. "
    #                          "Beyond the water, a row of pale green trees fades into the misty, gray horizon. "
    #                          "The atmosphere feels damp and foggy, with reduced visibility and a muted color palette "
    #                          "dominated by grays and washed-out greens. The camera is fixed at street level, moving forward smoothly")
    #    
    #    # Combine style_prompt with main positive prompt if provided
    #    if style_prompt:
    #        positive_prompt = f"{positive_prompt}, {style_prompt}"
    #        print(f"Combined prompt: {positive_prompt}")
    #    
    #    if negative_prompt is None:
    #        negative_prompt = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    #                          "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，CG, game, cartoon, anime, "
    #                          "render, 渲染，游戏，卡通")
    #    
    #    return self._process_single_video(video_file_path, output_prefix, positive_prompt, negative_prompt, 
    #                                    preprocess_option, None, num_frames, fps, seed)
    
    def _process_single_video(self, video_file_path: str, output_prefix: str, 
                            positive_prompt: str, negative_prompt: str, style_prompt: str, preprocess_option: str = "Intensity",
							num_frames: int = 81, fps: int = 16, seed: int = -1):
        """Internal method to process a single video with ComfyUI-style memory management."""
        with torch.inference_mode():
            
            # =============================================================================
            # STEP 1: Load Flux Models and Encode Text Prompts
            # =============================================================================
            
            # Load only Flux models for this step (lazy loading)
            
            self._load_flux_models()
            flux_models = [get_value_at_index(self.models['flux_unet'], 0)]
            load_models_gpu(flux_models)
            print("Flux models loaded for text encoding and image generation")
            
            # Encode prompts for Flux model
            positive_prompt_for_flux = f"{style_prompt}. {positive_prompt}"

            flux_positive_conditioning = self.models['clip_text_encode'].encode(
                text=positive_prompt_for_flux, 
                clip=get_value_at_index(self.models['flux_clip'], 0)
            )

            # Load WAN models for text encoding (needed early in the process)
            self._load_wan_models()
            
            # Encode prompts for WAN model (will be used later)
            wan_positive_conditioning = self.models['clip_text_encode'].encode(
                text=positive_prompt, 
                clip=get_value_at_index(self.models['wan_clip'], 0)
            )
            
            wan_negative_conditioning = self.models['clip_text_encode'].encode(
                text=negative_prompt, 
                clip=get_value_at_index(self.models['wan_clip'], 0)
            )

            # =============================================================================
            # STEP 2: Load Video and Extract Frame
            # =============================================================================
            
            # Load input video
            input_video = self.models['load_video'].EXECUTE_NORMALIZED(file=video_file_path)

            # =============================================================================
            # STEP 3: Process Reference Image
            # =============================================================================
            
            # Extract first frame as reference
            reference_frame = self.models['load_video_frame'].load_video_frame(
                frame_index=0, 
                video=get_value_at_index(input_video, 0)
            )
            reference_frame = (gaussian_blur_on_tensor(reference_frame[0], kernel_size=25, sigma=3.5), )

            # Scale the reference frame for Flux model
            scaled_reference = self.models['flux_kontext_image_scale'].scale(
                image=get_value_at_index(reference_frame, 0)
            )

            # Encode reference image to latent space
            reference_latent = self.models['vae_encode'].encode(
                pixels=get_value_at_index(scaled_reference, 0), 
                vae=get_value_at_index(self.models['flux_vae'], 0)
            )

            # =============================================================================
            # STEP 4: Generate Reference Image with Flux
            # =============================================================================
            
            # Get image dimensions
            image_dimensions = self.models['get_image_size'].get_size(
                image=get_value_at_index(reference_frame, 0), 
                unique_id=1883388692125059625
            )

            # Configure Flux model sampling
            flux_model = self.models['model_sampling_flux'].patch(
                max_shift=1.15, 
                base_shift=0.5, 
                width=get_value_at_index(image_dimensions, 0), 
                height=get_value_at_index(image_dimensions, 1), 
                model=get_value_at_index(self.models['flux_unet'], 0)
            )

            # Add guidance to conditioning
            guided_conditioning = self.models['flux_guidance'].append(
                guidance=2.5, 
                conditioning=get_value_at_index(flux_positive_conditioning, 0)
            )

            # Create reference latent with conditioning
            reference_latent_with_conditioning = self.models['reference_latent_node'].append(
                conditioning=get_value_at_index(guided_conditioning, 0), 
                latent=get_value_at_index(reference_latent, 0)
            )

            # Set up sampling parameters
            guider = self.models['basic_guider'].get_guider(
                model=get_value_at_index(flux_model, 0), 
                conditioning=get_value_at_index(reference_latent_with_conditioning, 0)
            )

            sigmas = self.models['basic_scheduler'].get_sigmas(
                scheduler="simple", 
                steps=28, 
                denoise=1, 
                model=get_value_at_index(self.models['flux_unet'], 0)
            )

            # Generate empty latent image
            empty_latent = self.models['empty_sd3_latent_image'].generate(
                width=get_value_at_index(image_dimensions, 0), 
                height=get_value_at_index(image_dimensions, 1), 
                batch_size=get_value_at_index(image_dimensions, 2)
            )

            # Generate random noise
            noise = self.models['random_noise'].get_noise(noise_seed=seed if seed != -1 else random.randint(1, 2**64))

            # Select sampler
            sampler = self.models['k_sampler_select'].get_sampler(sampler_name="euler")

            # Sample the reference image
            sampled_latent = self.models['sampler_custom_advanced'].sample(
                noise=get_value_at_index(noise, 0), 
                guider=get_value_at_index(guider, 0), 
                sampler=get_value_at_index(sampler, 0), 
                sigmas=get_value_at_index(sigmas, 0), 
                latent_image=get_value_at_index(empty_latent, 0)
            )

            # Decode to get final reference image
            reference_image = self.models['vae_decode'].decode(
                samples=get_value_at_index(sampled_latent, 0), 
                vae=get_value_at_index(self.models['flux_vae'], 0)
            )

            # Save intermediate results
            self._save_intermediate_results(output_prefix, {
                'reference_image': reference_image,
            })

            # =============================================================================
            # STEP 5: Switch to WAN Models and Generate Video
            # =============================================================================
            
            # Force unload ALL models and clear memory
            #unload_all_models()
            #torch.cuda.empty_cache()
            #gc.collect()
            print("All models unloaded, memory cleared")
            
            # Wait a moment for memory to be freed
            time.sleep(2)
            
            # Check memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                print(f"Memory after cleanup - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
            
            # Load only the high noise model first (lazy loading)
            self._load_wan_models()  # Load WAN CLIP and VAE first
            self._load_wan_high_noise_model()  # Load high noise model
            wan_high_noise_model = [get_value_at_index(self.models['wan_model_with_high_noise_lora'], 0)]
            load_models_gpu(wan_high_noise_model)
            print("High noise WAN model loaded for first pass")
            
            # Get video components and estimate depth
            video_components = self.models['get_video_components'].EXECUTE_NORMALIZED(
                video=get_value_at_index(input_video, 0)
            )

            if preprocess_option == "Intensity":
                # Estimate depth from video for control
                depth_map = self.models['intensity_depth_estimation'].estimate_depth(
                    method="intensity", 
                    depth_range=1, 
                    normalize=True, 
                    blur_radius=1, 
                    image=get_value_at_index(video_components, 0)
                )
            elif preprocess_option == "Canny":
                depth_map = self.models['canny_opencv'].detect_edges(
                    image=get_value_at_index(video_components, 0),
                    low_threshold=50,
                    high_threshold=150,
                    blur_kernel_size=5,
                    l2_gradient=False
                )
            else:
                depth_map = (get_value_at_index(video_components, 0),)

            # Get dimensions for video generation
            video_dimensions = self.models['get_image_size'].get_size(
                image=get_value_at_index(depth_map, 0), 
                unique_id=10193800039993504008
            )

            # Configure WAN model for video generation (high noise model is already loaded)
            wan_model_high_noise = self.models['model_sampling_sd3'].patch(
                shift=8.000000000000002, 
                model=get_value_at_index(self.models['wan_model_with_high_noise_lora'], 0)
            )

            # Generate control video using WAN
            control_video = self.models['wan_22_fun_control_to_video'].EXECUTE_NORMALIZED(
                width=get_value_at_index(video_dimensions, 0), 
                height=get_value_at_index(video_dimensions, 1), 
                length=num_frames, 
                batch_size=1, 
                positive=get_value_at_index(wan_positive_conditioning, 0), 
                negative=get_value_at_index(wan_negative_conditioning, 0), 
                vae=get_value_at_index(self.models['wan_vae'], 0), 
                ref_image=get_value_at_index(reference_image, 0), 
                control_video=get_value_at_index(depth_map, 0)
            )


            # =============================================================================
            # STEP 6: First Sampling Pass with High Noise Model
            # =============================================================================
            
            # First sampling pass with high noise model (already loaded)
            first_pass_result = self.models['k_sampler_advanced'].sample(
                add_noise="enable", 
                noise_seed=seed if seed != -1 else random.randint(1, 2**64), 
                steps=4, 
                cfg=1, 
                sampler_name="euler", 
                scheduler="simple", 
                start_at_step=0, 
                end_at_step=2, 
                return_with_leftover_noise="enable", 
                model=get_value_at_index(wan_model_high_noise, 0), 
                positive=get_value_at_index(control_video, 0), 
                negative=get_value_at_index(control_video, 1), 
                latent_image=get_value_at_index(control_video, 2)
            )

            # =============================================================================
            # STEP 7: Switch to Low Noise Model for Second Pass
            # =============================================================================
            
            # Force unload high noise model and load low noise model
            #unload_all_models()
            #torch.cuda.empty_cache()
            #gc.collect()
            print("High noise model unloaded, loading low noise model...")
            
            # Wait for memory to be freed
            time.sleep(1)
            
            # Check memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                print(f"Memory before loading low noise - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
            
            # Load low noise model (lazy loading)
            self._load_wan_low_noise_model()
            wan_low_noise_model = [get_value_at_index(self.models['wan_model_with_low_noise_lora'], 0)]
            load_models_gpu(wan_low_noise_model)
            print("Low noise WAN model loaded for second pass")
            
            # Configure low noise model
            wan_model_low_noise = self.models['model_sampling_sd3'].patch(
                shift=8.000000000000002, 
                model=get_value_at_index(self.models['wan_model_with_low_noise_lora'], 0)
            )

            # Second sampling pass with low noise model
            second_pass_result = self.models['k_sampler_advanced'].sample(
                add_noise="disable", 
                noise_seed=seed if seed != -1 else random.randint(1, 2**64), 
                steps=4, 
                cfg=1, 
                sampler_name="euler", 
                scheduler="simple", 
                start_at_step=2, 
                end_at_step=4, 
                return_with_leftover_noise="disable", 
                model=get_value_at_index(wan_model_low_noise, 0), 
                positive=get_value_at_index(control_video, 0), 
                negative=get_value_at_index(control_video, 1), 
                latent_image=get_value_at_index(first_pass_result, 0)
            )

            # Decode final video
            final_video_latent = self.models['vae_decode'].decode(
                samples=get_value_at_index(second_pass_result, 0), 
                vae=get_value_at_index(self.models['wan_vae'], 0)
            )

            # =============================================================================
            # STEP 7: Create and Save Final Video
            # =============================================================================
            
            # Create video from frames
            final_video = self.models['create_video'].EXECUTE_NORMALIZED(
                fps=fps, 
                images=get_value_at_index(final_video_latent, 0)
            )

            # Save the video using Python
            video_data = get_value_at_index(final_video, 0)
            print(f"Final video data type: {type(video_data)}")

            # Create output directory
            output_dir = os.path.dirname(output_prefix)
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename
            output_filename = f"{os.path.basename(output_prefix)}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            video_data.save_to(output_path)
            
            print(f"Video processing completed for: {output_path}")
            
            # =============================================================================
            # STEP 8: Final Cleanup
            # =============================================================================
            
            # Unload all models and cleanup
            #free_memory(0, torch.device("cuda"))
            #torch.cuda.empty_cache()
            #print("All models unloaded and memory cleaned up")
            
            return output_path

    def _save_intermediate_results(self, output_prefix: str, intermediates: dict):
        """Save intermediate results for debugging and analysis."""
        
        # Create intermediates directory
        base_dir = os.path.dirname(output_prefix)
        intermediates_dir = os.path.join(base_dir, "intermediates")
        os.makedirs(intermediates_dir, exist_ok=True)
        
        # Extract base filename
        base_name = os.path.basename(output_prefix)
        
        for name, data in intermediates.items():
            try:
                if name == 'reference_image':
                    # Save Flux-generated reference image
                    ref_data = get_value_at_index(data, 0)
                    if hasattr(ref_data, 'numpy'):
                        img_array = ref_data.numpy()
                        if img_array.ndim == 4:
                            img_array = img_array[0]
                        img = Image.fromarray((img_array * 255).astype(np.uint8))
                        img.save(os.path.join(intermediates_dir, f"{base_name}_flux_reference.png"))
                
                print(f"Saved intermediate: {name}")
                
            except Exception as e:
                print(f"Failed to save intermediate {name}: {e}")
                continue

    def process_batch(self, video_files: list, output_prefixes: list = None, 
                     positive_prompts: list = None, negative_prompts: list = None,
                     style_prompt: str = None, fps: int = 16, num_frames: int = 81, 
                     seed: int = -1, preprocess_option: str = "Canny"):
        """
        Process multiple video files efficiently using lazy-loaded models.
        
        Args:
            video_files: List of video file paths
            output_prefixes: List of output prefixes (uses default if None)
            positive_prompts: List of positive prompts (uses default if None)
            negative_prompts: List of negative prompts (uses default if None)
            style_prompt: Style prompt that will be combined with all positive prompts (optional)
            fps: Output video FPS (default: 16)
            num_frames: Number of frames to generate (default: 81)
            seed: Random seed for reproducible results (default: -1 for random)
            preprocess_option: Preprocessing method for control (default: "Canny")
        """
        # With lazy loading, we don't need to check models_loaded
        # Models will be loaded on-demand during processing
        
        results = []
        for i, video_file in enumerate(video_files):
            print(f"Processing video {i+1}/{len(video_files)}: {video_file}")
            
            # Use provided values or defaults
            output_prefix = output_prefixes[i] if output_prefixes and i < len(output_prefixes) else f"video/ComfyUI_{i}"
            positive_prompt = positive_prompts[i] if positive_prompts and i < len(positive_prompts) else None
            negative_prompt = negative_prompts[i] if negative_prompts and i < len(negative_prompts) else None
            
            try:
                result = processor._process_single_video(
                    video_file_path=video_file,
                    output_prefix=output_prefix,
                    positive_prompt=positive_prompt,
                    negative_prompt=negative_prompt,
                    style_prompt=style_prompt,
                    preprocess_option=args.preprocess,
                    num_frames=args.frames,
                    fps=args.fps,
                    seed=args.seed
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
                results.append(None)
        
        return results


def load_videos_and_prompts(directory_path: str):
    """
    Load videos and prompts from a directory.
    
    Args:
        directory_path: Path to directory containing .mp4 files and .txt files
        
    Returns:
        tuple: (video_files, positive_prompts) lists
    """
    
    # Find all mp4 files
    video_pattern = os.path.join(directory_path, "*.mp4")
    video_files = sorted(glob.glob(video_pattern))
    
    # Find corresponding txt files
    positive_prompts = []
    for video_file in video_files:
        # Get base name without extension
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        txt_file = os.path.join(directory_path, f"{base_name}.txt")
        
        if os.path.exists(txt_file):
            with open(txt_file, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
                positive_prompts.append(prompt)
        else:
            # Use default prompt if no txt file found
            positive_prompts.append("A beautiful video scene")
    
    return video_files, positive_prompts


def parse_arguments():
    """
    Parse command line arguments for the video processing script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Process videos with AI re-renderer and style transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single video file
  python wan22_style.py --input video.mp4 --output processed_video.mp4
  
  # Process all videos in a directory
  python wan22_style.py --input /path/to/videos/ --output /path/to/output/
  
  # Process with custom prompts
  python wan22_style.py --input video.mp4 --positive "A cinematic scene" --negative "blurry, low quality"
  
  # Process with style prompt
  python wan22_style.py --input video.mp4 --style-prompt "in the style of Van Gogh" --positive "A beautiful landscape"
  
  # Process directory with custom output directory
  python wan22_style.py --input /path/to/videos/ --output /path/to/output/ --batch
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        default="test/town04.mp4",
        help='Input video file or directory containing videos to process'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file or directory. For single file: specify output filename. For directory: specify output directory (default: video/)'
    )
    
    parser.add_argument(
        '--positive', '-p',
        default="A video of a wide, multi-lane highway in a mountainous region. The road curves gently to the right, with smooth asphalt and bright white dashed lane markings. A silver car drives slightly ahead in the left lane, with glowing blue tail lights. On the right side, a tall concrete barrier with a blue fence section lines the edge of the highway. Beyond it, a forest of tall evergreen trees rises against the base of mist-covered rocky mountains. Streetlights stand along the road, casting a faint industrial presence, though the ambient light comes mainly from the overcast sky. The air feels hazy, with muted visibility softening the distant trees and hills. The camera moves steadily forward",
        help='Positive prompt'
    )
    
    parser.add_argument(
        '--negative', '-n',
        help='Negative prompt',
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，CG, game, cartoon, anime, render, 渲染，游戏，卡通"
    )
    
    parser.add_argument(
        '--style-prompt', '-s',
        help='Style positive prompt that will be combined with the main positive prompt',
        default="Turn it into a photorealistic picture as if it's from a movie. Keep the original lane markers and number of lanes."
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=16,
        help='Output video FPS (default: 16)'
    )
    
    parser.add_argument(
        '--frames',
        type=int,
        default=81,
        help='Number of frames to generate (default: 81)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=-1,
        help='Random seed for reproducible results (default: -1 for random)'
    )
    
    parser.add_argument(
        '--preprocess',
        choices=['Canny', 'Intensity', 'None'],
        default='Intensity',
        help='Preprocessing method for control (default: Canny)'
    )
    
    return parser.parse_args()


def validate_input_path(input_path: str):
    """
    Validate the input path and determine if it's a file or directory.
    
    Args:
        input_path: Path to validate
        
    Returns:
        tuple: (is_file, is_directory, valid_path)
    """
    if not os.path.exists(input_path):
        return False, False, False
    
    is_file = os.path.isfile(input_path)
    is_directory = os.path.isdir(input_path)
    
    if is_file:
        # Check if it's a video file
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        if not any(input_path.lower().endswith(ext) for ext in video_extensions):
            print(f"Warning: {input_path} may not be a supported video format")
    
    return is_file, is_directory, True


def get_output_paths(input_path: str, output_arg: str, is_directory: bool):
    """
    Determine output paths based on input and output arguments.
    
    Args:
        input_path: Input file or directory path
        output_arg: Output argument from command line
        is_directory: Whether input is a directory
        
    Returns:
        tuple: (output_prefixes, output_dir)
    """
    if is_directory:
        # Processing directory
        if output_arg:
            output_dir = output_arg
        else:
            output_dir = "video"
        
        # Get all video files in directory
        video_files, _ = load_videos_and_prompts(input_path)
        output_prefixes = []
        
        for video_file in video_files:
            base_name = os.path.splitext(os.path.basename(video_file))[0]
            output_prefixes.append(os.path.join(output_dir, f"processed_{base_name}"))
        
        return output_prefixes, output_dir
    
    else:
        # Processing single file
        if output_arg:
            if os.path.isdir(output_arg):
                # Output is a directory, create filename
                base_name = os.path.splitext(os.path.basename(input_path))[0]
                output_file = os.path.join(output_arg, f"processed_{base_name}.mp4")
            else:
                # Output is a specific file
                output_file = output_arg
        else:
            # Default output
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_file = f"video/processed_{base_name}.mp4"
        
        # Create output directory
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        return [output_file], output_dir


if __name__ == "__main__":
    """
    Main entry point for command line video processing.
    """
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate input path
    is_file, is_directory, valid_path = validate_input_path(args.input)
    
    if not valid_path:
        print(f"Error: Input path '{args.input}' does not exist.")
        sys.exit(1)
    
    if not is_file and not is_directory:
        print(f"Error: Input path '{args.input}' is neither a file nor a directory.")
        sys.exit(1)
    
    # Initialize the processor
    print("Initializing Video Style Shaper...")
    processor = VideoProcessor()
    
    # Determine output paths
    output_prefixes, output_dir = get_output_paths(args.input, args.output, is_directory)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    args.input = os.path.abspath(args.input)
    
    if is_file:
        # Process single file
        print(f"Processing single video: {args.input}")
        print(f"Output will be saved to: {output_prefixes[0]}")
        
        # Use custom prompts if provided, otherwise use defaults
        positive_prompt = args.positive
        negative_prompt = args.negative
        style_prompt = args.style_prompt
        
        result = processor._process_single_video(
            video_file_path=args.input,
            output_prefix=output_prefixes[0],
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            style_prompt=style_prompt,
            preprocess_option=args.preprocess,
            num_frames=args.frames,
            fps=args.fps,
            seed=args.seed
        )
            
        if result:
            print(f"Successfully processed video: {result}")
        else:
            print("Video processing failed.")
            sys.exit(1)
    
    else:
        # Process directory
        print(f"Processing directory: {args.input}")
        print(f"Output directory: {output_dir}")
        
        # Load videos and prompts from directory
        video_files, positive_prompts = load_videos_and_prompts(args.input)
        
        if not video_files:
            print(f"No video files found in directory: {args.input}")
            sys.exit(1)
        
        print(f"Found {len(video_files)} videos to process:")
        for i, (video, prompt) in enumerate(zip(video_files, positive_prompts)):
            print(f"  {i+1}. {os.path.basename(video)}")
            if prompt != "A beautiful video scene":  # Only show custom prompts
                print(f"     Prompt: {prompt[:100]}...")
        
        if args.negative:
            negative_prompts = [args.negative] * len(video_files)
            print(f"Using custom negative prompt: {args.negative}")
        else:
            negative_prompts = None
        
        if args.style_prompt:
            print(f"Using style positive prompt: {args.style_prompt}")
        
        # Process all videos
        results = processor.process_batch(
            video_files=video_files,
            output_prefixes=output_prefixes,
            positive_prompts=positive_prompts,
            negative_prompts=negative_prompts,
            style_prompt=args.style_prompt,
            fps=args.fps,
            num_frames=args.frames,
            seed=args.seed,
            preprocess_option=args.preprocess
        )
        
        # Count successful results
        successful = sum(1 for r in results if r is not None)
        failed = len(results) - successful
        
        print(f"\nBatch processing completed!")
        print(f"Successfully processed: {successful} videos")
        if failed > 0:
            print(f"Failed: {failed} videos")
        print(f"Output directory: {output_dir}")
        print("Intermediate results saved to: video/intermediates/")
