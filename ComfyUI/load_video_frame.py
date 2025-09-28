import torch
import av
import numpy as np
from PIL import Image
import comfy.model_management
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from comfy_api.latest import io


class LoadVideoFrame(ComfyNodeABC):
    """
    Extract a specific frame from a video tensor as an image.
    Takes a video tensor [B, T, H, W, C] and extracts frame at specified index.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "video": (IO.VIDEO, {
                    "tooltip": "Video tensor [B, T, H, W, C]"
                }),
                "frame_index": (IO.INT, {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "step": 1,
                    "tooltip": "Frame index to extract (0 = first frame)"
                }),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_video_frame"
    CATEGORY = "image/loaders"
    DESCRIPTION = "Extract a specific frame from a video tensor as an image"
    
    def load_video_frame(self, video, frame_index):
        """
        Extract a specific frame from a video file.
        
        Args:
            video: VideoFromFile object
            frame_index: Index of the frame to extract (0-based)
            
        Returns:
            image: Extracted frame as image tensor [B, H, W, C]
        """
        try:
            # Get the video file source
            video_source = video.get_stream_source()
            
            # Open video file
            container = av.open(video_source)
            video_stream = container.streams.video[0]
            
            # Get total number of frames
            total_frames = video_stream.frames
            if total_frames is None:
                # If frame count is unknown, count them
                total_frames = sum(1 for _ in container.decode(video=0))
                container.close()
                container = av.open(video_source)
                video_stream = container.streams.video[0]
            
            # Validate frame index
            if frame_index >= total_frames:
                raise ValueError(f"Frame index {frame_index} is out of range. Video has {total_frames} frames.")
            
            # Seek to the desired frame
            container.seek(frame_index, stream=video_stream)
            
            # Decode the frame
            frame = None
            for frame in container.decode(video=0):
                break
            
            if frame is None:
                raise ValueError(f"Could not decode frame {frame_index}")
            
            # Convert to PIL Image
            pil_image = frame.to_image()
            
            # Convert to tensor
            image_array = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array).unsqueeze(0)  # Add batch dimension
            
            # Move to appropriate device
            device = comfy.model_management.get_torch_device()
            image_tensor = image_tensor.to(device)
            
            container.close()
            
            return (image_tensor,)
            
        except Exception as e:
            raise RuntimeError(f"Error extracting video frame: {str(e)}")
    


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "LoadVideoFrame": LoadVideoFrame,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadVideoFrame": "Load Video Frame",
}
