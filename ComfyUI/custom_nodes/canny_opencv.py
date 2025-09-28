import torch
import numpy as np
import cv2
import comfy.model_management
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from comfy_api.latest import io


class CannyOpenCV(ComfyNodeABC):
    """
    Canny edge detection using OpenCV (CPU-based).
    More efficient for CPU processing and provides consistent results.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "image": (IO.IMAGE, {
                    "tooltip": "Input image for edge detection"
                }),
                "low_threshold": (IO.FLOAT, {
                    "default": 50.0,
                    "min": 1.0,
                    "max": 255.0,
                    "step": 1.0,
                    "tooltip": "Lower threshold for edge detection"
                }),
                "high_threshold": (IO.FLOAT, {
                    "default": 150.0,
                    "min": 1.0,
                    "max": 255.0,
                    "step": 1.0,
                    "tooltip": "Upper threshold for edge detection"
                }),
            },
            "optional": {
                "blur_kernel_size": (IO.INT, {
                    "default": 5,
                    "min": 1,
                    "max": 15,
                    "step": 2,
                    "tooltip": "Gaussian blur kernel size (must be odd)"
                }),
                "l2_gradient": (IO.BOOLEAN, {
                    "default": False,
                    "tooltip": "Use L2 gradient for more accurate edge detection"
                }),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("canny_image",)
    FUNCTION = "detect_edges"
    CATEGORY = "image/preprocessors"
    DESCRIPTION = "Canny edge detection using OpenCV (CPU-based)"
    
    def detect_edges(self, image, low_threshold, high_threshold, blur_kernel_size=5, l2_gradient=False):
        """
        Detect edges using OpenCV Canny edge detection.
        
        Args:
            image: Input image tensor [B, H, W, C]
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
            blur_kernel_size: Gaussian blur kernel size
            l2_gradient: Whether to use L2 gradient
            
        Returns:
            canny_image: Edge-detected image tensor [B, H, W, C]
        """
        try:
            # Convert to CPU for OpenCV processing
            image_cpu = image.cpu()
            
            # Process each image in the batch
            results = []
            for i in range(image_cpu.shape[0]):
                # Get single image [H, W, C]
                single_image = image_cpu[i].numpy()
                
                # Convert to uint8 if needed
                if single_image.dtype != np.uint8:
                    single_image = (single_image * 255).astype(np.uint8)
                
                # Convert to grayscale if needed
                if len(single_image.shape) == 3 and single_image.shape[2] > 1:
                    gray = cv2.cvtColor(single_image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = single_image
                
                # Apply Gaussian blur
                if blur_kernel_size > 1:
                    # Ensure kernel size is odd
                    kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
                    gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
                
                # Apply Canny edge detection
                edges = cv2.Canny(
                    gray,
                    int(low_threshold),
                    int(high_threshold),
                    L2gradient=l2_gradient
                )
                
                # Convert back to RGB for consistency
                edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
                
                # Convert back to float32 and normalize
                edges_normalized = edges_rgb.astype(np.float32) / 255.0
                
                results.append(edges_normalized)
            
            # Stack results back into batch tensor
            result_tensor = torch.from_numpy(np.stack(results))
            
            # Move to appropriate device
            device = comfy.model_management.get_torch_device()
            result_tensor = result_tensor.to(device)
            
            return (result_tensor,)
            
        except Exception as e:
            raise RuntimeError(f"Error in Canny edge detection: {str(e)}")


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "CannyOpenCV": CannyOpenCV,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CannyOpenCV": "Canny Edge Detection (OpenCV)",
}
