import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import comfy.model_management
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from comfy_api.latest import io


class IntensityDepthEstimation(ComfyNodeABC):
    """
    Simple intensity-based depth estimation node.
    Converts grayscale intensity to depth values using various methods.
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "image": (IO.IMAGE, {"tooltip": "Input image for depth estimation"}),
                "method": (["intensity", "inverted_intensity", "gradient", "sobel"], {
                    "default": "intensity",
                    "tooltip": "Depth estimation method"
                }),
                "depth_range": (IO.FLOAT, {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Maximum depth range"
                }),
                "normalize": (IO.BOOLEAN, {
                    "default": True,
                    "tooltip": "Normalize depth values to 0-1 range"
                }),
            },
            "optional": {
                "blur_radius": (IO.FLOAT, {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Gaussian blur radius for smoothing"
                }),
            }
        }
    
    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("depth_image",)
    FUNCTION = "estimate_depth"
    CATEGORY = "image/processing"
    DESCRIPTION = "Simple intensity-based depth estimation using various methods"
    
    def estimate_depth(self, image, method, depth_range, normalize, blur_radius=1.0):
        """
        Estimate depth from image intensity using various methods.
        
        Args:
            image: Input image tensor [B, H, W, C]
            method: Depth estimation method
            depth_range: Maximum depth range
            normalize: Whether to normalize output
            blur_radius: Gaussian blur radius for smoothing
            
        Returns:
            depth_image: Estimated depth map [B, H, W, C]
        """
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        
        # Convert to grayscale if needed
        if image.shape[-1] > 1:
            # Convert RGB to grayscale using standard weights
            gray = 0.299 * image[..., 0:1] + 0.587 * image[..., 1:2] + 0.114 * image[..., 2:3]
        else:
            gray = image
        
        # Apply Gaussian blur if specified
        if blur_radius > 0:
            gray = self._apply_gaussian_blur(gray, blur_radius)
        
        # Apply depth estimation method
        if method == "intensity":
            depth = gray
        elif method == "inverted_intensity":
            depth = 1.0 - gray
        elif method == "gradient":
            depth = self._compute_gradient_depth(gray)
        elif method == "sobel":
            depth = self._compute_sobel_depth(gray)
        else:
            depth = gray
        
        # Scale by depth range
        depth = depth * depth_range
        
        # Normalize if requested
        if normalize:
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max > depth_min:
                depth = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth = torch.zeros_like(depth)
        
        # Convert back to RGB for visualization
        depth_rgb = depth.repeat(1, 1, 1, 3)
        
        # Move to intermediate device
        depth_rgb = depth_rgb.to(comfy.model_management.intermediate_device())
        
        return (depth_rgb,)
    
    def _apply_gaussian_blur(self, image, radius):
        """Apply Gaussian blur to the image."""
        if radius <= 0:
            return image
        
        # Create Gaussian kernel
        kernel_size = int(2 * radius + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Generate Gaussian kernel
        sigma = radius / 3.0
        x = torch.arange(kernel_size, dtype=torch.float32, device=image.device)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Create 2D kernel
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        
        # Reshape for convolution
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        
        # Apply convolution
        batch_size, height, width, channels = image.shape
        image_reshaped = image.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Pad image
        pad_size = kernel_size // 2
        padded = F.pad(image_reshaped, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        # Apply convolution
        blurred = F.conv2d(padded, kernel_2d, padding=0)
        
        # Reshape back
        blurred = blurred.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        return blurred
    
    def _compute_gradient_depth(self, image):
        """Compute depth using gradient magnitude."""
        # Compute gradients
        grad_x = torch.diff(image, dim=2, prepend=image[:, :, :1, :])
        grad_y = torch.diff(image, dim=1, prepend=image[:, :1, :, :])
        
        # Compute gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # Invert so edges have higher depth
        depth = 1.0 - grad_mag
        
        return depth
    
    def _compute_sobel_depth(self, image):
        """Compute depth using Sobel edge detection."""
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image.device)
        
        # Reshape kernels for convolution
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        # Reshape image for convolution
        batch_size, height, width, channels = image.shape
        image_reshaped = image.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        # Pad image
        padded = F.pad(image_reshaped, (1, 1, 1, 1), mode='reflect')
        
        # Apply Sobel filters
        grad_x = F.conv2d(padded, sobel_x, padding=0)
        grad_y = F.conv2d(padded, sobel_y, padding=0)
        
        # Compute gradient magnitude
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # Reshape back
        grad_mag = grad_mag.permute(0, 2, 3, 1)  # [B, H, W, C]
        
        # Invert so edges have higher depth
        depth = 1.0 - grad_mag
        
        return depth


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "IntensityDepthEstimation": IntensityDepthEstimation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IntensityDepthEstimation": "Intensity Depth Estimation",
}
