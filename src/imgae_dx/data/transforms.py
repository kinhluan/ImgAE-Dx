"""
Image transforms for medical imaging data preprocessing.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Union


class MedicalImageTransforms:
    """
    Collection of transforms specifically designed for medical imaging.
    
    Includes normalization, augmentation, and preprocessing suitable for
    chest X-ray anomaly detection tasks.
    """
    
    @staticmethod
    def get_basic_transforms(
        image_size: int = 128,
        normalize: bool = True,
        mean: float = 0.485,
        std: float = 0.229
    ) -> transforms.Compose:
        """
        Get basic preprocessing transforms.
        
        Args:
            image_size: Target image size
            normalize: Whether to normalize pixel values
            mean: Normalization mean
            std: Normalization standard deviation
            
        Returns:
            Composed transforms
        """
        transform_list = [
            transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
        ]
        
        if normalize:
            transform_list.append(
                transforms.Normalize(mean=[mean], std=[std])
            )
        
        return transforms.Compose(transform_list)
    
    @staticmethod
    def get_training_transforms(
        image_size: int = 128,
        rotation_range: int = 15,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        horizontal_flip: bool = True,
        normalize: bool = True,
        mean: float = 0.485,
        std: float = 0.229
    ) -> transforms.Compose:
        """
        Get training transforms with augmentation.
        
        Args:
            image_size: Target image size
            rotation_range: Maximum rotation degrees
            brightness_range: Brightness adjustment range
            contrast_range: Contrast adjustment range
            horizontal_flip: Whether to apply horizontal flip
            normalize: Whether to normalize pixel values
            mean: Normalization mean
            std: Normalization standard deviation
            
        Returns:
            Composed transforms with augmentation
        """
        transform_list = [
            transforms.Resize((image_size, image_size), interpolation=Image.LANCZOS),
        ]
        
        # Add augmentation
        if rotation_range > 0:
            transform_list.append(
                transforms.RandomRotation(
                    degrees=rotation_range,
                    interpolation=Image.BILINEAR,
                    fill=0
                )
            )
        
        if horizontal_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        
        # Color adjustments
        if brightness_range != (1.0, 1.0) or contrast_range != (1.0, 1.0):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=brightness_range,
                    contrast=contrast_range
                )
            )
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalize
        if normalize:
            transform_list.append(
                transforms.Normalize(mean=[mean], std=[std])
            )
        
        return transforms.Compose(transform_list)
    
    @staticmethod
    def get_evaluation_transforms(
        image_size: int = 128,
        normalize: bool = True,
        mean: float = 0.485,
        std: float = 0.229
    ) -> transforms.Compose:
        """
        Get evaluation transforms (no augmentation).
        
        Args:
            image_size: Target image size
            normalize: Whether to normalize pixel values
            mean: Normalization mean
            std: Normalization standard deviation
            
        Returns:
            Composed transforms for evaluation
        """
        return MedicalImageTransforms.get_basic_transforms(
            image_size=image_size,
            normalize=normalize,
            mean=mean,
            std=std
        )


class ContrastLimitedAHE:
    """
    Contrast Limited Adaptive Histogram Equalization for medical images.
    """
    
    def __init__(self, clip_limit: float = 2.0, grid_size: int = 8):
        self.clip_limit = clip_limit
        self.grid_size = grid_size
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply CLAHE to PIL image."""
        try:
            import cv2
            
            # Convert PIL to numpy
            img_array = np.array(image)
            
            # Apply CLAHE
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit,
                tileGridSize=(self.grid_size, self.grid_size)
            )
            
            enhanced = clahe.apply(img_array)
            
            # Convert back to PIL
            return Image.fromarray(enhanced)
            
        except ImportError:
            # Fallback to simple histogram equalization
            return self._simple_histogram_equalization(image)
    
    def _simple_histogram_equalization(self, image: Image.Image) -> Image.Image:
        """Simple histogram equalization fallback."""
        img_array = np.array(image)
        
        # Calculate histogram
        hist, bins = np.histogram(img_array.flatten(), 256, [0, 256])
        
        # Calculate cumulative distribution
        cdf = hist.cumsum()
        cdf_normalized = cdf * 255 / cdf[-1]
        
        # Apply equalization
        equalized = np.interp(img_array.flatten(), bins[:-1], cdf_normalized)
        equalized = equalized.reshape(img_array.shape).astype(np.uint8)
        
        return Image.fromarray(equalized)


class WindowLevelTransform:
    """
    Window/Level transform commonly used in medical imaging.
    """
    
    def __init__(self, window_width: float, window_center: float):
        self.window_width = window_width
        self.window_center = window_center
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply window/level transform to tensor."""
        window_min = self.window_center - (self.window_width / 2)
        window_max = self.window_center + (self.window_width / 2)
        
        # Apply windowing
        windowed = torch.clamp(tensor, window_min, window_max)
        
        # Normalize to [0, 1]
        windowed = (windowed - window_min) / (window_max - window_min)
        
        return windowed


class GaussianNoise:
    """Add Gaussian noise for data augmentation."""
    
    def __init__(self, mean: float = 0.0, std: float = 0.01):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to tensor."""
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)


class RandomErasing:
    """
    Random erasing augmentation for medical images.
    Similar to cutout but with configurable fill value.
    """
    
    def __init__(
        self,
        probability: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        fill_value: float = 0.0
    ):
        self.probability = probability
        self.scale = scale
        self.ratio = ratio
        self.fill_value = fill_value
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply random erasing to tensor."""
        if torch.rand(1) > self.probability:
            return tensor
        
        _, height, width = tensor.shape
        area = height * width
        
        # Random scale and ratio
        target_area = area * torch.empty(1).uniform_(self.scale[0], self.scale[1])
        aspect_ratio = torch.empty(1).uniform_(self.ratio[0], self.ratio[1])
        
        h = int(round(torch.sqrt(target_area * aspect_ratio).item()))
        w = int(round(torch.sqrt(target_area / aspect_ratio).item()))
        
        if h < height and w < width:
            y = torch.randint(0, height - h + 1, (1,)).item()
            x = torch.randint(0, width - w + 1, (1,)).item()
            
            tensor[:, y:y+h, x:x+w] = self.fill_value
        
        return tensor


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: float = 0.485,
    std: float = 0.229
) -> torch.Tensor:
    """
    Denormalize a normalized tensor for visualization.
    
    Args:
        tensor: Normalized tensor
        mean: Normalization mean used
        std: Normalization std used
        
    Returns:
        Denormalized tensor
    """
    denormalized = tensor * std + mean
    return torch.clamp(denormalized, 0.0, 1.0)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert tensor to PIL Image.
    
    Args:
        tensor: Image tensor (C, H, W) or (H, W)
        
    Returns:
        PIL Image
    """
    # Handle different tensor shapes
    if tensor.dim() == 3:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # Remove channel dimension for grayscale
        elif tensor.shape[0] == 3:
            tensor = tensor  # Keep RGB
    
    # Convert to numpy
    array = tensor.detach().cpu().numpy()
    
    # Scale to [0, 255]
    if array.max() <= 1.0:
        array = (array * 255).astype(np.uint8)
    else:
        array = np.clip(array, 0, 255).astype(np.uint8)
    
    # Convert to PIL
    if array.ndim == 2:
        return Image.fromarray(array, mode='L')
    else:
        return Image.fromarray(array.transpose(1, 2, 0), mode='RGB')


def calculate_image_statistics(dataset_loader) -> dict:
    """
    Calculate mean and std statistics for a dataset.
    
    Args:
        dataset_loader: DataLoader for the dataset
        
    Returns:
        Dictionary with mean and std values
    """
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    for data, _ in dataset_loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    
    return {
        'mean': mean.tolist() if mean.numel() > 1 else mean.item(),
        'std': std.tolist() if std.numel() > 1 else std.item()
    }