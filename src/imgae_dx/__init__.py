"""
ImgAE-Dx: Medical Image Anomaly Detection using Autoencoder architectures.

This package provides implementations for comparing U-Net and Reversed Autoencoder
architectures for unsupervised anomaly detection in medical images.
"""

__version__ = "0.1.0"
__author__ = "Luan BHK"
__email__ = "luanbhk@example.com"

from .models import UNet, ReversedAutoencoder
from .utils import ConfigManager, load_config
from .data import StreamingNIHDataset
from .training import Trainer, Evaluator

# Factory functions for easy instantiation
def create_model(model_type: str, config: dict):
    """Factory function to create models."""
    if model_type.lower() == "unet":
        return UNet(**config)
    elif model_type.lower() == "reversed_ae":
        return ReversedAutoencoder(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_trainer(trainer_type: str, model, config: dict):
    """Factory function to create trainers."""
    if trainer_type.lower() == "streaming":
        from .training import StreamingTrainer
        return StreamingTrainer(model=model, config=config)
    else:
        return Trainer(model=model, config=config)

def load_config(config_path: str):
    """Load configuration from YAML file."""
    from .utils import ConfigManager
    return ConfigManager.load_project_config(config_path)

__all__ = [
    "UNet",
    "ReversedAutoencoder",
    "ConfigManager", 
    "StreamingNIHDataset",
    "Trainer",
    "Evaluator",
    "create_model",
    "create_trainer", 
    "load_config",
]