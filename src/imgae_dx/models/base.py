"""
Base autoencoder class for medical image anomaly detection.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class BaseAutoencoder(nn.Module, ABC):
    """Abstract base class for autoencoder models."""
    
    def __init__(
        self, 
        input_channels: int = 1,
        input_size: int = 128,
        latent_dim: int = 512,
        **kwargs
    ):
        super().__init__()
        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        
        # Model metadata
        self.model_name = self.__class__.__name__
        self.training_history: Dict[str, list] = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to output."""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction
    
    def compute_loss(
        self, 
        x: torch.Tensor, 
        reconstruction: torch.Tensor, 
        loss_fn: nn.Module = None
    ) -> torch.Tensor:
        """Compute reconstruction loss."""
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        return loss_fn(reconstruction, x)
    
    def compute_anomaly_score(
        self, 
        x: torch.Tensor, 
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """Compute anomaly score based on reconstruction error."""
        self.eval()
        with torch.no_grad():
            reconstruction = self.forward(x)
            
            # Compute per-pixel reconstruction error
            error = torch.pow(x - reconstruction, 2)
            
            if reduction == 'mean':
                return torch.mean(error, dim=(1, 2, 3))
            elif reduction == 'sum':
                return torch.sum(error, dim=(1, 2, 3))
            elif reduction == 'none':
                return error
            else:
                raise ValueError(f"Unknown reduction: {reduction}")
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent space representation."""
        self.eval()
        with torch.no_grad():
            return self.encode(x)
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information summary."""
        return {
            'model_name': self.model_name,
            'input_channels': self.input_channels,
            'input_size': self.input_size,
            'latent_dim': self.latent_dim,
            'total_parameters': self.count_parameters(),
            'device': next(self.parameters()).device,
        }
    
    def save_checkpoint(
        self, 
        filepath: str, 
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        loss: float = 0.0,
        **kwargs
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': self.get_model_info(),
            'epoch': epoch,
            'loss': loss,
            'training_history': self.training_history,
            **kwargs
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_checkpoint(
        cls, 
        filepath: str, 
        device: str = 'cpu',
        **model_kwargs
    ) -> Tuple['BaseAutoencoder', Dict[str, Any]]:
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model instance
        model_config = checkpoint.get('model_config', {})
        model_config.update(model_kwargs)
        
        # Remove non-constructor parameters
        constructor_args = {
            'input_channels': model_config.get('input_channels', 1),
            'input_size': model_config.get('input_size', 128),
            'latent_dim': model_config.get('latent_dim', 512),
        }
        
        model = cls(**constructor_args)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Restore training history
        if 'training_history' in checkpoint:
            model.training_history = checkpoint['training_history']
        
        return model, checkpoint
    
    def update_training_history(
        self, 
        epoch: int, 
        train_loss: float, 
        val_loss: Optional[float] = None
    ) -> None:
        """Update training history."""
        self.training_history['epoch'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        
        if val_loss is not None:
            if 'val_loss' not in self.training_history:
                self.training_history['val_loss'] = []
            self.training_history['val_loss'].append(val_loss)
    
    def __repr__(self) -> str:
        """String representation of the model."""
        info = self.get_model_info()
        return (
            f"{info['model_name']}(\n"
            f"  input_channels={info['input_channels']},\n"
            f"  input_size={info['input_size']},\n"
            f"  latent_dim={info['latent_dim']},\n"
            f"  total_parameters={info['total_parameters']:,}\n"
            f")"
        )