"""
U-Net architecture for medical image anomaly detection.

Based on the original U-Net paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
Modified for autoencoder-style reconstruction tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .base import BaseAutoencoder


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block with maxpool then double conv."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block with transposed conv and skip connections."""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True, dropout: float = 0.0):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle input size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(BaseAutoencoder):
    """
    U-Net autoencoder for medical image anomaly detection.
    
    The U-Net architecture with skip connections helps preserve spatial information
    during reconstruction, making it effective for detecting localized anomalies.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_size: int = 128,
        latent_dim: int = 512,
        features: List[int] = None,
        dropout: float = 0.1,
        bilinear: bool = True,
        **kwargs
    ):
        super().__init__(input_channels, input_size, latent_dim, **kwargs)
        
        if features is None:
            features = [64, 128, 256, 512]
        
        self.features = features
        self.dropout = dropout
        self.bilinear = bilinear
        self.n_classes = input_channels  # Output same number of channels as input
        
        # Encoder (downward path)
        self.inc = DoubleConv(input_channels, features[0], dropout)
        self.down1 = Down(features[0], features[1], dropout)
        self.down2 = Down(features[1], features[2], dropout)
        self.down3 = Down(features[2], features[3], dropout)
        
        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(features[3], features[3] * 2 // factor, dropout)
        
        # Latent space projection
        # Calculate bottleneck spatial size
        bottleneck_size = input_size // (2 ** 4)  # 4 downsampling layers
        bottleneck_features = features[3] * 2 // factor
        self.bottleneck_flat_size = bottleneck_features * bottleneck_size * bottleneck_size
        
        # Latent space layers
        self.encoder_fc = nn.Linear(self.bottleneck_flat_size, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, self.bottleneck_flat_size)
        
        # Decoder (upward path)
        self.up1 = Up(features[3] * 2, features[3] // factor, bilinear, dropout)
        self.up2 = Up(features[3], features[2] // factor, bilinear, dropout)
        self.up3 = Up(features[2], features[1] // factor, bilinear, dropout)
        self.up4 = Up(features[1], features[0], bilinear, dropout)
        
        # Output layer
        self.outc = nn.Conv2d(features[0], self.n_classes, kernel_size=1)
        
        # Store bottleneck dimensions for reshaping
        self.bottleneck_channels = bottleneck_features
        self.bottleneck_spatial = bottleneck_size
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        # Encoder path (store skip connections)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Flatten and project to latent space
        x5_flat = x5.view(x5.size(0), -1)
        latent = self.encoder_fc(x5_flat)
        
        return latent
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to output."""
        # This is a simplified decode that doesn't use skip connections
        # For full reconstruction with skip connections, use forward() method
        x5_flat = self.decoder_fc(z)
        x5 = x5_flat.view(
            z.size(0), 
            self.bottleneck_channels, 
            self.bottleneck_spatial, 
            self.bottleneck_spatial
        )
        
        # Note: Skip connections are not available in decode-only mode
        # This will produce lower quality reconstructions
        x = F.interpolate(x5, scale_factor=16, mode='bilinear', align_corners=True)
        x = self.outc(x)
        
        return torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass with skip connections."""
        # Encoder path (save skip connections)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Latent space bottleneck
        x5_flat = x5.view(x5.size(0), -1)
        latent = self.encoder_fc(x5_flat)
        x5_reconstructed_flat = self.decoder_fc(latent)
        x5_reconstructed = x5_reconstructed_flat.view(x5.shape)
        
        # Decoder path (with skip connections)
        x = self.up1(x5_reconstructed, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output layer
        logits = self.outc(x)
        return torch.sigmoid(logits)
    
    def forward_with_latent(self, x: torch.Tensor) -> tuple:
        """Forward pass that also returns latent representation."""
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Latent space
        x5_flat = x5.view(x5.size(0), -1)
        latent = self.encoder_fc(x5_flat)
        x5_reconstructed_flat = self.decoder_fc(latent)
        x5_reconstructed = x5_reconstructed_flat.view(x5.shape)
        
        # Decoder path
        x = self.up1(x5_reconstructed, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        reconstruction = torch.sigmoid(logits)
        
        return reconstruction, latent
    
    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """Get intermediate feature maps for visualization."""
        features = {}
        
        # Encoder
        x1 = self.inc(x)
        features['enc_1'] = x1
        
        x2 = self.down1(x1)
        features['enc_2'] = x2
        
        x3 = self.down2(x2)
        features['enc_3'] = x3
        
        x4 = self.down3(x3)
        features['enc_4'] = x4
        
        x5 = self.down4(x4)
        features['bottleneck'] = x5
        
        # Latent
        x5_flat = x5.view(x5.size(0), -1)
        latent = self.encoder_fc(x5_flat)
        features['latent'] = latent
        
        return features
    
    def get_model_info(self) -> dict:
        """Get detailed model information."""
        info = super().get_model_info()
        info.update({
            'architecture': 'U-Net',
            'features': self.features,
            'dropout': self.dropout,
            'bilinear': self.bilinear,
            'bottleneck_size': f"{self.bottleneck_channels}x{self.bottleneck_spatial}x{self.bottleneck_spatial}",
            'skip_connections': True,
        })
        return info