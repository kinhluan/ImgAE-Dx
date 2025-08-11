"""
Reversed Autoencoder (RA) architecture for medical image anomaly detection.

Based on the paper: "Towards Universal Unsupervised Anomaly Detection in Medical Imaging"
The RA uses asymmetric encoder-decoder architecture without skip connections to create
"pseudo-healthy" reconstructions for anomaly detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from .base import BaseAutoencoder


class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and activation."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.0,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class TransposeConvBlock(nn.Module):
    """Transposed convolution block for upsampling."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        dropout: float = 0.0,
        activation: str = 'relu'
    ):
        super().__init__()
        
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_transpose(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ReversedAutoencoder(BaseAutoencoder):
    """
    Reversed Autoencoder for medical image anomaly detection.
    
    Key characteristics:
    1. Asymmetric architecture (different encoder/decoder depths)
    2. NO skip connections (unlike U-Net)
    3. Designed to create "pseudo-healthy" reconstructions
    4. Anomalies show higher reconstruction error
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_size: int = 128,
        latent_dim: int = 512,
        encoder_features: List[int] = None,
        decoder_features: List[int] = None,
        dropout: float = 0.1,
        activation: str = 'relu',
        **kwargs
    ):
        super().__init__(input_channels, input_size, latent_dim, **kwargs)
        
        if encoder_features is None:
            encoder_features = [64, 128, 256, 512]
        if decoder_features is None:
            decoder_features = [256, 128, 64]
        
        self.encoder_features = encoder_features
        self.decoder_features = decoder_features
        self.dropout = dropout
        self.activation = activation
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Calculate bottleneck spatial dimensions
        # Encoder reduces spatial dims by 2^(len(encoder_features)-1) due to stride=2 in all but first layer
        self.reduction_factor = 2 ** (len(encoder_features) - 1)
        self.bottleneck_spatial = input_size // self.reduction_factor
        self.bottleneck_channels = encoder_features[-1]
        self.bottleneck_flat_size = (
            self.bottleneck_channels * 
            self.bottleneck_spatial * 
            self.bottleneck_spatial
        )
        
        # Latent space projection
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.bottleneck_flat_size, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, self.bottleneck_flat_size)
        )
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Output layer
        self.output_layer = nn.Conv2d(
            decoder_features[-1], 
            input_channels, 
            kernel_size=1,
            padding=0
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_encoder(self) -> nn.Sequential:
        """Build encoder layers."""
        layers = []
        
        # First layer (input channels -> first feature)
        layers.append(ConvBlock(
            self.input_channels, 
            self.encoder_features[0],
            dropout=self.dropout,
            activation=self.activation
        ))
        
        # Intermediate layers with downsampling
        for i in range(1, len(self.encoder_features)):
            # Downsampling convolution
            layers.append(ConvBlock(
                self.encoder_features[i-1],
                self.encoder_features[i],
                stride=2,
                dropout=self.dropout,
                activation=self.activation
            ))
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self) -> nn.Sequential:
        """Build decoder layers."""
        layers = []
        
        # Start from bottleneck channels
        current_channels = self.bottleneck_channels
        
        # Upsampling layers
        for i, out_channels in enumerate(self.decoder_features):
            if i < len(self.decoder_features) - 1:
                # Intermediate upsampling layers
                layers.append(TransposeConvBlock(
                    current_channels,
                    out_channels,
                    dropout=self.dropout,
                    activation=self.activation
                ))
            else:
                # Final upsampling layer (no dropout, ReLU activation)
                layers.append(TransposeConvBlock(
                    current_channels,
                    out_channels,
                    dropout=0.0,
                    activation=self.activation
                ))
            
            current_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
        # Pass through encoder
        x = self.encoder(x)
        
        # Flatten and project to latent space
        x_flat = x.view(x.size(0), -1)
        latent = self.encoder_fc(x_flat)
        
        return latent
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent space to output."""
        # Project from latent space and reshape
        x = self.decoder_fc(z)
        x = x.view(
            z.size(0),
            self.bottleneck_channels,
            self.bottleneck_spatial,
            self.bottleneck_spatial
        )
        
        # Pass through decoder
        x = self.decoder(x)
        
        # Final output layer
        x = self.output_layer(x)
        
        # Apply sigmoid for normalized output
        return torch.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder."""
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction
    
    def forward_with_latent(self, x: torch.Tensor) -> tuple:
        """Forward pass that also returns latent representation."""
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent
    
    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """Get intermediate feature maps for visualization."""
        features = {}
        
        # Encoder features
        current = x
        for i, layer in enumerate(self.encoder):
            current = layer(current)
            features[f'encoder_stage_{i+1}'] = current
        
        # Latent representation
        current_flat = current.view(current.size(0), -1)
        latent = self.encoder_fc(current_flat)
        features['latent'] = latent
        
        # Decoder features
        current = self.decoder_fc(latent)
        current = current.view(
            current.size(0),
            self.bottleneck_channels,
            self.bottleneck_spatial,
            self.bottleneck_spatial
        )
        features['decoder_input'] = current
        
        for i, layer in enumerate(self.decoder):
            current = layer(current)
            features[f'decoder_stage_{i+1}'] = current
        
        return features
    
    def compute_pseudo_healthy_score(
        self, 
        x: torch.Tensor, 
        threshold: float = 0.5
    ) -> dict:
        """
        Compute pseudo-healthy reconstruction score.
        
        The idea is that the RA learns to reconstruct "normal" patterns,
        so anomalous regions will have higher reconstruction error.
        """
        self.eval()
        with torch.no_grad():
            reconstruction = self.forward(x)
            
            # Compute reconstruction error
            error_map = torch.pow(x - reconstruction, 2)
            
            # Per-sample anomaly scores
            anomaly_scores = torch.mean(error_map, dim=(1, 2, 3))
            
            # Binary anomaly predictions
            predictions = anomaly_scores > threshold
            
            return {
                'anomaly_scores': anomaly_scores,
                'predictions': predictions,
                'error_map': error_map,
                'reconstruction': reconstruction
            }
    
    def get_model_info(self) -> dict:
        """Get detailed model information."""
        info = super().get_model_info()
        info.update({
            'architecture': 'Reversed Autoencoder',
            'encoder_features': self.encoder_features,
            'decoder_features': self.decoder_features,
            'dropout': self.dropout,
            'activation': self.activation,
            'bottleneck_size': f"{self.bottleneck_channels}x{self.bottleneck_spatial}x{self.bottleneck_spatial}",
            'skip_connections': False,
            'asymmetric': True,
        })
        return info
    
    def compare_with_baseline(
        self, 
        x: torch.Tensor, 
        baseline_model: 'BaseAutoencoder'
    ) -> dict:
        """Compare reconstruction with baseline model (e.g., U-Net)."""
        self.eval()
        baseline_model.eval()
        
        with torch.no_grad():
            # Get reconstructions
            ra_reconstruction = self.forward(x)
            baseline_reconstruction = baseline_model.forward(x)
            
            # Compute reconstruction errors
            ra_error = torch.pow(x - ra_reconstruction, 2)
            baseline_error = torch.pow(x - baseline_reconstruction, 2)
            
            # Compute scores
            ra_scores = torch.mean(ra_error, dim=(1, 2, 3))
            baseline_scores = torch.mean(baseline_error, dim=(1, 2, 3))
            
            return {
                'ra_scores': ra_scores,
                'baseline_scores': baseline_scores,
                'ra_reconstruction': ra_reconstruction,
                'baseline_reconstruction': baseline_reconstruction,
                'ra_error_map': ra_error,
                'baseline_error_map': baseline_error,
            }