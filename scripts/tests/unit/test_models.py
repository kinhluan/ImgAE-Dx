"""Basic model tests."""

import torch
import pytest
from imgae_dx.models import UNet, ReversedAutoencoder


def test_unet_creation():
    """Test UNet model creation."""
    model = UNet(input_channels=1, input_size=64, latent_dim=256)
    assert model is not None
    assert model.count_parameters() > 0


def test_unet_forward():
    """Test UNet forward pass."""
    model = UNet(input_channels=1, input_size=64, latent_dim=256)
    x = torch.randn(2, 1, 64, 64)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == x.shape


def test_reversed_ae_creation():
    """Test Reversed Autoencoder creation."""
    model = ReversedAutoencoder(input_channels=1, input_size=64, latent_dim=256)
    assert model is not None
    assert model.count_parameters() > 0


def test_reversed_ae_forward():
    """Test Reversed AE forward pass."""
    model = ReversedAutoencoder(input_channels=1, input_size=64, latent_dim=256)
    x = torch.randn(2, 1, 64, 64)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == x.shape
