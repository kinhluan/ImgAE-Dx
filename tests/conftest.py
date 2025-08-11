"""
Pytest configuration and fixtures for ImgAE-Dx tests.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil

from src.imgae_dx.models import UNet, ReversedAutoencoder
from src.imgae_dx.utils import ConfigManager


@pytest.fixture
def device():
    """Get the best available device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    from src.imgae_dx.utils.config_manager import ProjectConfig
    
    config = ProjectConfig()
    config.model.input_channels = 1
    config.model.input_size = 64  # Smaller for faster testing
    config.model.latent_dim = 128
    config.training.batch_size = 4
    config.training.epochs = 2
    config.data.image_size = 64
    
    return config


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor for testing."""
    return torch.randn(1, 1, 64, 64)


@pytest.fixture
def sample_batch():
    """Create a sample batch of images for testing."""
    return torch.randn(4, 1, 64, 64), torch.zeros(4, dtype=torch.long)


@pytest.fixture
def unet_model(device):
    """Create a U-Net model for testing."""
    model = UNet(
        input_channels=1,
        input_size=64,
        latent_dim=128
    )
    model.to(device)
    return model


@pytest.fixture
def reversed_ae_model(device):
    """Create a Reversed Autoencoder model for testing."""
    model = ReversedAutoencoder(
        input_channels=1,
        input_size=64,
        latent_dim=128
    )
    model.to(device)
    return model


@pytest.fixture
def sample_pil_image():
    """Create a sample PIL image for testing."""
    return Image.new('L', (64, 64), color=128)


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def config_manager():
    """Create a config manager for testing."""
    return ConfigManager()


@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset for testing."""
    from torch.utils.data import TensorDataset
    
    # Create dummy images and labels
    images = torch.randn(100, 1, 64, 64)
    labels = torch.randint(0, 2, (100,))
    
    return TensorDataset(images, labels)


@pytest.fixture
def dummy_dataloader(dummy_dataset):
    """Create a dummy data loader for testing."""
    from torch.utils.data import DataLoader
    
    return DataLoader(dummy_dataset, batch_size=4, shuffle=False)


@pytest.fixture
def trained_model_checkpoint(unet_model, temp_checkpoint_dir):
    """Create a dummy trained model checkpoint."""
    checkpoint_path = temp_checkpoint_dir / "test_model.pth"
    
    # Create a minimal checkpoint
    checkpoint = {
        'model_state_dict': unet_model.state_dict(),
        'epoch': 5,
        'loss': 0.1,
        'model_info': unet_model.get_model_info()
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture(scope="session")
def test_data_dir():
    """Create test data directory."""
    test_dir = Path(__file__).parent / "fixtures"
    test_dir.mkdir(exist_ok=True)
    return test_dir


# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


# Skip GPU tests if CUDA not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle GPU requirements."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)