"""
Unit tests for model architectures.
"""

import pytest
import torch
import numpy as np

from src.imgae_dx.models import UNet, ReversedAutoencoder, BaseAutoencoder


@pytest.mark.unit
class TestBaseAutoencoder:
    """Test the base autoencoder class."""
    
    def test_abstract_methods(self):
        """Test that BaseAutoencoder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAutoencoder()
    
    def test_anomaly_score_computation(self, unet_model, sample_image_tensor, device):
        """Test anomaly score computation."""
        sample_image_tensor = sample_image_tensor.to(device)
        
        scores = unet_model.compute_anomaly_score(sample_image_tensor, reduction='mean')
        assert isinstance(scores, torch.Tensor)
        assert scores.shape == torch.Size([1])
        assert scores.item() >= 0
    
    def test_parameter_count(self, unet_model):
        """Test parameter counting."""
        param_count = unet_model.count_parameters()
        assert isinstance(param_count, int)
        assert param_count > 0
    
    def test_model_info(self, unet_model):
        """Test model information retrieval."""
        info = unet_model.get_model_info()
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'total_parameters' in info
        assert 'input_channels' in info


@pytest.mark.unit
class TestUNet:
    """Test the U-Net architecture."""
    
    def test_unet_creation(self):
        """Test U-Net model creation."""
        model = UNet(input_channels=1, input_size=64, latent_dim=128)
        assert isinstance(model, UNet)
        assert model.input_channels == 1
        assert model.input_size == 64
        assert model.latent_dim == 128
    
    def test_unet_forward_pass(self, unet_model, sample_batch, device):
        """Test U-Net forward pass."""
        images, _ = sample_batch
        images = images.to(device)
        
        with torch.no_grad():
            output = unet_model(images)
        
        assert output.shape == images.shape
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_unet_encode_decode(self, unet_model, sample_image_tensor, device):
        """Test U-Net encoding and decoding."""
        sample_image_tensor = sample_image_tensor.to(device)
        
        with torch.no_grad():
            # Test encoding
            latent = unet_model.encode(sample_image_tensor)
            assert latent.shape == torch.Size([1, 128])  # latent_dim = 128
            
            # Test decoding
            reconstruction = unet_model.decode(latent)
            assert reconstruction.shape == sample_image_tensor.shape
    
    def test_unet_forward_with_latent(self, unet_model, sample_image_tensor, device):
        """Test U-Net forward pass with latent representation."""
        sample_image_tensor = sample_image_tensor.to(device)
        
        with torch.no_grad():
            reconstruction, latent = unet_model.forward_with_latent(sample_image_tensor)
        
        assert reconstruction.shape == sample_image_tensor.shape
        assert latent.shape == torch.Size([1, 128])
    
    def test_unet_feature_maps(self, unet_model, sample_image_tensor, device):
        """Test U-Net feature map extraction."""
        sample_image_tensor = sample_image_tensor.to(device)
        
        with torch.no_grad():
            features = unet_model.get_feature_maps(sample_image_tensor)
        
        assert isinstance(features, dict)
        assert 'enc_1' in features
        assert 'latent' in features
    
    def test_unet_different_input_sizes(self):
        """Test U-Net with different input sizes."""
        for size in [32, 64, 128]:
            model = UNet(input_channels=1, input_size=size, latent_dim=256)
            x = torch.randn(2, 1, size, size)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == x.shape
    
    def test_unet_checkpoint_save_load(self, unet_model, temp_checkpoint_dir):
        """Test U-Net checkpoint saving and loading."""
        checkpoint_path = temp_checkpoint_dir / "unet_test.pth"
        
        # Save checkpoint
        unet_model.save_checkpoint(str(checkpoint_path), epoch=1, loss=0.5)
        assert checkpoint_path.exists()
        
        # Load checkpoint
        loaded_model, checkpoint = UNet.load_checkpoint(str(checkpoint_path))
        assert isinstance(loaded_model, UNet)
        assert checkpoint['epoch'] == 1
        assert checkpoint['loss'] == 0.5


@pytest.mark.unit
class TestReversedAutoencoder:
    """Test the Reversed Autoencoder architecture."""
    
    def test_reversed_ae_creation(self):
        """Test Reversed AE model creation."""
        model = ReversedAutoencoder(input_channels=1, input_size=64, latent_dim=128)
        assert isinstance(model, ReversedAutoencoder)
        assert model.input_channels == 1
        assert model.input_size == 64
        assert model.latent_dim == 128
    
    def test_reversed_ae_forward_pass(self, reversed_ae_model, sample_batch, device):
        """Test Reversed AE forward pass."""
        images, _ = sample_batch
        images = images.to(device)
        
        with torch.no_grad():
            output = reversed_ae_model(images)
        
        assert output.shape == images.shape
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_reversed_ae_encode_decode(self, reversed_ae_model, sample_image_tensor, device):
        """Test Reversed AE encoding and decoding."""
        sample_image_tensor = sample_image_tensor.to(device)
        
        with torch.no_grad():
            # Test encoding
            latent = reversed_ae_model.encode(sample_image_tensor)
            assert latent.shape == torch.Size([1, 128])
            
            # Test decoding
            reconstruction = reversed_ae_model.decode(latent)
            assert reconstruction.shape == sample_image_tensor.shape
    
    def test_reversed_ae_asymmetric_architecture(self, reversed_ae_model):
        """Test that Reversed AE has asymmetric encoder/decoder."""
        info = reversed_ae_model.get_model_info()
        assert info['asymmetric'] is True
        assert info['skip_connections'] is False
    
    def test_reversed_ae_pseudo_healthy_score(self, reversed_ae_model, sample_batch, device):
        """Test pseudo-healthy scoring functionality."""
        images, _ = sample_batch
        images = images.to(device)
        
        with torch.no_grad():
            results = reversed_ae_model.compute_pseudo_healthy_score(images)
        
        assert 'anomaly_scores' in results
        assert 'predictions' in results
        assert 'error_map' in results
        assert 'reconstruction' in results
        
        assert results['anomaly_scores'].shape[0] == images.shape[0]
    
    def test_reversed_ae_feature_maps(self, reversed_ae_model, sample_image_tensor, device):
        """Test Reversed AE feature map extraction."""
        sample_image_tensor = sample_image_tensor.to(device)
        
        with torch.no_grad():
            features = reversed_ae_model.get_feature_maps(sample_image_tensor)
        
        assert isinstance(features, dict)
        assert 'latent' in features
        assert len([k for k in features.keys() if 'encoder_stage' in k]) > 0
        assert len([k for k in features.keys() if 'decoder_stage' in k]) > 0


@pytest.mark.unit
class TestModelComparison:
    """Test model comparison functionality."""
    
    def test_model_output_consistency(self, unet_model, reversed_ae_model, sample_image_tensor, device):
        """Test that models produce consistent output shapes."""
        sample_image_tensor = sample_image_tensor.to(device)
        
        with torch.no_grad():
            unet_output = unet_model(sample_image_tensor)
            rae_output = reversed_ae_model(sample_image_tensor)
        
        assert unet_output.shape == rae_output.shape == sample_image_tensor.shape
    
    def test_model_parameter_differences(self, unet_model, reversed_ae_model):
        """Test that models have different parameter counts."""
        unet_params = unet_model.count_parameters()
        rae_params = reversed_ae_model.count_parameters()
        
        # They should be different due to different architectures
        assert unet_params != rae_params
        assert unet_params > 0
        assert rae_params > 0
    
    def test_compare_with_baseline(self, unet_model, reversed_ae_model, sample_image_tensor, device):
        """Test baseline comparison functionality."""
        sample_image_tensor = sample_image_tensor.to(device)
        
        with torch.no_grad():
            comparison = reversed_ae_model.compare_with_baseline(
                sample_image_tensor, unet_model
            )
        
        assert 'ra_scores' in comparison
        assert 'baseline_scores' in comparison
        assert 'ra_reconstruction' in comparison
        assert 'baseline_reconstruction' in comparison


@pytest.mark.unit
@pytest.mark.slow
class TestModelTraining:
    """Test model training capabilities."""
    
    def test_model_gradient_computation(self, unet_model, sample_batch, device):
        """Test that models can compute gradients."""
        images, _ = sample_batch
        images = images.to(device)
        
        # Forward pass
        output = unet_model(images)
        loss = torch.nn.MSELoss()(output, images)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        has_gradients = False
        for param in unet_model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients
    
    def test_model_optimizer_step(self, unet_model, sample_batch, device):
        """Test that model parameters can be updated."""
        images, _ = sample_batch
        images = images.to(device)
        
        optimizer = torch.optim.Adam(unet_model.parameters(), lr=1e-4)
        
        # Store initial parameter
        initial_param = next(unet_model.parameters()).clone()
        
        # Training step
        optimizer.zero_grad()
        output = unet_model(images)
        loss = torch.nn.MSELoss()(output, images)
        loss.backward()
        optimizer.step()
        
        # Check that parameters changed
        updated_param = next(unet_model.parameters())
        assert not torch.equal(initial_param, updated_param)


@pytest.mark.unit
class TestModelValidation:
    """Test model validation and error cases."""
    
    def test_invalid_input_shapes(self, unet_model, device):
        """Test model behavior with invalid input shapes."""
        unet_model.to(device)
        
        # Test with wrong number of channels
        wrong_channels = torch.randn(1, 3, 64, 64).to(device)
        
        with pytest.raises((RuntimeError, ValueError)):
            unet_model(wrong_channels)
    
    def test_model_device_consistency(self, unet_model, device):
        """Test that model and inputs are on the same device."""
        unet_model.to(device)
        
        # Input on different device should raise error
        if device != "cpu":
            cpu_input = torch.randn(1, 1, 64, 64)  # CPU tensor
            
            with pytest.raises(RuntimeError):
                unet_model(cpu_input)
    
    def test_model_eval_mode(self, unet_model, sample_image_tensor, device):
        """Test model behavior in evaluation mode."""
        sample_image_tensor = sample_image_tensor.to(device)
        
        # Set to eval mode
        unet_model.eval()
        
        with torch.no_grad():
            output1 = unet_model(sample_image_tensor)
            output2 = unet_model(sample_image_tensor)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2, atol=1e-6)