"""
Unit tests for configuration management.
"""

import pytest
import json
import yaml
import tempfile
from pathlib import Path

from src.imgae_dx.utils import ConfigManager
from src.imgae_dx.utils.config_manager import (
    APIKeysConfig, StreamingConfig, ModelConfig, 
    TrainingConfig, DataConfig, ProjectConfig
)


@pytest.mark.unit
class TestAPIKeysConfig:
    """Test API keys configuration."""
    
    def test_api_keys_creation(self):
        """Test APIKeysConfig creation."""
        config = APIKeysConfig()
        assert config.kaggle_username == ""
        assert config.kaggle_key == ""
        assert config.wandb_key == ""
    
    def test_api_keys_validation(self):
        """Test API keys validation."""
        # Empty config should not validate
        config = APIKeysConfig()
        assert not config.validate()
        
        # Partial config should not validate
        config.kaggle_username = "test_user"
        assert not config.validate()
        
        # Complete config should validate
        config.kaggle_key = "test_key"
        config.wandb_key = "test_wandb_key"
        assert config.validate()
    
    def test_api_keys_from_files(self, temp_checkpoint_dir):
        """Test loading API keys from files."""
        # Create temporary config files
        kaggle_file = temp_checkpoint_dir / "kaggle.json"
        wandb_file = temp_checkpoint_dir / "wandb.json"
        
        # Create Kaggle config
        kaggle_data = {"username": "test_user", "key": "test_key"}
        with open(kaggle_file, 'w') as f:
            json.dump(kaggle_data, f)
        
        # Create W&B config
        wandb_data = {"api_key": "test_wandb_key"}
        with open(wandb_file, 'w') as f:
            json.dump(wandb_data, f)
        
        # Create configs directory
        configs_dir = temp_checkpoint_dir / "configs"
        configs_dir.mkdir()
        
        # Move files to configs directory
        (kaggle_file).rename(configs_dir / "kaggle.json")
        (wandb_file).rename(configs_dir / "wandb.json")
        
        # Test loading
        config = APIKeysConfig.from_files(temp_checkpoint_dir)
        assert config.kaggle_username == "test_user"
        assert config.kaggle_key == "test_key" 
        assert config.wandb_key == "test_wandb_key"
        assert config.validate()


@pytest.mark.unit
class TestStreamingConfig:
    """Test streaming configuration."""
    
    def test_streaming_config_defaults(self):
        """Test streaming config default values."""
        config = StreamingConfig()
        assert config.memory_limit_gb == 4
        assert config.batch_size == 32
        assert config.num_workers == 0
        assert config.prefetch_factor == 2
        assert isinstance(config.dataset_stages, list)
    
    def test_streaming_config_custom(self):
        """Test streaming config with custom values."""
        config = StreamingConfig(
            memory_limit_gb=8,
            batch_size=64,
            cache_size_mb=1024
        )
        assert config.memory_limit_gb == 8
        assert config.batch_size == 64
        assert config.cache_size_mb == 1024


@pytest.mark.unit
class TestModelConfig:
    """Test model configuration."""
    
    def test_model_config_defaults(self):
        """Test model config default values."""
        config = ModelConfig()
        assert config.input_channels == 1
        assert config.input_size == 128
        assert config.latent_dim == 512
        assert isinstance(config.unet_features, list)
        assert isinstance(config.rae_encoder_features, list)
    
    def test_model_config_custom(self):
        """Test model config with custom values."""
        config = ModelConfig(
            input_size=256,
            latent_dim=1024,
            unet_dropout=0.2
        )
        assert config.input_size == 256
        assert config.latent_dim == 1024
        assert config.unet_dropout == 0.2


@pytest.mark.unit  
class TestTrainingConfig:
    """Test training configuration."""
    
    def test_training_config_defaults(self):
        """Test training config default values."""
        config = TrainingConfig()
        assert config.epochs == 10
        assert config.learning_rate == 1e-4
        assert config.batch_size == 32
        assert config.optimizer == "adam"
        assert config.loss_function == "mse"
    
    def test_training_config_custom(self):
        """Test training config with custom values."""
        config = TrainingConfig(
            epochs=20,
            learning_rate=1e-3,
            optimizer="adamw",
            patience=15
        )
        assert config.epochs == 20
        assert config.learning_rate == 1e-3
        assert config.optimizer == "adamw"
        assert config.patience == 15


@pytest.mark.unit
class TestDataConfig:
    """Test data configuration."""
    
    def test_data_config_defaults(self):
        """Test data config default values."""
        config = DataConfig()
        assert config.num_normal_samples == 2000
        assert config.num_abnormal_samples == 1000
        assert config.image_size == 128
        assert config.enable_augmentation is True
        assert isinstance(config.brightness_range, tuple)
    
    def test_data_config_custom(self):
        """Test data config with custom values."""
        config = DataConfig(
            image_size=256,
            num_normal_samples=5000,
            enable_augmentation=False
        )
        assert config.image_size == 256
        assert config.num_normal_samples == 5000
        assert config.enable_augmentation is False


@pytest.mark.unit
class TestProjectConfig:
    """Test complete project configuration."""
    
    def test_project_config_creation(self):
        """Test project config creation."""
        config = ProjectConfig()
        assert hasattr(config, 'streaming')
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert hasattr(config, 'data')
        assert hasattr(config, 'api_keys')
    
    def test_project_config_from_yaml(self, temp_checkpoint_dir):
        """Test loading project config from YAML."""
        yaml_content = """
project_name: Test Project
version: 1.0.0

model:
  input_size: 256
  latent_dim: 1024

training:
  epochs: 15
  batch_size: 64

data:
  image_size: 256
  num_normal_samples: 3000

streaming:
  memory_limit_gb: 8
  batch_size: 64
"""
        
        yaml_file = temp_checkpoint_dir / "test_config.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        config = ProjectConfig.from_yaml(yaml_file)
        
        assert config.project_name == "Test Project"
        assert config.version == "1.0.0"
        assert config.model.input_size == 256
        assert config.model.latent_dim == 1024
        assert config.training.epochs == 15
        assert config.training.batch_size == 64
        assert config.data.image_size == 256
        assert config.streaming.memory_limit_gb == 8


@pytest.mark.unit
class TestConfigManager:
    """Test configuration manager."""
    
    def test_config_manager_creation(self):
        """Test config manager creation."""
        manager = ConfigManager()
        assert manager is not None
        assert manager.project_root is not None
    
    def test_device_detection(self, config_manager):
        """Test device detection."""
        device = config_manager.get_device()
        assert device in ["cpu", "cuda", "mps"]
    
    def test_colab_detection(self, config_manager):
        """Test Google Colab environment detection."""
        is_colab = config_manager.is_colab_environment()
        assert isinstance(is_colab, bool)
        # Should be False in test environment
        assert is_colab is False
    
    def test_config_loading_nonexistent_file(self, config_manager):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            config_manager.load_config("nonexistent.yaml")
    
    def test_config_validation_without_loading(self, config_manager):
        """Test validation without loading config."""
        # Should return False since no config is loaded
        assert config_manager.validate_config() is False
    
    def test_static_load_project_config(self, temp_checkpoint_dir):
        """Test static config loading method."""
        # Create minimal YAML config
        yaml_content = """
project_name: Static Test
model:
  input_size: 128
"""
        yaml_file = temp_checkpoint_dir / "static_test.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        config = ConfigManager.load_project_config(yaml_file)
        assert config.project_name == "Static Test"
        assert config.model.input_size == 128


@pytest.mark.unit
class TestConfigValidation:
    """Test configuration validation."""
    
    def test_invalid_yaml_format(self, temp_checkpoint_dir):
        """Test handling of invalid YAML format."""
        # Create invalid YAML
        invalid_yaml = temp_checkpoint_dir / "invalid.yaml"
        with open(invalid_yaml, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")
        
        with pytest.raises(yaml.YAMLError):
            ProjectConfig.from_yaml(invalid_yaml)
    
    def test_missing_config_sections(self, temp_checkpoint_dir):
        """Test handling of missing config sections."""
        # Create minimal YAML with missing sections
        minimal_yaml = temp_checkpoint_dir / "minimal.yaml"
        with open(minimal_yaml, 'w') as f:
            f.write("project_name: Minimal\n")
        
        config = ProjectConfig.from_yaml(minimal_yaml)
        
        # Should still have default values for missing sections
        assert hasattr(config, 'model')
        assert hasattr(config, 'training')
        assert config.project_name == "Minimal"
    
    def test_config_type_validation(self):
        """Test configuration type validation."""
        config = ModelConfig()
        
        # Test that types are correct
        assert isinstance(config.input_channels, int)
        assert isinstance(config.input_size, int)
        assert isinstance(config.latent_dim, int)
        assert isinstance(config.unet_dropout, float)
        assert isinstance(config.unet_features, list)


@pytest.mark.unit
class TestConfigEdgeCases:
    """Test edge cases in configuration management."""
    
    def test_empty_yaml_file(self, temp_checkpoint_dir):
        """Test handling of empty YAML file."""
        empty_yaml = temp_checkpoint_dir / "empty.yaml"
        empty_yaml.touch()  # Create empty file
        
        config = ProjectConfig.from_yaml(empty_yaml)
        
        # Should create config with defaults
        assert isinstance(config, ProjectConfig)
        assert hasattr(config, 'model')
    
    def test_config_with_extra_fields(self, temp_checkpoint_dir):
        """Test handling of YAML with extra unknown fields."""
        yaml_content = """
project_name: Extra Fields Test
unknown_field: some_value
another_unknown: 123

model:
  input_size: 128
  unknown_model_field: test
"""
        yaml_file = temp_checkpoint_dir / "extra_fields.yaml"
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        # Should load successfully, ignoring unknown fields
        config = ProjectConfig.from_yaml(yaml_file)
        assert config.project_name == "Extra Fields Test"
        assert config.model.input_size == 128
    
    def test_api_keys_with_missing_files(self, temp_checkpoint_dir):
        """Test API key loading with missing files."""
        # Test with non-existent configs directory
        config = APIKeysConfig.from_files(temp_checkpoint_dir)
        
        # Should create empty config
        assert config.kaggle_username == ""
        assert config.kaggle_key == ""
        assert config.wandb_key == ""
        assert not config.validate()