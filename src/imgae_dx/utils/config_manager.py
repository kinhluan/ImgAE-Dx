"""
Secure configuration management for ImgAE-Dx project.
Handles API keys, project settings, and environment configuration.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class APIKeysConfig:
    """API Keys configuration"""
    kaggle_username: str = ""
    kaggle_key: str = ""
    wandb_key: str = ""
    
    @classmethod
    def from_files(cls, project_root: Path) -> 'APIKeysConfig':
        """Load API keys from separate files"""
        config = cls()
        
        # Load Kaggle credentials
        kaggle_path = project_root / "configs" / "kaggle.json"
        if kaggle_path.exists():
            with open(kaggle_path, 'r') as f:
                kaggle_data = json.load(f)
                config.kaggle_username = kaggle_data.get("username", "")
                config.kaggle_key = kaggle_data.get("key", "")
        
        # Load W&B key
        wandb_path = project_root / "configs" / "wandb.json"
        if wandb_path.exists():
            with open(wandb_path, 'r') as f:
                wandb_data = json.load(f)
                config.wandb_key = wandb_data.get("api_key", "")
        
        return config
    
    def validate(self) -> bool:
        """Validate that all required keys are present"""
        return bool(self.kaggle_username and self.kaggle_key and self.wandb_key)


@dataclass 
class StreamingConfig:
    """Streaming-specific configuration"""
    memory_limit_gb: int = 4
    batch_size: int = 32
    num_workers: int = 0  # Single process for streaming
    prefetch_factor: int = 2
    cache_size_mb: int = 512
    
    # Kaggle dataset configuration
    dataset_name: str = "nih-chest-xray-dataset"
    dataset_stages: list = field(default_factory=lambda: ["images_001", "images_002", "images_003"])
    
    # Memory management
    cleanup_frequency: int = 100  # batches
    gc_threshold: float = 0.8  # memory usage threshold for garbage collection


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    input_channels: int = 1
    input_size: int = 128
    latent_dim: int = 512
    
    # U-Net specific
    unet_features: list = field(default_factory=lambda: [64, 128, 256, 512])
    unet_dropout: float = 0.1
    
    # Reversed AE specific
    rae_encoder_features: list = field(default_factory=lambda: [64, 128, 256])
    rae_decoder_features: list = field(default_factory=lambda: [256, 128, 64])
    rae_skip_connections: bool = False


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 32
    
    # Optimizer settings
    optimizer: str = "adam"
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    
    # Loss function
    loss_function: str = "mse"
    
    # Checkpointing
    save_every: int = 5
    checkpoint_dir: str = "checkpoints"
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Validation
    val_split: float = 0.2
    val_frequency: int = 1


@dataclass
class DataConfig:
    """Data processing configuration"""
    # Dataset parameters
    num_normal_samples: int = 2000
    num_abnormal_samples: int = 1000
    train_val_split: float = 0.8
    
    # Image preprocessing
    image_size: int = 128
    normalize_mean: float = 0.485
    normalize_std: float = 0.229
    
    # Data augmentation
    enable_augmentation: bool = True
    rotation_range: int = 15
    horizontal_flip: bool = True
    brightness_range: tuple = field(default_factory=lambda: (0.8, 1.2))
    
    # Streaming parameters
    streaming_enabled: bool = True
    memory_limit_gb: int = 4


@dataclass
class ProjectConfig:
    """Complete project configuration"""
    # API keys (loaded separately for security)
    api_keys: APIKeysConfig = field(default_factory=APIKeysConfig)
    
    # Configuration sections
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Project metadata
    project_name: str = "ImgAE-Dx"
    version: str = "0.1.0"
    experiment_name: str = "unet_vs_reversed_ae"
    
    # Logging
    wandb_project: str = "imgae-dx"
    log_level: str = "INFO"
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'ProjectConfig':
        """Load configuration from YAML file"""
        yaml_path = Path(yaml_path)
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Create config objects from dictionaries
        config = cls()
        
        if 'streaming' in data:
            config.streaming = StreamingConfig(**data['streaming'])
        if 'model' in data:
            config.model = ModelConfig(**data['model'])
        if 'training' in data:
            config.training = TrainingConfig(**data['training'])
        if 'data' in data:
            config.data = DataConfig(**data['data'])
            
        # Update project metadata
        for key in ['project_name', 'version', 'experiment_name', 'wandb_project', 'log_level']:
            if key in data:
                setattr(config, key, data[key])
        
        return config


class ConfigManager:
    """Central configuration management system"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self._config: Optional[ProjectConfig] = None
        self._api_keys: Optional[APIKeysConfig] = None
    
    @property
    def config(self) -> ProjectConfig:
        """Get the loaded project configuration"""
        if self._config is None:
            raise ValueError("Configuration not loaded. Call load_config() first.")
        return self._config
    
    @property  
    def api_keys(self) -> APIKeysConfig:
        """Get the loaded API keys"""
        if self._api_keys is None:
            self._api_keys = APIKeysConfig.from_files(self.project_root)
        return self._api_keys
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> ProjectConfig:
        """Load project configuration"""
        if config_path is None:
            config_path = self.project_root / "configs" / "project_config.yaml"
        
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load main config
        self._config = ProjectConfig.from_yaml(config_path)
        
        # Load API keys
        self._config.api_keys = self.api_keys
        
        return self._config
    
    def validate_config(self) -> bool:
        """Validate the loaded configuration"""
        if self._config is None:
            return False
        
        # Validate API keys
        if not self._config.api_keys.validate():
            print("Warning: Missing API keys. Some features may not work.")
        
        # Validate paths
        checkpoint_dir = Path(self._config.training.checkpoint_dir)
        if not checkpoint_dir.exists():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        return True
    
    def get_device(self) -> str:
        """Get the best available device for training"""
        try:
            import torch
            
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"
    
    def is_colab_environment(self) -> bool:
        """Check if running in Google Colab"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def setup_kaggle_api(self) -> bool:
        """Setup Kaggle API with credentials"""
        try:
            import kaggle
            
            # Set environment variables for Kaggle API
            os.environ['KAGGLE_USERNAME'] = self.api_keys.kaggle_username
            os.environ['KAGGLE_KEY'] = self.api_keys.kaggle_key
            
            # Authenticate
            kaggle.api.authenticate()
            return True
            
        except Exception as e:
            print(f"Failed to setup Kaggle API: {e}")
            return False
    
    def setup_wandb(self) -> bool:
        """Setup Weights & Biases with API key"""
        try:
            import wandb
            
            # Login with API key
            wandb.login(key=self.api_keys.wandb_key)
            return True
            
        except Exception as e:
            print(f"Failed to setup W&B: {e}")
            return False
    
    @staticmethod
    def load_project_config(config_path: Union[str, Path]) -> ProjectConfig:
        """Convenience method to load configuration"""
        manager = ConfigManager()
        return manager.load_config(config_path)