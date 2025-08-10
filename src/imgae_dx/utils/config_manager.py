"""
Secure configuration management for ImgAE-Dx project.
Handles API keys, project settings, and environment configuration.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
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
        wandb_path = project_root / "configs" / "wandb.md"
        if wandb_path.exists():
            with open(wandb_path, 'r') as f:
                config.wandb_key = f.read().strip()
        
        return config
    
    def validate(self) -> bool:
        """Validate that all required keys are present"""
        return bool(self.kaggle_username and self.kaggle_key and self.wandb_key)


@dataclass 
class StreamingConfig:
    """Streaming-specific configuration"""
    memory_limit_gb: int = 4
    cache_frequently_accessed: bool = True
    cleanup_after_stage: bool = True
    parallel_download: bool = False
    temp_dir: str = "temp"


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    name: str
    input_size: tuple = (128, 128)
    channels: list = field(default_factory=list)
    skip_connections: bool = True
    
    
@dataclass
class ProjectConfig:
    """Main project configuration"""
    
    # Dataset settings
    dataset_name: str = "nih-chest-xray/data"
    dataset_stages: list = field(default_factory=lambda: [
        "images_001.zip", "images_002.zip", "images_003.zip"
    ])
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 1e-4
    device: str = "auto"
    
    # Streaming settings
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    
    # API keys (loaded separately)
    api_keys: Optional[APIKeysConfig] = None
    
    # Models configuration
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> 'ProjectConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Create base config
        config = cls()
        
        # Load basic settings
        if 'dataset' in config_data:
            dataset_cfg = config_data['dataset']
            config.dataset_name = dataset_cfg.get('name', config.dataset_name)
            config.dataset_stages = [
                stage for stage in dataset_cfg.get('stages', config.dataset_stages)
            ]
        
        if 'training' in config_data:
            training_cfg = config_data['training']
            config.batch_size = training_cfg.get('batch_size', config.batch_size)
            config.learning_rate = training_cfg.get('learning_rate', config.learning_rate)
            config.device = training_cfg.get('device', config.device)
            
            # Streaming config
            if 'streaming' in training_cfg:
                stream_cfg = training_cfg['streaming']
                config.streaming = StreamingConfig(
                    memory_limit_gb=stream_cfg.get('memory_limit_gb', 4),
                    cache_frequently_accessed=stream_cfg.get('cache_frequently_accessed', True),
                    cleanup_after_stage=stream_cfg.get('cleanup_after_stage', True),
                    parallel_download=stream_cfg.get('parallel_download', False)
                )
        
        # Load model configurations
        if 'models' in config_data:
            models_cfg = config_data['models']
            for model_name, model_data in models_cfg.items():
                config.models[model_name] = ModelConfig(
                    name=model_data.get('name', model_name),
                    channels=model_data.get('channels', []),
                    input_size=tuple(model_data.get('input_size', [128, 128])),
                    skip_connections=model_data.get('skip_connections', True)
                )
        
        return config


class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self._config: Optional[ProjectConfig] = None
        self._api_keys: Optional[APIKeysConfig] = None
    
    @property
    def config(self) -> ProjectConfig:
        """Get project configuration"""
        if self._config is None:
            config_path = self.project_root / "configs" / "project_config.yaml"
            self._config = ProjectConfig.from_yaml(config_path)
            
            # Load API keys
            self._config.api_keys = self.api_keys
            
        return self._config
    
    @property 
    def api_keys(self) -> APIKeysConfig:
        """Get API keys configuration"""
        if self._api_keys is None:
            self._api_keys = APIKeysConfig.from_files(self.project_root)
        return self._api_keys
    
    def setup_kaggle_auth(self) -> bool:
        """Setup Kaggle API authentication"""
        try:
            import kaggle
            
            # Set credentials from config
            os.environ['KAGGLE_USERNAME'] = self.api_keys.kaggle_username
            os.environ['KAGGLE_KEY'] = self.api_keys.kaggle_key
            
            # Validate credentials
            kaggle.api.authenticate()
            return True
            
        except Exception as e:
            print(f"Kaggle authentication failed: {e}")
            return False
    
    def setup_wandb_auth(self) -> bool:
        """Setup Weights & Biases authentication"""
        try:
            import wandb
            
            # Login with API key
            wandb.login(key=self.api_keys.wandb_key)
            return True
            
        except Exception as e:
            print(f"W&B authentication failed: {e}")
            return False
    
    def setup_colab_environment(self) -> Dict[str, Any]:
        """Setup Google Colab environment"""
        colab_info = {
            "kaggle_auth": False,
            "wandb_auth": False,
            "drive_mounted": False,
            "gpu_available": False
        }
        
        try:
            # Check if running in Colab
            import google.colab
            
            # Setup API authentications
            colab_info["kaggle_auth"] = self.setup_kaggle_auth()
            colab_info["wandb_auth"] = self.setup_wandb_auth()
            
            # Mount Google Drive
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                colab_info["drive_mounted"] = True
            except Exception as e:
                print(f"Drive mounting failed: {e}")
            
            # Check GPU availability
            import torch
            colab_info["gpu_available"] = torch.cuda.is_available()
            
        except ImportError:
            # Not running in Colab
            pass
            
        return colab_info
    
    def get_device(self) -> str:
        """Get optimal device for training"""
        import torch
        
        if self.config.device != "auto":
            return self.config.device
            
        # Auto-detect best device
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps" 
        else:
            return "cpu"
    
    def validate_setup(self) -> Dict[str, bool]:
        """Validate complete project setup"""
        validation = {
            "config_loaded": self._config is not None,
            "api_keys_valid": self.api_keys.validate(),
            "kaggle_auth": False,
            "wandb_auth": False,
        }
        
        # Test authentications
        validation["kaggle_auth"] = self.setup_kaggle_auth()
        validation["wandb_auth"] = self.setup_wandb_auth()
        
        return validation


# Global config manager instance
_config_manager = None

def get_config_manager(project_root: Optional[Path] = None) -> ConfigManager:
    """Get global config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(project_root)
    return _config_manager