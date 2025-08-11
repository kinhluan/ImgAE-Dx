"""
Configuration management command-line interface.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any

from ..utils import ConfigManager


def config_command():
    """Main configuration command entry point."""
    parser = argparse.ArgumentParser(
        description="Manage ImgAE-Dx configuration and validate setup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Configuration commands"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate configuration and API keys"
    )
    validate_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    # Show command
    show_parser = subparsers.add_parser(
        "show",
        help="Show current configuration"
    )
    show_parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    show_parser.add_argument(
        "--section",
        type=str,
        choices=["all", "model", "training", "data", "streaming"],
        default="all",
        help="Configuration section to show"
    )
    
    # Test command
    test_parser = subparsers.add_parser(
        "test",
        help="Test API connections"
    )
    test_parser.add_argument(
        "--kaggle",
        action="store_true",
        help="Test Kaggle API connection"
    )
    test_parser.add_argument(
        "--wandb",
        action="store_true",
        help="Test W&B API connection"
    )
    test_parser.add_argument(
        "--all",
        action="store_true",
        help="Test all API connections"
    )
    
    # Init command
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize configuration files"
    )
    init_parser.add_argument(
        "--output-dir",
        type=str,
        default="configs",
        help="Output directory for configuration files"
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing configuration files"
    )
    
    # Device command
    device_parser = subparsers.add_parser(
        "device",
        help="Show device information"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == "validate":
            validate_config(args)
        elif args.command == "show":
            show_config(args)
        elif args.command == "test":
            test_apis(args)
        elif args.command == "init":
            init_config(args)
        elif args.command == "device":
            show_device_info(args)
    except Exception as e:
        print(f"Command failed: {e}")
        sys.exit(1)


def validate_config(args):
    """Validate configuration and setup."""
    print("🔍 Validating ImgAE-Dx Configuration")
    print("=" * 40)
    
    config_manager = ConfigManager()
    
    # Load configuration
    if args.config:
        try:
            config = config_manager.load_config(args.config)
            print(f"✅ Configuration loaded from: {args.config}")
        except Exception as e:
            print(f"❌ Configuration loading failed: {e}")
            return
    else:
        print("ℹ️  Using default configuration")
        from ..utils.config_manager import ProjectConfig
        config = ProjectConfig()
    
    # Validate configuration structure
    validation_results = []
    
    # Check required sections
    required_sections = ['model', 'training', 'data', 'streaming']
    for section in required_sections:
        if hasattr(config, section):
            validation_results.append((f"Config section: {section}", True, None))
        else:
            validation_results.append((f"Config section: {section}", False, "Missing section"))
    
    # Validate API keys
    api_keys = config_manager.api_keys
    
    # Kaggle API validation
    if api_keys.kaggle_username and api_keys.kaggle_key:
        validation_results.append(("Kaggle API keys", True, None))
    else:
        validation_results.append(("Kaggle API keys", False, "Missing credentials"))
    
    # W&B API validation
    if api_keys.wandb_key:
        validation_results.append(("W&B API key", True, None))
    else:
        validation_results.append(("W&B API key", False, "Missing API key"))
    
    # Check device availability
    device = config_manager.get_device()
    validation_results.append((f"Device ({device})", True, None))
    
    # Check environment
    is_colab = config_manager.is_colab_environment()
    env_name = "Google Colab" if is_colab else "Local"
    validation_results.append((f"Environment: {env_name}", True, None))
    
    # Print results
    print("\nValidation Results:")
    print("-" * 20)
    
    all_passed = True
    for check, passed, error in validation_results:
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if error:
            print(f"   Error: {error}")
            all_passed = False
    
    # Summary
    print("\nValidation Summary:")
    if all_passed:
        print("🎉 All validations passed! Configuration is ready.")
    else:
        print("⚠️  Some validations failed. Please check the configuration.")
        print("\nNext steps:")
        print("1. Setup API keys: configs/kaggle.json, configs/wandb.json")
        print("2. Run: poetry run imgae-config test --all")
    
    print(f"\nConfiguration details:")
    print(f"  Project: {getattr(config, 'project_name', 'ImgAE-Dx')}")
    print(f"  Version: {getattr(config, 'version', '0.1.0')}")
    print(f"  Device: {device}")
    print(f"  Environment: {env_name}")


def show_config(args):
    """Show current configuration."""
    print("📋 ImgAE-Dx Configuration")
    print("=" * 30)
    
    config_manager = ConfigManager()
    
    # Load configuration
    if args.config:
        try:
            config = config_manager.load_config(args.config)
            print(f"Configuration from: {args.config}")
        except Exception as e:
            print(f"❌ Configuration loading failed: {e}")
            return
    else:
        print("Using default configuration")
        from ..utils.config_manager import ProjectConfig
        config = ProjectConfig()
    
    print()
    
    # Show requested sections
    if args.section == "all" or args.section == "model":
        print("🧠 Model Configuration:")
        if hasattr(config, 'model'):
            model_config = config.model
            print(f"  Input channels: {getattr(model_config, 'input_channels', 1)}")
            print(f"  Input size: {getattr(model_config, 'input_size', 128)}")
            print(f"  Latent dim: {getattr(model_config, 'latent_dim', 512)}")
        print()
    
    if args.section == "all" or args.section == "training":
        print("🏋️  Training Configuration:")
        if hasattr(config, 'training'):
            training_config = config.training
            print(f"  Epochs: {getattr(training_config, 'epochs', 10)}")
            print(f"  Batch size: {getattr(training_config, 'batch_size', 32)}")
            print(f"  Learning rate: {getattr(training_config, 'learning_rate', 1e-4)}")
            print(f"  Optimizer: {getattr(training_config, 'optimizer', 'adam')}")
        print()
    
    if args.section == "all" or args.section == "data":
        print("📊 Data Configuration:")
        if hasattr(config, 'data'):
            data_config = config.data
            print(f"  Image size: {getattr(data_config, 'image_size', 128)}")
            print(f"  Normal samples: {getattr(data_config, 'num_normal_samples', 2000)}")
            print(f"  Abnormal samples: {getattr(data_config, 'num_abnormal_samples', 1000)}")
            print(f"  Augmentation: {getattr(data_config, 'enable_augmentation', True)}")
        print()
    
    if args.section == "all" or args.section == "streaming":
        print("🌊 Streaming Configuration:")
        if hasattr(config, 'streaming'):
            streaming_config = config.streaming
            print(f"  Memory limit: {getattr(streaming_config, 'memory_limit_gb', 4)} GB")
            print(f"  Batch size: {getattr(streaming_config, 'batch_size', 32)}")
            print(f"  Cache size: {getattr(streaming_config, 'cache_size_mb', 512)} MB")
        print()
    
    # Show API keys status (without revealing actual keys)
    api_keys = config_manager.api_keys
    print("🔑 API Keys Status:")
    print(f"  Kaggle: {'✅ Configured' if api_keys.kaggle_username and api_keys.kaggle_key else '❌ Missing'}")
    print(f"  W&B: {'✅ Configured' if api_keys.wandb_key else '❌ Missing'}")


def test_apis(args):
    """Test API connections."""
    print("🔗 Testing API Connections")
    print("=" * 30)
    
    config_manager = ConfigManager()
    
    if args.all or args.kaggle:
        print("\n📦 Testing Kaggle API...")
        try:
            success = config_manager.setup_kaggle_api()
            if success:
                print("✅ Kaggle API connection successful")
                
                # Test data access
                try:
                    from ..streaming import KaggleStreamClient
                    client = KaggleStreamClient()
                    files = client.list_dataset_files()[:5]  # First 5 files
                    print(f"   Sample files: {files}")
                except Exception as e:
                    print(f"⚠️  Data access test failed: {e}")
            else:
                print("❌ Kaggle API connection failed")
        except Exception as e:
            print(f"❌ Kaggle API test failed: {e}")
    
    if args.all or args.wandb:
        print("\n📈 Testing W&B API...")
        try:
            success = config_manager.setup_wandb()
            if success:
                print("✅ W&B API connection successful")
            else:
                print("❌ W&B API connection failed")
        except Exception as e:
            print(f"❌ W&B API test failed: {e}")
    
    print("\nAPI test completed.")


def init_config(args):
    """Initialize configuration files."""
    print("🚀 Initializing ImgAE-Dx Configuration")
    print("=" * 40)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create project configuration
    config_path = output_dir / "project_config.yaml"
    
    if config_path.exists() and not args.force:
        print(f"⚠️  Configuration file exists: {config_path}")
        print("Use --force to overwrite")
    else:
        create_default_config(config_path)
        print(f"✅ Created: {config_path}")
    
    # Create API key templates
    templates = {
        "kaggle.json.template": {
            "username": "your-kaggle-username",
            "key": "your-kaggle-api-key"
        },
        "wandb.json.template": {
            "api_key": "your-wandb-api-key"
        }
    }
    
    for filename, content in templates.items():
        template_path = output_dir / filename
        
        if template_path.exists() and not args.force:
            print(f"⚠️  Template exists: {template_path}")
        else:
            with open(template_path, 'w') as f:
                json.dump(content, f, indent=2)
            print(f"✅ Created: {template_path}")
    
    print(f"\n📁 Configuration files created in: {output_dir}")
    print("\n📝 Next steps:")
    print(f"1. Copy templates to actual config files:")
    print(f"   cp {output_dir}/kaggle.json.template {output_dir}/kaggle.json")
    print(f"   cp {output_dir}/wandb.json.template {output_dir}/wandb.json")
    print("2. Edit the JSON files with your actual API keys")
    print("3. Run: poetry run imgae-config validate")


def create_default_config(config_path: Path):
    """Create default project configuration."""
    default_config = """# ImgAE-Dx Project Configuration

project_name: ImgAE-Dx
version: 0.1.0
experiment_name: unet_vs_reversed_ae
wandb_project: imgae-dx
log_level: INFO

# Model configuration
model:
  input_channels: 1
  input_size: 128
  latent_dim: 512
  
  # U-Net specific
  unet_features: [64, 128, 256, 512]
  unet_dropout: 0.1
  
  # Reversed AE specific
  rae_encoder_features: [64, 128, 256]
  rae_decoder_features: [256, 128, 64]
  rae_skip_connections: false

# Training configuration
training:
  epochs: 10
  learning_rate: 0.0001
  batch_size: 32
  optimizer: adam
  weight_decay: 0.00001
  scheduler: cosine
  
  # Loss function
  loss_function: mse
  
  # Checkpointing
  save_every: 5
  checkpoint_dir: checkpoints
  
  # Early stopping
  patience: 10
  min_delta: 0.0001
  
  # Validation
  val_split: 0.2
  val_frequency: 1

# Data configuration
data:
  num_normal_samples: 2000
  num_abnormal_samples: 1000
  train_val_split: 0.8
  
  # Image preprocessing
  image_size: 128
  normalize_mean: 0.485
  normalize_std: 0.229
  
  # Data augmentation
  enable_augmentation: true
  rotation_range: 15
  horizontal_flip: true
  brightness_range: [0.8, 1.2]
  
  # Streaming parameters
  streaming_enabled: true
  memory_limit_gb: 4

# Streaming configuration
streaming:
  memory_limit_gb: 4
  batch_size: 32
  num_workers: 0
  prefetch_factor: 2
  cache_size_mb: 512
  
  # Kaggle dataset configuration
  dataset_name: nih-chest-xray-dataset
  dataset_stages: [images_001, images_002, images_003]
  
  # Memory management
  cleanup_frequency: 100
  gc_threshold: 0.8
"""
    
    with open(config_path, 'w') as f:
        f.write(default_config)


def show_device_info(args):
    """Show device and system information."""
    print("🖥️  Device Information")
    print("=" * 25)
    
    config_manager = ConfigManager()
    
    # Device info
    device = config_manager.get_device()
    print(f"Selected device: {device}")
    
    # PyTorch info
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
    
    # MPS info (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Apple Silicon) available: Yes")
    
    # Memory info
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"System RAM: {memory.total // 1024**3} GB")
        print(f"Available RAM: {memory.available // 1024**3} GB")
    except ImportError:
        print("Memory info: psutil not available")
    
    # Environment
    is_colab = config_manager.is_colab_environment()
    print(f"Environment: {'Google Colab' if is_colab else 'Local'}")


if __name__ == "__main__":
    config_command()