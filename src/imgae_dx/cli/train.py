"""
Training command-line interface.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

from ..utils import ConfigManager
from ..models import UNet, ReversedAutoencoder
from ..training import Trainer
from ..streaming import KaggleStreamClient, StreamingMemoryManager, HuggingFaceStreamClient
from ..data import create_streaming_dataloaders, create_hf_streaming_dataloaders


def train_command():
    """Main training command entry point."""
    parser = argparse.ArgumentParser(
        description="Train ImgAE-Dx models for medical image anomaly detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model", 
        type=str, 
        choices=["unet", "reversed_ae"], 
        required=True,
        help="Model architecture to train"
    )
    
    # Data arguments
    parser.add_argument(
        "--samples", 
        type=int, 
        default=2000,
        help="Number of samples to use for training"
    )
    
    parser.add_argument(
        "--stage",
        type=str,
        default="images",
        help="Dataset stage to use (for Kaggle)"
    )
    
    # Data source arguments
    parser.add_argument(
        "--data-source",
        type=str,
        choices=["kaggle", "huggingface", "hf"],
        default="kaggle",
        help="Data source: kaggle or huggingface"
    )
    
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="alkzar90/NIH-Chest-X-ray-dataset",
        help="HuggingFace dataset name"
    )
    
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="HuggingFace dataset split"
    )
    
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HuggingFace authentication token"
    )
    
    parser.add_argument(
        "--hf-streaming",
        action="store_true",
        default=True,
        help="Enable HuggingFace streaming mode"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=1e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Input image size"
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs/checkpoints",
        help="Output directory for checkpoints"
    )
    
    # Resume training
    parser.add_argument(
        "--resume", 
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    # Device and T4 GPU optimizations
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=True,
        help="Enable mixed precision training (T4 GPU optimization)"
    )
    
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training"
    )
    
    parser.add_argument(
        "--optimize-for-t4",
        action="store_true",
        default=True,
        help="Enable T4 GPU specific optimizations"
    )
    
    # Memory management
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=4.0,
        help="Memory limit in GB"
    )
    
    # Logging
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="imgae-dx",
        help="W&B project name"
    )
    
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging"
    )
    
    parser.add_argument(
        "--no-wandb-artifacts",
        action="store_true",
        help="Disable W&B artifact saving"
    )
    
    # Validation
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    
    # Early stopping
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum change for early stopping"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute training
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)


def main(args):
    """Main training logic."""
    print("🧠 ImgAE-Dx Model Training")
    print("=" * 40)
    print(f"Model: {args.model}")
    print(f"Samples: {args.samples}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Setup configuration
    config_manager = ConfigManager()
    
    if args.config:
        config = config_manager.load_config(args.config)
        print(f"✅ Loaded config from: {args.config}")
    else:
        # Use default configuration with command line overrides
        from ..utils.config_manager import ProjectConfig
        config = ProjectConfig()
        print("⚠️  Using default configuration")
    
    # Override config with command line arguments
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.data.image_size = args.image_size
    config.data.num_normal_samples = args.samples
    config.streaming.memory_limit_gb = args.memory_limit
    
    # Setup device
    if args.device == "auto":
        device = config_manager.get_device()
    else:
        device = args.device
    
    print(f"🖥️  Using device: {device}")
    
    # Setup memory management
    memory_manager = StreamingMemoryManager(
        memory_limit_gb=args.memory_limit,
        enable_monitoring=True
    )
    
    # Setup data client based on source
    data_source = args.data_source.lower()
    if data_source in ['huggingface', 'hf']:
        # Setup HuggingFace client
        try:
            hf_client = HuggingFaceStreamClient(
                dataset_name=args.hf_dataset,
                streaming=args.hf_streaming,
                token=args.hf_token
            )
            print("✅ HuggingFace client initialized")
            print(f"Dataset: {args.hf_dataset}")
            kaggle_client = None
        except Exception as e:
            print(f"⚠️  HuggingFace client error: {e}")
            print("Falling back to dummy data...")
            hf_client = None
            kaggle_client = None
    else:
        # Setup Kaggle client (default)
        try:
            dataset_name = getattr(config.streaming, 'dataset_name', 'nih-chest-xray/data')
            kaggle_client = KaggleStreamClient(dataset_name=dataset_name)
            print("✅ Kaggle client initialized")
            print(f"Dataset: {dataset_name}")
            hf_client = None
        except Exception as e:
            print(f"⚠️  Kaggle client error: {e}")
            print("Using dummy data for testing...")
            kaggle_client = None
            hf_client = None
    
    # Create model
    model = create_model(args.model, config.model, device)
    print(f"✅ Model created: {model.count_parameters():,} parameters")
    
    # Create data loaders based on data source
    if hf_client:
        # HuggingFace data loaders
        train_loader, val_loader, dataset_info = create_hf_streaming_dataloaders(
            dataset_name=args.hf_dataset,
            split=args.hf_split,
            batch_size=args.batch_size,
            train_ratio=1.0 - args.val_split,
            max_samples=args.samples,
            memory_manager=memory_manager,
            image_size=args.image_size,
            streaming=args.hf_streaming,
            hf_token=args.hf_token
        )
    elif kaggle_client:
        # Kaggle data loaders
        train_loader, val_loader, dataset_info = create_streaming_dataloaders(
            kaggle_client=kaggle_client,
            stage=args.stage,
            batch_size=args.batch_size,
            train_ratio=1.0 - args.val_split,
            max_samples=args.samples,
            memory_manager=memory_manager,
            image_size=args.image_size
        )
    else:
        # Create dummy data loaders for testing
        train_loader, val_loader = create_dummy_dataloaders(
            batch_size=args.batch_size,
            image_size=args.image_size,
            num_samples=args.samples
        )
        dataset_info = {
            'total_samples': args.samples,
            'train_samples': int(args.samples * (1.0 - args.val_split)),
            'val_samples': int(args.samples * args.val_split)
        }
    
    print(f"✅ Data loaders created:")
    print(f"   Train: {dataset_info['train_samples']} samples")
    print(f"   Val:   {dataset_info['val_samples']} samples")
    
    # Auto-adjust batch size if too large
    train_samples = dataset_info['train_samples']
    if train_samples < args.batch_size:
        old_batch_size = args.batch_size
        args.batch_size = max(1, train_samples // 2)  # Use half of train samples as batch size
        print(f"⚠️  Batch size adjusted from {old_batch_size} to {args.batch_size} (train samples: {train_samples})")
        
        # Recreate data loaders with adjusted batch size
        if hf_client:
            train_loader, val_loader, dataset_info = create_hf_streaming_dataloaders(
                dataset_name=args.hf_dataset,
                split=args.hf_split,
                batch_size=args.batch_size,
                train_ratio=1.0 - args.val_split,
                max_samples=args.samples,
                memory_manager=memory_manager,
                image_size=args.image_size,
                streaming=args.hf_streaming,
                hf_token=args.hf_token
            )
        elif kaggle_client:
            train_loader, val_loader, dataset_info = create_streaming_dataloaders(
                kaggle_client=kaggle_client,
                stage=args.stage,
                batch_size=args.batch_size,
                train_ratio=1.0 - args.val_split,
                max_samples=args.samples,
                memory_manager=memory_manager,
                image_size=args.image_size
            )
        else:
            train_loader, val_loader = create_dummy_dataloaders(
                batch_size=args.batch_size,
                image_size=args.image_size,
                num_samples=args.samples
            )
    
    # Setup trainer with T4 optimizations
    wandb_project = None if args.no_wandb else args.wandb_project
    save_artifacts = not args.no_wandb_artifacts
    
    # Handle mixed precision settings
    use_mixed_precision = args.mixed_precision and not args.no_mixed_precision
    
    # T4 GPU batch size optimization
    optimized_batch_size = args.batch_size
    if args.optimize_for_t4 and device == "cuda":
        # Create temporary trainer to get optimal batch size
        temp_trainer = Trainer(
            model=model,
            config={},
            device=device,
            use_mixed_precision=use_mixed_precision
        )
        optimized_batch_size = temp_trainer.get_optimal_batch_size(args.batch_size)
        
        if optimized_batch_size != args.batch_size:
            print(f"🎯 T4 Optimization: Batch size adjusted to {optimized_batch_size}")
            args.batch_size = optimized_batch_size
    
    # Clean config dict by removing device field that shouldn't be passed to Trainer
    clean_config = config.__dict__ if hasattr(config, '__dict__') else config
    if isinstance(clean_config, dict) and 'device' in clean_config:
        clean_config = {k: v for k, v in clean_config.items() if k != 'device'}
    
    trainer = Trainer(
        model=model,
        config=clean_config,
        device=device,
        wandb_project=wandb_project,
        save_artifacts=save_artifacts,
        use_mixed_precision=use_mixed_precision
    )
    
    # Setup training components
    trainer.setup_training(
        learning_rate=args.learning_rate,
        optimizer_name="adam",
        scheduler_name="cosine"
    )
    
    print("✅ Trainer initialized")
    
    # Resume from checkpoint if specified
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            trainer.load_checkpoint(resume_path)
            print(f"✅ Resumed from: {resume_path}")
        else:
            print(f"⚠️  Checkpoint not found: {resume_path}")
    
    # Start training
    print("\n🚀 Starting training...")
    
    try:
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            checkpoint_dir=args.output_dir,
            early_stopping_patience=args.patience,
            min_delta=args.min_delta
        )
        
        print(f"\n🎉 Training completed!")
        print(f"Final train loss: {history['train_losses'][-1]:.4f}")
        if history['val_losses']:
            print(f"Final val loss: {history['val_losses'][-1]:.4f}")
        print(f"Best loss: {history['best_loss']:.4f}")
        print(f"Training time: {history['total_time']:.2f} seconds")
        
        # Plot learning curves
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            fig = trainer.plot_learning_curves(
                save_path=f"{args.output_dir}/{args.model}_learning_curves.png"
            )
            print(f"📊 Learning curves saved to: {args.output_dir}/{args.model}_learning_curves.png")
        except Exception as e:
            print(f"⚠️  Could not save learning curves: {e}")
        
    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        raise
    
    finally:
        # Cleanup
        memory_manager.stop_monitoring()
        if kaggle_client:
            kaggle_client.cleanup_cache()
        if hf_client:
            hf_client.cleanup_cache()


def create_model(model_type: str, model_config, device: str):
    """Create model based on type and configuration."""
    
    # Default model configuration
    default_config = {
        'input_channels': 1,
        'input_size': 128,
        'latent_dim': 512
    }
    
    # Update with provided config
    if hasattr(model_config, '__dict__'):
        config_dict = model_config.__dict__
    else:
        config_dict = model_config if model_config else {}
    
    for key, value in config_dict.items():
        if key in default_config:
            default_config[key] = value
    
    if model_type == "unet":
        model = UNet(**default_config)
    elif model_type == "reversed_ae":
        model = ReversedAutoencoder(**default_config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def create_dummy_dataloaders(batch_size: int, image_size: int, num_samples: int):
    """Create dummy data loaders for testing."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dummy data
    train_size = int(num_samples * 0.8)
    val_size = num_samples - train_size
    
    # Generate random data
    train_images = torch.randn(train_size, 1, image_size, image_size)
    train_labels = torch.zeros(train_size, dtype=torch.long)
    
    val_images = torch.randn(val_size, 1, image_size, image_size)
    val_labels = torch.zeros(val_size, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print("⚠️  Using dummy data for testing")
    
    return train_loader, val_loader


if __name__ == "__main__":
    train_command()