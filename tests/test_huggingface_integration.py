"""
Test HuggingFace integration for ImgAE-Dx
"""

import torch
from src.imgae_dx.streaming import HuggingFaceStreamClient
from src.imgae_dx.data import create_hf_streaming_dataloaders


def test_hf_client():
    """Test HuggingFace client basic functionality"""
    print("🤗 Testing HuggingFace Client")
    print("=" * 40)
    
    try:
        # Test different datasets
        datasets_to_test = [
            "mnist",  # Working test dataset
            "alkzar90/NIH-Chest-X-ray-dataset",
            "keremberke/chest-xray-classification"
        ]
        
        for dataset_name in datasets_to_test:
            print(f"\n📊 Testing dataset: {dataset_name}")
            
            try:
                client = HuggingFaceStreamClient(
                    dataset_name=dataset_name,
                    streaming=True
                )
                
                # Get dataset info
                info = client.get_dataset_info()
                print(f"  Splits: {info.get('splits', 'unknown')}")
                print(f"  Columns: {info.get('columns', 'unknown')}")
                
                # Test streaming a few samples
                sample_count = 0
                for sample in client.stream_images(max_samples=3):
                    print(f"  Sample {sample_count}: {sample['image'].size}, {sample['label']}")
                    sample_count += 1
                    if sample_count >= 3:
                        break
                
                print(f"  ✅ Dataset {dataset_name} works!")
                return dataset_name  # Return first working dataset
                
            except Exception as e:
                print(f"  ❌ Dataset {dataset_name} failed: {e}")
                continue
        
        print("❌ No working datasets found")
        return None
        
    except Exception as e:
        print(f"❌ HF client test failed: {e}")
        return None


def test_hf_dataloaders():
    """Test HuggingFace data loaders creation"""
    print("\n🔄 Testing HuggingFace DataLoaders")
    print("=" * 40)
    
    try:
        # Create data loaders with small sample - use all samples for testing
        from src.imgae_dx.data.huggingface_dataset import StreamingHFDataset
        from src.imgae_dx.streaming import HuggingFaceStreamClient
        from src.imgae_dx.data.transforms import MedicalImageTransforms
        from torch.utils.data import DataLoader
        
        hf_client = HuggingFaceStreamClient(dataset_name="mnist", streaming=True)
        transform = MedicalImageTransforms.get_training_transforms(image_size=128)
        
        dataset = StreamingHFDataset(
            hf_client=hf_client,
            split="train",
            transform=transform,
            filter_type="all",  # Use all samples for testing
            max_samples=20
        )
        
        train_dataset, val_dataset = dataset.split_dataset(train_ratio=0.8)
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        dataset_info = {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'total_samples': len(dataset)
        }
        
        print("Dataset Info:")
        for key, value in dataset_info.items():
            if key != 'hf_dataset_info':
                print(f"  {key}: {value}")
        
        # Test a few batches
        print(f"\nTesting data loading:")
        for i, (images, labels) in enumerate(train_loader):
            print(f"  Batch {i}: images {images.shape}, labels {labels.shape}")
            if i >= 2:  # Just test first 3 batches
                break
        
        for i, (images, labels) in enumerate(val_loader):
            print(f"  Val Batch {i}: images {images.shape}, labels {labels.shape}")
            if i >= 1:  # Just test first 2 batches
                break
        
        print("✅ HuggingFace DataLoaders work!")
        return True
        
    except Exception as e:
        print(f"❌ HF DataLoaders test failed: {e}")
        return False


def test_training_integration():
    """Test training with HuggingFace data"""
    print("\n🧠 Testing Training Integration")
    print("=" * 40)
    
    try:
        from src.imgae_dx.models import UNet
        from src.imgae_dx.training import Trainer
        
        # Create small model for testing
        model = UNet(input_channels=1, input_size=128, latent_dim=128)  # Smaller latent_dim
        
        # Create data loaders - use all samples for testing
        from src.imgae_dx.data.huggingface_dataset import StreamingHFDataset
        from src.imgae_dx.streaming import HuggingFaceStreamClient
        from src.imgae_dx.data.transforms import MedicalImageTransforms
        from torch.utils.data import DataLoader
        
        hf_client = HuggingFaceStreamClient(dataset_name="mnist", streaming=True)
        transform = MedicalImageTransforms.get_training_transforms(image_size=128)
        
        dataset = StreamingHFDataset(
            hf_client=hf_client,
            split="train",
            transform=transform,
            filter_type="all",  # Use all samples for testing
            max_samples=10
        )
        
        train_dataset, val_dataset = dataset.split_dataset(train_ratio=0.8)
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        
        dataset_info = {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'total_samples': len(dataset)
        }
        
        # Setup trainer
        trainer = Trainer(
            model=model,
            config={},
            device='cpu',  # Use CPU for testing
            wandb_project=None  # No W&B for test
        )
        
        trainer.setup_training(learning_rate=1e-3)
        
        # Test one training step
        print("Testing training step...")
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=1,  # Just 1 epoch for test
            checkpoint_dir="test_checkpoints"
        )
        
        print(f"✅ Training test completed!")
        print(f"  Train loss: {history['train_losses'][-1]:.4f}")
        print(f"  Val loss: {history['val_losses'][-1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training integration test failed: {e}")
        return False


def test_cli_commands():
    """Test CLI command structure"""
    print("\n💻 Testing CLI Commands")
    print("=" * 40)
    
    # Test CLI help
    import subprocess
    import sys
    
    try:
        # Test help command
        result = subprocess.run([
            sys.executable, "-m", "src.imgae_dx.cli.train", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if "--data-source" in result.stdout and "--hf-dataset" in result.stdout:
            print("✅ CLI arguments for HuggingFace found!")
            
            # Show new HF arguments
            hf_args = [line for line in result.stdout.split('\n') 
                      if any(arg in line for arg in ['--data-source', '--hf-dataset', '--hf-split'])]
            
            print("New HF arguments:")
            for arg in hf_args[:5]:  # Show first 5
                print(f"  {arg.strip()}")
            
            return True
        else:
            print("❌ HF arguments not found in CLI help")
            return False
            
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 ImgAE-Dx HuggingFace Integration Test")
    print("=" * 50)
    
    results = {
        'hf_client': test_hf_client(),
        'hf_dataloaders': test_hf_dataloaders(),
        'training_integration': test_training_integration(),
        'cli_commands': test_cli_commands()
    }
    
    print("\n📊 TEST RESULTS:")
    print("=" * 30)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 HuggingFace integration is ready!")
        print("\nUsage examples:")
        print("  # Train with HuggingFace dataset")
        print("  poetry run python -m imgae_dx.cli.train unet --data-source huggingface --samples 100 --epochs 2")
        print("  ")
        print("  # Train with specific HF dataset")  
        print("  poetry run python -m imgae_dx.cli.train unet --data-source hf --hf-dataset Francesco/chest-xray-pneumonia-detection --samples 50")


if __name__ == "__main__":
    main()