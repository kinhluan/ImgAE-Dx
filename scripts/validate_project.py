#!/usr/bin/env python3
"""
Project validation script for ImgAE-Dx.

This script validates that all core components are working correctly.
"""

import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def validate_imports():
    """Validate that all modules can be imported."""
    print("🔍 Validating imports...")
    
    try:
        from imgae_dx import create_model, create_trainer, load_config
        from imgae_dx.models import UNet, ReversedAutoencoder
        from imgae_dx.utils import ConfigManager
        from imgae_dx.data import StreamingNIHDataset
        from imgae_dx.training import Trainer, Evaluator
        from imgae_dx.streaming import KaggleStreamClient, StreamingMemoryManager
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def validate_models():
    """Validate that models work correctly."""
    print("🧠 Validating models...")
    
    try:
        from imgae_dx import create_model
        
        # Create models
        unet = create_model('unet', {
            'input_channels': 1, 
            'input_size': 128, 
            'latent_dim': 512
        })
        
        reversed_ae = create_model('reversed_ae', {
            'input_channels': 1, 
            'input_size': 128, 
            'latent_dim': 512
        })
        
        # Test forward passes
        x = torch.randn(1, 1, 128, 128)
        
        with torch.no_grad():
            unet_out = unet(x)
            rae_out = reversed_ae(x)
            
            # Test encode/decode
            unet_encoded = unet.encode(x)
            unet_decoded = unet.decode(unet_encoded)
            
            rae_encoded = reversed_ae.encode(x)
            rae_decoded = reversed_ae.decode(rae_encoded)
        
        print(f"✅ U-Net: {unet.count_parameters():,} parameters")
        print(f"✅ Reversed AE: {reversed_ae.count_parameters():,} parameters")
        print(f"✅ Forward passes: {x.shape} -> {unet_out.shape}")
        print(f"✅ Encode/decode working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

def validate_configuration():
    """Validate configuration system."""
    print("⚙️  Validating configuration...")
    
    try:
        from imgae_dx.utils import ConfigManager
        
        config_manager = ConfigManager()
        device = config_manager.get_device()
        
        print(f"✅ Device detection: {device}")
        print(f"✅ Configuration system working")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def validate_streaming():
    """Validate streaming components (without requiring Kaggle)."""
    print("🌊 Validating streaming components...")
    
    try:
        from imgae_dx.streaming import StreamingMemoryManager
        
        # Test memory manager
        memory_manager = StreamingMemoryManager(memory_limit_gb=1.0)
        memory_info = memory_manager.check_memory_usage()
        
        print(f"✅ Memory manager: {memory_info['system_used_percent']:.1f}% system memory used")
        print(f"✅ Streaming components working")
        
        return True
        
    except Exception as e:
        print(f"❌ Streaming validation failed: {e}")
        return False

def validate_training():
    """Validate training components."""
    print("🏋️  Validating training components...")
    
    try:
        from imgae_dx import create_model
        from imgae_dx.training import Trainer, Evaluator
        from imgae_dx.training.metrics import AnomalyMetrics
        
        # Create dummy model
        unet = create_model('unet', {
            'input_channels': 1, 
            'input_size': 64, 
            'latent_dim': 128
        })
        
        # Create trainer
        trainer = Trainer(unet)
        
        # Test metrics
        metrics = AnomalyMetrics()
        
        # Dummy data for testing
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        
        eval_results = metrics.evaluate_anomaly_detection(y_true, y_scores)
        
        print(f"✅ Trainer created successfully")
        print(f"✅ Metrics working: AUC = {eval_results['auc_roc']:.3f}")
        print(f"✅ Training components working")
        
        return True
        
    except Exception as e:
        print(f"❌ Training validation failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 ImgAE-Dx Project Validation")
    print("=" * 40)
    
    tests = [
        validate_imports,
        validate_models, 
        validate_configuration,
        validate_streaming,
        validate_training
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("📊 Validation Summary")
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All validation tests passed!")
        print("✅ ImgAE-Dx project is ready for use!")
        return 0
    else:
        print("❌ Some validation tests failed!")
        print("🔧 Please check the errors above and fix issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())