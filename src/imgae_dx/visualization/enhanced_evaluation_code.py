# ========================================
# Enhanced Model Evaluation with Detailed Logging
# ========================================

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
from datetime import datetime
import json

def compute_reconstruction_error(model, images, batch_size=32, model_name="Model"):
    """Compute reconstruction error for anomaly detection with detailed logging"""
    model.eval()
    errors = []
    batch_errors = []
    
    print(f"\n🔄 Computing reconstruction errors for {model_name}...")
    print(f"   Total images: {len(images)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Expected batches: {len(images) // batch_size + (1 if len(images) % batch_size else 0)}")
    
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, len(images), batch_size)):
            batch = images[i:i+batch_size].to(device)
            
            # Forward pass
            if CONFIG.get('mixed_precision', False):
                with autocast():
                    reconstructed = model(batch)
            else:
                reconstructed = model(batch)

            # Compute MSE per image
            mse_per_image = torch.mean((batch - reconstructed)**2, dim=[1,2,3])
            batch_mse = mse_per_image.cpu().numpy()
            
            errors.extend(batch_mse)
            batch_errors.append(np.mean(batch_mse))
            
            # Progress logging
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(range(0, len(images), batch_size)):
                avg_error = np.mean(batch_mse)
                print(f"   Batch {batch_idx + 1:3d}: Avg error = {avg_error:.6f}, "
                      f"Min = {np.min(batch_mse):.6f}, Max = {np.max(batch_mse):.6f}")
    
    errors = np.array(errors)
    processing_time = time.time() - start_time
    
    # Detailed statistics
    print(f"\n📊 {model_name} Reconstruction Error Statistics:")
    print(f"   Processing time: {processing_time:.2f} seconds")
    print(f"   Images per second: {len(images) / processing_time:.1f}")
    print(f"   Total errors computed: {len(errors)}")
    print(f"   Error statistics:")
    print(f"      Mean: {np.mean(errors):.6f}")
    print(f"      Std:  {np.std(errors):.6f}")
    print(f"      Min:  {np.min(errors):.6f}")
    print(f"      Max:  {np.max(errors):.6f}")
    print(f"      Median: {np.median(errors):.6f}")
    print(f"      Q25:  {np.percentile(errors, 25):.6f}")
    print(f"      Q75:  {np.percentile(errors, 75):.6f}")
    print(f"      Q95:  {np.percentile(errors, 95):.6f}")
    print(f"      Q99:  {np.percentile(errors, 99):.6f}")
    
    # Error distribution analysis
    low_errors = np.sum(errors < np.median(errors))
    high_errors = np.sum(errors >= np.median(errors))
    print(f"   Error distribution:")
    print(f"      Below median: {low_errors} ({low_errors/len(errors)*100:.1f}%)")
    print(f"      Above median: {high_errors} ({high_errors/len(errors)*100:.1f}%)")
    
    return errors, {
        'processing_time': processing_time,
        'mean': np.mean(errors),
        'std': np.std(errors),
        'min': np.min(errors),
        'max': np.max(errors),
        'median': np.median(errors),
        'q25': np.percentile(errors, 25),
        'q75': np.percentile(errors, 75),
        'q95': np.percentile(errors, 95),
        'q99': np.percentile(errors, 99),
        'batch_errors': batch_errors
    }

def analyze_test_set(test_images, test_labels):
    """Analyze test set composition"""
    print(f"\n📊 TEST SET ANALYSIS:")
    print(f"   Total test images: {len(test_images)}")
    print(f"   Test labels shape: {test_labels.shape if hasattr(test_labels, 'shape') else 'Not available'}")
    
    if test_labels is not None:
        normal_count = np.sum(test_labels == 0)
        abnormal_count = np.sum(test_labels == 1)
        print(f"   NORMAL samples: {normal_count} ({normal_count/len(test_labels)*100:.1f}%)")
        print(f"   PNEUMONIA samples: {abnormal_count} ({abnormal_count/len(test_labels)*100:.1f}%)")
        print(f"   Class balance ratio: 1:{abnormal_count/normal_count:.2f}")
    else:
        print(f"   Labels not available for analysis")

# Enhanced model loading and evaluation
print("="*80)
print("🎯 ENHANCED MODEL EVALUATION FOR ANOMALY DETECTION")
print("="*80)

# Analyze test set first
print(f"\n📋 EVALUATION SETUP:")
print(f"   Device: {device}")
print(f"   Mixed precision: {CONFIG.get('mixed_precision', False)}")
print(f"   Checkpoint directory: {CONFIG['checkpoint_dir']}")

# Check if test_images and test_labels exist
try:
    test_set_size = len(test_images)
    print(f"   Test set ready: {test_set_size} images")
    analyze_test_set(test_images, test_labels if 'test_labels' in globals() else None)
except:
    print("   ⚠️  Test set not available - need to create test set first")
    print("   Please run test set creation code before evaluation")

# Load and evaluate both models
eval_models = {}
model_errors = {}
model_stats = {}

print(f"\n🤖 MODEL LOADING AND EVALUATION:")

# Load U-Net
if 'unet' in [name for name, _ in models_to_train] or 'unet' in trained_models:
    print(f"\n📂 Loading U-Net model...")
    unet_eval = UNet(
        in_channels=1,
        out_channels=1,
        features=[64, 128, 256, 512]
    ).to(device)

    # Try to load from checkpoint first
    unet_path = f"{CONFIG['checkpoint_dir']}/unet_best.pth"
    loaded_from_checkpoint = False
    
    try:
        if os.path.exists(unet_path):
            unet_eval.load_state_dict(torch.load(unet_path, map_location=device))
            print(f"   ✅ U-Net loaded from checkpoint: {unet_path}")
            
            # Get file info
            file_size = os.path.getsize(unet_path) / (1024*1024)  # MB
            file_time = datetime.fromtimestamp(os.path.getmtime(unet_path))
            print(f"   📁 Checkpoint size: {file_size:.1f} MB")
            print(f"   🕒 Checkpoint date: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
            loaded_from_checkpoint = True
        else:
            print(f"   ⚠️  Checkpoint not found: {unet_path}")
            raise FileNotFoundError("Checkpoint not available")
            
    except Exception as e:
        print(f"   ⚠️  Could not load from checkpoint: {e}")
        if 'unet' in trained_models:
            unet_eval = trained_models['unet']
            print(f"   ✅ Using U-Net from training memory")
        else:
            print(f"   ❌ U-Net not available for evaluation")
            unet_eval = None

    if unet_eval is not None:
        # Model info
        total_params = sum(p.numel() for p in unet_eval.parameters())
        trainable_params = sum(p.numel() for p in unet_eval.parameters() if p.requires_grad)
        print(f"   📊 U-Net parameters: {trainable_params:,} trainable, {total_params:,} total")
        
        eval_models['unet'] = unet_eval
        
        # Compute errors with detailed logging
        try:
            unet_errors, unet_stats = compute_reconstruction_error(
                unet_eval, test_images, batch_size=CONFIG.get('batch_size', 32), model_name="U-Net"
            )
            model_errors['unet'] = unet_errors
            model_stats['unet'] = unet_stats
            model_stats['unet']['loaded_from_checkpoint'] = loaded_from_checkpoint
            print(f"   ✅ U-Net evaluation completed successfully")
        except Exception as e:
            print(f"   ❌ U-Net evaluation failed: {e}")

# Load Reversed AE
if 'reversed_ae' in [name for name, _ in models_to_train] or 'reversed_ae' in trained_models:
    print(f"\n📂 Loading Reversed Autoencoder model...")
    ra_eval = ReversedAutoencoder(
        in_channels=1,
        latent_dim=128,
        image_size=CONFIG['image_size']
    ).to(device)

    # Try to load from checkpoint first
    ra_path = f"{CONFIG['checkpoint_dir']}/reversed_ae_best.pth"
    loaded_from_checkpoint = False
    
    try:
        if os.path.exists(ra_path):
            ra_eval.load_state_dict(torch.load(ra_path, map_location=device))
            print(f"   ✅ RA loaded from checkpoint: {ra_path}")
            
            # Get file info
            file_size = os.path.getsize(ra_path) / (1024*1024)  # MB
            file_time = datetime.fromtimestamp(os.path.getmtime(ra_path))
            print(f"   📁 Checkpoint size: {file_size:.1f} MB")
            print(f"   🕒 Checkpoint date: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
            loaded_from_checkpoint = True
        else:
            print(f"   ⚠️  Checkpoint not found: {ra_path}")
            raise FileNotFoundError("Checkpoint not available")
            
    except Exception as e:
        print(f"   ⚠️  Could not load from checkpoint: {e}")
        if 'reversed_ae' in trained_models:
            ra_eval = trained_models['reversed_ae']
            print(f"   ✅ Using RA from training memory")
        else:
            print(f"   ❌ RA not available for evaluation")
            ra_eval = None

    if ra_eval is not None:
        # Model info
        total_params = sum(p.numel() for p in ra_eval.parameters())
        trainable_params = sum(p.numel() for p in ra_eval.parameters() if p.requires_grad)
        print(f"   📊 RA parameters: {trainable_params:,} trainable, {total_params:,} total")
        
        eval_models['reversed_ae'] = ra_eval
        
        # Compute errors with detailed logging
        try:
            ra_errors, ra_stats = compute_reconstruction_error(
                ra_eval, test_images, batch_size=CONFIG.get('batch_size', 32), model_name="Reversed AE"
            )
            model_errors['reversed_ae'] = ra_errors
            model_stats['reversed_ae'] = ra_stats
            model_stats['reversed_ae']['loaded_from_checkpoint'] = loaded_from_checkpoint
            print(f"   ✅ RA evaluation completed successfully")
        except Exception as e:
            print(f"   ❌ RA evaluation failed: {e}")

# Final summary
print(f"\n" + "="*80)
print(f"📋 EVALUATION SUMMARY")
print(f"="*80)

if len(eval_models) > 0:
    print(f"✅ Models successfully evaluated: {list(eval_models.keys())}")
    
    for model_name in eval_models.keys():
        if model_name in model_errors:
            errors = model_errors[model_name]
            stats = model_stats[model_name]
            
            print(f"\n🤖 {model_name.upper()}:")
            print(f"   Samples processed: {len(errors)}")
            print(f"   Processing time: {stats['processing_time']:.2f}s")
            print(f"   Throughput: {len(errors)/stats['processing_time']:.1f} images/sec")
            print(f"   Error range: [{stats['min']:.6f} - {stats['max']:.6f}]")
            print(f"   Mean ± Std: {stats['mean']:.6f} ± {stats['std']:.6f}")
            print(f"   Median (Q50): {stats['median']:.6f}")
            print(f"   IQR: [{stats['q25']:.6f} - {stats['q75']:.6f}]")
            print(f"   Extreme values: Q95={stats['q95']:.6f}, Q99={stats['q99']:.6f}")
            print(f"   Source: {'Checkpoint' if stats['loaded_from_checkpoint'] else 'Memory'}")
    
    print(f"\n🎯 READY FOR ANOMALY DETECTION ANALYSIS")
    print(f"   Next steps: Compute AUC-ROC, precision-recall curves")
    print(f"   Research questions validation can proceed")
    
else:
    print(f"❌ No models available for evaluation")
    print(f"   Please check training completion and checkpoint availability")

print(f"\n💾 All evaluation data stored in:")
print(f"   model_errors: {list(model_errors.keys())}")
print(f"   model_stats: {list(model_stats.keys())}")
print(f"   eval_models: {list(eval_models.keys())}")

# Save evaluation results
if len(model_stats) > 0:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_results = {
        'timestamp': timestamp,
        'config': CONFIG,
        'model_stats': model_stats,
        'evaluation_summary': {
            'models_evaluated': list(eval_models.keys()),
            'test_set_size': len(test_images) if 'test_images' in globals() else 0,
            'evaluation_successful': len(model_errors) > 0
        }
    }
    
    results_file = f"{CONFIG['checkpoint_dir']}/evaluation_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        # Convert numpy types to regular Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        json.dump(eval_results, f, indent=2, default=convert_numpy)
    
    print(f"\n💾 Evaluation results saved: {results_file}")