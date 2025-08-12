# T4 GPU Training Guide for ImgAE-Dx

**Complete guide for training medical image anomaly detection models on T4 GPU with HuggingFace streaming**

## 🎯 Overview

This guide provides comprehensive instructions for training ImgAE-Dx models on NVIDIA T4 GPU (16GB VRAM) with optimized performance, mixed precision training, and HuggingFace dataset streaming.

## 🚀 Quick Start (Google Colab)

### 1. One-Command Setup

```bash
# Clone and setup ImgAE-Dx for T4 GPU
!git clone https://github.com/luanbhk/imgae-dx.git
%cd ImgAE-Dx
!chmod +x scripts/setup_colab_t4.sh
!./scripts/setup_colab_t4.sh
```

### 2. Quick Training

```bash
# T4-optimized training with default settings
!./scripts/train_colab_t4.sh unet

# Full training with custom parameters
!./scripts/train_colab_t4.sh unet --samples 5000 --epochs 25 --batch-size 48
```

## 🔧 T4 GPU Optimizations

### Mixed Precision Training (AMP)

T4 GPUs excel with mixed precision training, providing:
- **2x memory efficiency**: Train with larger batch sizes
- **1.5-2x speed improvement**: Faster forward/backward passes
- **Maintained accuracy**: Automatic loss scaling prevents underflow

```python
# Automatically enabled in T4 trainer
trainer = Trainer(
    model=model,
    device="cuda",
    use_mixed_precision=True  # Default for T4
)
```

### Memory Optimization

**T4 Memory Configuration:**
- Total: 16GB VRAM
- Reserved: 85% (13.6GB for training)
- System: 15% (2.4GB for CUDA operations)

```yaml
# configs/t4_gpu_config.yaml
performance:
  memory_fraction: 0.85
  empty_cache_frequency: 10
  pin_memory: true
```

### Batch Size Optimization

**Recommended batch sizes for 128x128 images:**

| Precision | Batch Size | Memory Usage | Training Speed |
|-----------|------------|--------------|----------------|
| Mixed (FP16) | 48-64 | ~12GB | Fast ⚡ |
| Full (FP32) | 24-32 | ~14GB | Slower |

```bash
# Optimal T4 settings
./scripts/train_colab_t4.sh unet \
  --batch-size 48 \
  --mixed-precision \
  --t4-optimizations
```

## 📊 HuggingFace Streaming Integration

### Supported Datasets

| Dataset | Size | Speed | Use Case |
|---------|------|-------|----------|
| `keremberke/chest-xray-classification` | ~5GB | Fast | Quick training |
| `alkzar90/NIH-Chest-X-ray-dataset` | ~45GB | Streaming | Full research |
| `Francesco/chest-xray-pneumonia-detection` | ~2GB | Very Fast | Prototyping |

### Streaming Configuration

```bash
# Training with HuggingFace streaming
./scripts/train_colab_t4.sh unet \
  --hf-dataset "keremberke/chest-xray-classification" \
  --samples 3000 \
  --epochs 20
```

### Authentication Setup

```bash
# Set HuggingFace token for private datasets
export HF_TOKEN="hf_your_token_here"

# Or pass directly
./scripts/train_colab_t4.sh unet --hf-token "hf_your_token_here"
```

## 🎮 Training Commands

### Basic Commands

```bash
# Quick test (5 minutes)
./scripts/train_colab_t4.sh unet --samples 500 --epochs 5

# Standard training (30-45 minutes)
./scripts/train_colab_t4.sh unet --samples 3000 --epochs 20

# Full research training (1-2 hours)
./scripts/train_colab_t4.sh unet --samples 5000 --epochs 30
```

### Advanced Commands

```bash
# Both models with custom dataset
./scripts/train_colab_t4.sh both \
  --hf-dataset "alkzar90/NIH-Chest-X-ray-dataset" \
  --samples 4000 \
  --epochs 25 \
  --learning-rate 2e-4 \
  --wandb-project "my-medical-ai"

# Resume training from checkpoint
./scripts/train.sh unet \
  --resume outputs/checkpoints/UNet_epoch_10.pth \
  --epochs 30
```

## 📈 Performance Benchmarks

### T4 GPU Performance (128x128 images)

| Model | Batch Size | Precision | Samples/sec | Memory Usage |
|-------|------------|-----------|-------------|--------------|
| U-Net | 48 | Mixed | ~850 | 12.1GB |
| U-Net | 32 | Full | ~420 | 13.8GB |
| Reversed AE | 64 | Mixed | ~1100 | 11.6GB |
| Reversed AE | 48 | Full | ~520 | 14.2GB |

### Training Time Estimates

| Configuration | Samples | Epochs | Time (T4) |
|---------------|---------|--------|-----------|
| Quick test | 500 | 5 | 3-5 min |
| Development | 2000 | 15 | 15-25 min |
| Research | 5000 | 25 | 45-75 min |
| Full dataset | 10000 | 30 | 2-3 hours |

## 🔍 Monitoring & Debugging

### GPU Memory Monitoring

```python
# In Colab cell
!nvidia-smi -l 1  # Monitor every second

# Or use built-in monitoring
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.1f}GB")
print(f"Cached: {torch.cuda.memory_reserved()/1e9:.1f}GB")
```

### Common Issues & Solutions

#### Out of Memory (OOM)

```bash
# Solution 1: Reduce batch size
./scripts/train_colab_t4.sh unet --batch-size 24

# Solution 2: Use smaller image size  
./scripts/train_colab_t4.sh unet --image-size 96

# Solution 3: Reduce samples
./scripts/train_colab_t4.sh unet --samples 2000
```

#### Slow Training

```bash
# Ensure mixed precision is enabled
./scripts/train_colab_t4.sh unet --mixed-precision

# Use optimal batch size
./scripts/train_colab_t4.sh unet --batch-size 48

# Check T4 optimizations are enabled
./scripts/train_colab_t4.sh unet --t4-optimizations
```

#### HuggingFace Connection Issues

```bash
# Test connection
python -c "from datasets import load_dataset; print('HF connection OK')"

# Use alternative dataset
./scripts/train_colab_t4.sh unet --hf-dataset "keremberke/chest-xray-classification"

# Check token
echo $HF_TOKEN
```

## 📊 Experiment Tracking

### Weights & Biases Integration

```bash
# Login to W&B (one time setup)
!wandb login

# Training with W&B logging
./scripts/train_colab_t4.sh unet \
  --wandb-project "imgae-dx-experiments" \
  --samples 3000 \
  --epochs 20
```

### View Results

- **W&B Dashboard**: https://wandb.ai/your-username/imgae-dx-t4
- **Local logs**: `outputs/logs/`
- **Checkpoints**: `outputs/checkpoints/`
- **Google Drive** (Colab): `/content/drive/MyDrive/ImgAE-Dx/`

## ⚙️ Configuration Files

### T4-Optimized Config

Use the pre-configured T4 settings:

```bash
# Use T4 optimized configuration
./scripts/train.sh unet --config configs/t4_gpu_config.yaml
```

### Custom Configuration

```yaml
# configs/my_t4_config.yaml
training:
  batch_size: 48
  epochs: 25
  mixed_precision: true

streaming:
  memory_limit_gb: 14
  num_workers: 2
  prefetch_factor: 3

performance:
  memory_fraction: 0.85
  cudnn_benchmark: true
```

## 🏆 Best Practices

### 1. Memory Management
- Always use mixed precision on T4
- Monitor GPU memory usage
- Clear cache regularly
- Use appropriate batch sizes

### 2. Training Strategy
- Start with quick tests (500 samples)
- Scale up gradually
- Use early stopping
- Save checkpoints frequently

### 3. Data Handling
- Use HuggingFace streaming for large datasets
- Enable data loading optimizations
- Monitor data pipeline bottlenecks

### 4. Experiment Organization
- Use meaningful W&B project names
- Save results to Google Drive
- Document hyperparameter changes
- Track model performance metrics

## 📚 Advanced Usage

### Custom Model Training

```python
from imgae_dx.training import Trainer
from imgae_dx.models import UNet

# Create T4-optimized trainer
trainer = Trainer(
    model=UNet(input_channels=1, input_size=128),
    device="cuda",
    use_mixed_precision=True,
    wandb_project="my-experiment"
)

# Get optimal batch size for T4
optimal_batch = trainer.get_optimal_batch_size(base_batch_size=32)
print(f"T4 optimal batch size: {optimal_batch}")
```

### Multi-GPU Setup (if available)

```bash
# Enable data parallel training
export CUDA_VISIBLE_DEVICES=0,1
./scripts/train_colab_t4.sh unet --batch-size 96
```

## 🔧 Troubleshooting

### Performance Issues

1. **Check GPU utilization**:
   ```bash
   !nvidia-smi dmon -s u
   ```

2. **Monitor data loading**:
   ```python
   # Add to training script
   import time
   start_time = time.time()
   for batch in train_loader:
       load_time = time.time() - start_time
       print(f"Batch load time: {load_time:.3f}s")
       break
   ```

3. **Profile memory usage**:
   ```python
   torch.cuda.memory._record_memory_history(True)
   # ... training code ...
   torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
   ```

### Error Codes & Solutions

| Error | Solution |
|-------|----------|
| `CUDA out of memory` | Reduce batch_size or image_size |
| `RuntimeError: cuDNN error` | Restart runtime, check CUDA version |
| `ConnectionError: HuggingFace` | Check internet, verify dataset name |
| `W&B login failed` | Run `wandb login` manually |

## 📞 Support

- **GitHub Issues**: [ImgAE-Dx Issues](https://github.com/luanbhk/imgae-dx/issues)
- **Documentation**: [Full Documentation](../README.md)
- **Examples**: [Training Examples](../../examples/)

## 🎯 Summary

**Key Commands for T4 GPU Training:**

```bash
# 1. Setup (one-time)
!./scripts/setup_colab_t4.sh

# 2. Quick training
!./scripts/train_colab_t4.sh unet

# 3. Full training
!./scripts/train_colab_t4.sh unet --samples 5000 --epochs 25

# 4. Both models
!./scripts/train_colab_t4.sh both
```

**T4 GPU provides excellent performance for medical image anomaly detection with the right optimizations. This guide ensures you get maximum performance from your T4 training sessions!** 🚀

---

*Last updated: August 2025 | ImgAE-Dx v0.1.0*