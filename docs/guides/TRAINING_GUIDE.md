# ImgAE-Dx Training Guide

Hướng dẫn chi tiết cách train các model U-Net và Reversed Autoencoder cho medical image anomaly detection.

## Tổng quan

ImgAE-Dx hỗ trợ 2 loại autoencoder architecture:
- **U-Net**: Baseline model với skip connections cho detailed reconstruction
- **Reversed Autoencoder (RA)**: Experimental asymmetric architecture không có skip connections

## Cài đặt và chuẩn bị

### 1. Cài đặt dependencies

```bash
# Cài đặt poetry dependencies
poetry install

# Hoặc với pip
pip install -r requirements.txt
```

### 2. Chuẩn bị data sources

#### Kaggle (mặc định)
```bash
# Cài đặt Kaggle API credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### HuggingFace
```bash
# Set HuggingFace token (optional, cho private datasets)
export HF_TOKEN=your_huggingface_token
```

### 3. Chuẩn bị W&B (optional)

```bash
# Login W&B
wandb login

# Hoặc set API key
export WANDB_API_KEY=your_wandb_key
```

## Cách sử dụng Train Script

### Syntax cơ bản

```bash
./scripts/train.sh MODEL_TYPE [OPTIONS]
```

**MODEL_TYPE:**
- `unet` - Train U-Net autoencoder (baseline)
- `reversed_ae` - Train Reversed Autoencoder
- `both` - Train cả hai models tuần tự

## Training U-Net

### 1. Train cơ bản với Kaggle dataset

```bash
# Quick training cho test
./scripts/train.sh unet --samples 500 --epochs 5

# Full training
./scripts/train.sh unet --samples 2000 --epochs 20 --gpu
```

### 2. Train với HuggingFace dataset

```bash
# Basic HuggingFace training
./scripts/train.sh unet --data-source huggingface --samples 1000 --epochs 15

# Với custom dataset
./scripts/train.sh unet \
  --data-source hf \
  --hf-dataset alkzar90/NIH-Chest-X-ray-dataset \
  --hf-split train \
  --samples 2000 \
  --epochs 20
```

### 3. Train với advanced options

```bash
# Full production training
./scripts/train.sh unet \
  --data-source huggingface \
  --samples 5000 \
  --epochs 30 \
  --batch-size 32 \
  --gpu \
  --memory-limit 8 \
  --wandb-project medical-anomaly-detection \
  --output-dir models/unet
```

## Training Reversed Autoencoder

### 1. Train cơ bản

```bash
# Quick test
./scripts/train.sh reversed_ae --samples 500 --epochs 5

# Standard training
./scripts/train.sh reversed_ae --samples 2000 --epochs 20 --gpu
```

### 2. Train với HuggingFace

```bash
# Basic RA training với HF
./scripts/train.sh reversed_ae \
  --data-source huggingface \
  --samples 1500 \
  --epochs 25 \
  --batch-size 16

# Production RA training
./scripts/train.sh reversed_ae \
  --data-source hf \
  --hf-dataset alkzar90/NIH-Chest-X-ray-dataset \
  --samples 3000 \
  --epochs 35 \
  --gpu \
  --memory-limit 6
```

### 3. Train với custom config

```bash
# Với custom configuration file
./scripts/train.sh reversed_ae \
  --config configs/reversed_ae_config.yaml \
  --samples 2000 \
  --epochs 30 \
  --gpu
```

## Training cả hai models

### 1. Sequential training

```bash
# Train cả U-Net và RA tuần tự
./scripts/train.sh both --samples 2000 --epochs 20 --gpu

# Với HuggingFace dataset
./scripts/train.sh both \
  --data-source huggingface \
  --samples 1500 \
  --epochs 25 \
  --batch-size 32 \
  --gpu
```

## Resume Training

### 1. Resume từ checkpoint

```bash
# Resume U-Net training
./scripts/train.sh unet --resume checkpoints/unet_epoch_15.pth

# Resume RA training
./scripts/train.sh reversed_ae --resume checkpoints/reversed_ae_best.pth
```

## Advanced Training Options

### 1. Memory và Performance

```bash
# Training với memory limit
./scripts/train.sh unet --memory-limit 4 --batch-size 16

# Force CPU training
./scripts/train.sh reversed_ae --cpu --batch-size 8

# Force GPU training
./scripts/train.sh unet --gpu --batch-size 64
```

### 2. W&B Configuration

```bash
# Custom W&B project
./scripts/train.sh unet --wandb-project my-medical-ai

# Disable W&B logging
./scripts/train.sh reversed_ae --no-wandb

# Disable W&B artifacts (chỉ log metrics)
./scripts/train.sh unet --no-wandb-artifacts
```

### 3. HuggingFace Advanced

```bash
# Với authentication token
./scripts/train.sh reversed_ae \
  --data-source hf \
  --hf-token your_token_here \
  --hf-dataset private/dataset

# Disable streaming mode
./scripts/train.sh unet \
  --data-source huggingface \
  --no-hf-streaming

# Custom split
./scripts/train.sh reversed_ae \
  --data-source hf \
  --hf-split validation
```

## Training với CLI trực tiếp

Nếu không muốn dùng script, có thể gọi CLI trực tiếp:

### U-Net

```bash
poetry run python -m imgae_dx.cli.train \
  --model unet \
  --data-source huggingface \
  --samples 2000 \
  --epochs 20 \
  --batch-size 32 \
  --gpu
```

### Reversed Autoencoder

```bash
poetry run python -m imgae_dx.cli.train \
  --model reversed_ae \
  --data-source kaggle \
  --samples 1500 \
  --epochs 25 \
  --batch-size 16 \
  --memory-limit 6
```

## Output và Results

### 1. Checkpoints

Models được save vào `outputs/checkpoints/` directory:
```
outputs/
├── checkpoints/
│   ├── unet_epoch_10.pth
│   ├── unet_best.pth
│   ├── reversed_ae_epoch_15.pth
│   └── reversed_ae_best.pth
├── logs/
│   ├── unet_20241212_143022.log
│   └── reversed-ae_20241212_150333.log
├── cache/
├── artifacts/
└── results/
```

### 3. W&B Artifacts

Nếu enable W&B, models sẽ được upload như artifacts:
- `unet-checkpoint`: Regular checkpoints
- `unet-best-model`: Best model
- `reversed-ae-checkpoint`: RA checkpoints  
- `reversed-ae-best-model`: RA best model

### 4. Learning Curves

Training curves được save như PNG files:
```
outputs/checkpoints/
├── unet_learning_curves.png
└── reversed_ae_learning_curves.png
```

## Monitoring Training

### 1. Xem logs real-time

```bash
# Tail log file
tail -f outputs/logs/unet_20241212_143022.log

# Với grep
tail -f outputs/logs/unet_20241212_143022.log | grep "Epoch"
```

### 2. W&B Dashboard

Nếu sử dụng W&B, có thể monitor qua web interface:
- Training/validation loss curves
- Model metrics
- System metrics (GPU, memory)
- Model artifacts

## Troubleshooting

### 1. Memory Issues

```bash
# Reduce batch size
./scripts/train.sh unet --batch-size 8 --memory-limit 2

# Use CPU
./scripts/train.sh reversed_ae --cpu
```

### 2. Data Issues

```bash
# Test với sample nhỏ
./scripts/train.sh unet --samples 100 --epochs 2

# Check data source
./scripts/train.sh reversed_ae --data-source kaggle
```

### 3. GPU Issues

```bash
# Check GPU availability
nvidia-smi

# Force CPU nếu GPU có vấn đề
./scripts/train.sh unet --cpu
```

## Best Practices

### 1. Recommended Settings

**U-Net (baseline):**
```bash
./scripts/train.sh unet \
  --data-source huggingface \
  --samples 3000 \
  --epochs 25 \
  --batch-size 32 \
  --gpu \
  --memory-limit 8
```

**Reversed AE (experimental):**
```bash
./scripts/train.sh reversed_ae \
  --data-source huggingface \
  --samples 2000 \
  --epochs 30 \
  --batch-size 16 \
  --gpu \
  --memory-limit 6
```

### 2. Development vs Production

**Development/Testing:**
- Samples: 500-1000
- Epochs: 5-10
- Batch size: 16-32

**Production:**
- Samples: 2000-5000
- Epochs: 20-50
- Batch size: 32-64

### 3. Model Comparison

Để so sánh hiệu suất:
```bash
# Train cả hai với cùng settings
./scripts/train.sh both \
  --data-source huggingface \
  --samples 2000 \
  --epochs 25 \
  --batch-size 32 \
  --gpu

# Sau đó evaluate
./scripts/evaluate.sh
./scripts/compare.sh
```

## Evaluation sau Training

Sau khi train xong, sử dụng các script evaluation:

```bash
# Evaluate single model
./scripts/evaluate.sh outputs/checkpoints/unet_best.pth

# Compare models
./scripts/compare.sh outputs/checkpoints/unet_best.pth outputs/checkpoints/reversed_ae_best.pth

# View results trong Jupyter
./scripts/jupyter.sh
```