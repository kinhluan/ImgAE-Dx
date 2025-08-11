# ImgAE-Dx Quick Start Guide

🚀 **Fast setup guide for ImgAE-Dx medical image anomaly detection project**

## 📋 Prerequisites

- Python 3.8.1+ installed
- Poetry package manager
- Kaggle account with API credentials
- W&B account (optional, for experiment tracking)

## ⚡ Quick Setup (5 minutes)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/kinhluan/ImgAE-Dx.git
cd ImgAE-Dx

# Run automatic setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 2. Configure API Keys

```bash
# Add your Kaggle credentials
cp configs/kaggle.json.template configs/kaggle.json
# Edit kaggle.json with your username and key

# Add W&B API key (optional)
cp configs/wandb.json.template configs/wandb.json
# Edit wandb.json with your API key
```

### 3. Quick Test

```bash
# Test the installation
poetry run python scripts/test_setup.py
```

## 🚀 Usage Examples

### Basic Model Training

```bash
# Train U-Net on sample data
./scripts/train.sh unet --samples 100

# Train Reversed Autoencoder
./scripts/train.sh reversed_ae --samples 100

# Compare both models
./scripts/compare.sh --samples 500
```

### Interactive Development

```bash
# Start Jupyter Lab
./scripts/jupyter.sh

# Or activate environment and work directly
poetry shell
python -c "from imgae_dx.models import UNet; print('✅ Models imported successfully!')"
```

### Production Training

```bash
# Full training with streaming data
./scripts/train_full.sh --config configs/production.yaml
```

## 📁 Project Structure

```
ImgAE-Dx/
├── src/imgae_dx/           # Main package
│   ├── models/             # U-Net, Reversed AE
│   ├── streaming/          # Kaggle data streaming
│   ├── data/              # Data processing
│   └── utils/             # Configuration, utilities
├── configs/               # Configuration files
├── scripts/              # Automation scripts
├── notebooks/            # Research notebooks
└── tests/               # Test suite
```

## 🔧 Configuration

### Basic Config (`configs/project_config.yaml`)

```yaml
data:
  num_normal_samples: 2000
  num_abnormal_samples: 1000
  image_size: 128

training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-4

model:
  input_channels: 1
  latent_dim: 512
```

### Memory Settings

```yaml
streaming:
  memory_limit_gb: 4
  batch_size: 32
  cache_size_mb: 512
```

## 🎯 Common Workflows

### 1. Research Development

```bash
# Quick experiment with small dataset
./scripts/experiment.sh --model unet --samples 100 --epochs 5

# View results in notebook
./scripts/jupyter.sh
```

### 2. Model Comparison

```bash
# Compare U-Net vs Reversed AE
./scripts/compare_models.sh --dataset small --metrics auc,reconstruction_error

# Generate comparison report
./scripts/generate_report.sh results/comparison/
```

### 3. Production Training

```bash
# Full dataset training with checkpointing
./scripts/train_production.sh --resume-from checkpoints/unet_epoch_10.pth
```

## 🔍 Troubleshooting

### Poetry Installation Issues

```bash
# Check Poetry version
poetry --version

# Reinstall dependencies
poetry install --no-cache
```

### Memory Issues

```bash
# Reduce batch size in config
./scripts/config.sh set training.batch_size 16

# Monitor memory usage
./scripts/monitor.sh
```

### Kaggle API Issues

```bash
# Test Kaggle connection
poetry run python -c "import kaggle; kaggle.api.authenticate(); print('✅ Kaggle API working')"

# Check credentials
ls -la ~/.kaggle/kaggle.json
```

## 📊 Expected Results

### U-Net Baseline

- **Training time**: ~30 minutes (2000 samples)
- **Expected AUC**: 0.85-0.90
- **Memory usage**: ~2-3GB

### Reversed Autoencoder

- **Training time**: ~25 minutes (2000 samples)  
- **Expected AUC**: 0.80-0.85
- **Memory usage**: ~1.5-2GB

## 🚨 Quick Commands Reference

```bash
# Setup and installation
./scripts/setup.sh                    # Complete setup
./scripts/install.sh                  # Install dependencies only

# Training
./scripts/train.sh unet               # Train U-Net
./scripts/train.sh reversed_ae        # Train Reversed AE
./scripts/train_both.sh               # Train both models

# Development  
./scripts/jupyter.sh                  # Start Jupyter Lab
./scripts/test.sh                     # Run tests
./scripts/lint.sh                     # Code quality checks

# Utilities
./scripts/config.sh validate          # Validate configuration
./scripts/monitor.sh                  # Monitor system resources
./scripts/clean.sh                    # Clean temporary files
```

## 🎓 Next Steps

1. **Explore notebooks**: Check `notebooks/research/` for detailed examples
2. **Modify config**: Adjust `configs/project_config.yaml` for your needs
3. **Add custom models**: Extend `src/imgae_dx/models/`
4. **Run experiments**: Use W&B integration for tracking

## 📞 Support

- **Documentation**: Check `docs/` folder
- **Issues**: GitHub Issues
- **Config help**: `poetry run imgae-config --help`
- **API reference**: `poetry run python -c "help(imgae_dx)"`

---
⭐ **Happy anomaly detection!**
