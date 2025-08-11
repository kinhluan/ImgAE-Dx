# ImgAE-Dx Quick Start Guide

🚀 **Complete guide for ImgAE-Dx medical image anomaly detection project**

## 📋 Prerequisites

- **Python 3.8.1+** installed
- **Poetry** package manager ([Install Poetry](https://python-poetry.org/docs/#installation))
- **Kaggle account** with API credentials (optional, for real datasets)
- **W&B account** (optional, for experiment tracking)

## ⚡ Quick Setup (3 minutes)

### 1. Clone and Initialize

```bash
# Clone the repository
git clone https://github.com/kinhluan/ImgAE-Dx.git
cd ImgAE-Dx

# Run complete setup (installs dependencies, sets up environment)
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 2. Configure API Keys (Optional)

```bash
# Add your Kaggle credentials (for real datasets)
cp configs/kaggle.json.template configs/kaggle.json
# Edit kaggle.json: {"username": "your_username", "key": "your_api_key"}

# Copy to standard Kaggle location
mkdir -p ~/.kaggle
cp configs/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Add W&B API key (for experiment tracking)
cp configs/wandb.json.template configs/wandb.json  
# Edit wandb.json: {"api_key": "your_wandb_api_key"}

# Get API keys from:
# - Kaggle: https://www.kaggle.com/settings/account (API section)
# - W&B: https://wandb.ai/settings (API Keys section)
```

### 3. Verify Installation

```bash
# Validate project setup and run tests
poetry run python scripts/validate_project.py

# Or run comprehensive test suite
./scripts/test.sh --all
```

## 🚀 Core Scripts Guide

### 🏋️ Training Scripts

#### Basic Model Training

```bash
# Train U-Net (baseline model)
./scripts/train.sh unet --samples 100 --epochs 5

# Train Reversed Autoencoder (experimental model)  
./scripts/train.sh reversed_ae --samples 100 --epochs 5

# Train both models sequentially
./scripts/train.sh both --samples 200 --epochs 10
```

#### Advanced Training Options

```bash
# Training with W&B logging
./scripts/train.sh unet --samples 500 --epochs 15 --wandb-project my-experiment

# Training without W&B (local only)
./scripts/train.sh unet --samples 500 --epochs 15 --no-wandb

# Resume from checkpoint
./scripts/train.sh unet --resume checkpoints/UNet_best.pth --epochs 10

# Custom batch size and memory limits
./scripts/train.sh unet --batch-size 16 --memory-limit 2 --samples 200

# Production training (large dataset)
./scripts/train.sh unet --samples 5000 --epochs 30 --batch-size 64
```

#### Training Parameters

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `MODEL_TYPE` | Model to train: `unet`, `reversed_ae`, `both` | - | `unet` |
| `--samples NUM` | Number of samples to use | 2000 | `--samples 100` |
| `--epochs NUM` | Training epochs | 10 | `--epochs 20` |
| `--batch-size NUM` | Batch size | 32 | `--batch-size 16` |
| `--no-wandb` | Disable W&B logging | false | `--no-wandb` |
| `--resume PATH` | Resume from checkpoint | - | `--resume checkpoints/model.pth` |
| `--memory-limit GB` | Memory limit | 4 | `--memory-limit 8` |

### 🧪 Testing Scripts

#### Comprehensive Testing

```bash
# Run all tests (unit + integration + validation)
./scripts/test.sh --all

# Run only unit tests
./scripts/test.sh --unit

# Run only integration tests  
./scripts/test.sh --integration

# Run tests with coverage report
./scripts/test.sh --coverage

# Run specific test categories
./scripts/test.sh --models     # Test model architectures
./scripts/test.sh --streaming  # Test data pipeline
./scripts/test.sh --training   # Test training system
```

#### Project Validation

```bash
# Comprehensive project validation (imports, models, configs, etc.)
poetry run python scripts/validate_project.py

# Quick functionality check
poetry run python -c "from imgae_dx.models import UNet; print('✅ Import successful')"
```

### 🔧 Development Scripts

#### Jupyter Notebooks

```bash
# Start Jupyter Lab server (best for development)
./scripts/jupyter.sh

# Start Jupyter Lab on specific port
./scripts/jupyter.sh --port 8889

# Start with specific notebook directory  
./scripts/jupyter.sh --notebook-dir notebooks/research/
```

#### Environment Setup

```bash
# Complete project setup (run once)
./scripts/setup.sh

# Reinstall dependencies only
./scripts/setup.sh --deps-only  

# Setup with development tools
./scripts/setup.sh --dev
```

## 📊 Real-World Usage Examples

### 🔬 Research Workflow

```bash
# 1. Quick experiment with small dataset
./scripts/train.sh unet --samples 50 --epochs 3 --no-wandb

# 2. Check results
ls checkpoints/  # View saved models
ls logs/         # Check training logs

# 3. Interactive analysis
./scripts/jupyter.sh  # Open notebooks for analysis
```

### 🏭 Production Workflow

```bash
# 1. Full model training with tracking
./scripts/train.sh both --samples 2000 --epochs 25 --wandb-project production

# 2. Model comparison
# View W&B dashboard for comparison: https://wandb.ai/your-username/production

# 3. Best model deployment
# Use checkpoints/[ModelName]_best.pth for deployment
```

### 🧪 Experimentation Workflow

```bash
# 1. Test different configurations
./scripts/train.sh unet --samples 100 --epochs 5 --batch-size 16 --wandb-project exp-1
./scripts/train.sh unet --samples 100 --epochs 5 --batch-size 32 --wandb-project exp-2

# 2. Compare models
./scripts/train.sh unet --samples 200 --epochs 10 --wandb-project model-comparison
./scripts/train.sh reversed_ae --samples 200 --epochs 10 --wandb-project model-comparison

# 3. Hyperparameter tuning
for lr in 1e-3 1e-4 1e-5; do
  ./scripts/train.sh unet --samples 100 --epochs 5 --wandb-project hp-tuning --config configs/custom_lr_${lr}.yaml
done
```

## 🎯 Quick Commands Reference

### Essential Commands

```bash
# Setup & Installation
./scripts/setup.sh                          # Complete project setup
poetry install                              # Install dependencies only
poetry shell                                # Activate virtual environment

# Training  
./scripts/train.sh unet                     # Train U-Net
./scripts/train.sh reversed_ae               # Train Reversed AE
./scripts/train.sh both                     # Train both models

# Testing & Validation
./scripts/test.sh --all                     # Run complete test suite
poetry run python scripts/validate_project.py  # Validate project

# Development
./scripts/jupyter.sh                        # Start Jupyter Lab
poetry run python -m imgae_dx.cli.train --help  # CLI help
```

### Useful One-liners

```bash
# Quick model test
poetry run python -c "from imgae_dx.models import UNet; m=UNet(); print(f'U-Net: {m.count_parameters():,} parameters')"

# Check training progress
tail -f logs/unet_*.log

# View latest results
ls -la checkpoints/ | tail -5

# Test Kaggle API
poetry run python -c "import kaggle; kaggle.api.authenticate(); print('✅ Kaggle API working')"

# Test W&B connection  
poetry run python -c "import wandb; print('✅ W&B available')"
```

## 📁 Project Structure

```
ImgAE-Dx/
├── src/imgae_dx/           # Main package
│   ├── models/             # U-Net, Reversed AE architectures
│   ├── training/           # Training system with W&B integration
│   ├── streaming/          # Kaggle data streaming pipeline
│   ├── data/               # Data processing & transforms
│   ├── utils/              # Configuration management
│   └── cli/                # Command-line interfaces
├── scripts/                # Automation scripts ⭐
│   ├── setup.sh            # Project setup
│   ├── train.sh            # Model training
│   ├── test.sh             # Testing suite  
│   ├── jupyter.sh          # Jupyter Lab launcher
│   └── validate_project.py # Project validation
├── configs/                # Configuration files
├── checkpoints/            # Trained model checkpoints
├── logs/                   # Training logs
└── tests/                  # Test suite
```

## 🔧 Configuration

### Project Config (`configs/project_config.yaml`)

```yaml
# Model settings
model:
  input_channels: 1
  input_size: 128
  latent_dim: 512

# Training settings  
training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-4
  optimizer: "adam"

# Data settings
data:
  num_normal_samples: 2000
  image_size: 128
  enable_augmentation: true

# Streaming settings
streaming:
  memory_limit_gb: 4
  dataset_name: "bachrr/covid-chest-xray"
  dataset_stages: ["images"]
```

## 🔍 Troubleshooting

### Common Issues

#### Poetry/Installation Issues

```bash
# Check Poetry installation
poetry --version

# Reinstall dependencies
poetry install --no-cache

# Clear cache and reinstall
poetry cache clear . --all && poetry install
```

#### Memory Issues

```bash
# Use smaller batch sizes
./scripts/train.sh unet --batch-size 8 --samples 100

# Monitor memory usage
# macOS: Activity Monitor
# Linux: htop or free -h
```

#### Kaggle API Issues  

```bash
# Test Kaggle authentication
poetry run python -c "import kaggle; kaggle.api.authenticate(); print('✅ OK')"

# Check credentials file
ls -la ~/.kaggle/kaggle.json
cat ~/.kaggle/kaggle.json  # Should show {"username": "...", "key": "..."}

# Redownload credentials from https://www.kaggle.com/settings/account
```

#### Training Issues

```bash
# Check logs for errors
tail -f logs/unet_*.log

# Reduce sample size for testing
./scripts/train.sh unet --samples 10 --epochs 1 --no-wandb

# Validate project setup
poetry run python scripts/validate_project.py
```

## 📊 Expected Results

### Model Performance

| Model | Parameters | Training Time* | Expected AUC | Memory Usage |
|-------|------------|---------------|--------------|--------------|
| **U-Net** | ~55M | 15-30 min | 0.85-0.90 | 2-3GB |
| **Reversed AE** | ~273M | 30-60 min | 0.80-0.85 | 3-4GB |

*Times based on 2000 samples, 10 epochs on Apple Silicon M1

### Training Outputs

```
checkpoints/
├── UNet_best.pth                 # Best U-Net model
├── ReversedAutoencoder_best.pth   # Best Reversed AE model  
├── unet_learning_curves.png       # Training progress plots
└── reversed_ae_learning_curves.png

logs/
├── unet_20250111_120000.log      # Training logs
└── reversed_ae_20250111_130000.log
```

## 🎓 Next Steps

1. **Start Experimenting**: Run `./scripts/train.sh unet --samples 50 --epochs 2 --no-wandb`
2. **Explore Notebooks**: `./scripts/jupyter.sh` and check `notebooks/`
3. **Modify Configs**: Edit `configs/project_config.yaml` for your needs
4. **Track Experiments**: Set up W&B and use `--wandb-project` flag
5. **Scale Up**: Increase `--samples` and `--epochs` for production runs
6. **Compare Models**: Train both architectures and compare results

## 📞 Support

- **Documentation**: Full docs in `docs/` folder
- **CLI Help**: `poetry run imgae-train --help` or `poetry run imgae-evaluate --help`  
- **API Reference**: `poetry run python -c "help(imgae_dx)"`
- **Issues**: Report bugs via GitHub Issues

---

🎯 **Ready to detect medical anomalies? Start with:** `./scripts/train.sh unet --samples 100 --epochs 5`

⭐ **Happy anomaly detection!**
