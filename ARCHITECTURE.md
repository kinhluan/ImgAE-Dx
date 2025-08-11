# ImgAE-Dx Architecture Documentation

🏗️ **Professional medical image anomaly detection system built with Poetry**

## 📋 Project Overview

ImgAE-Dx is a production-ready research project that compares U-Net and Reversed Autoencoder architectures for unsupervised anomaly detection in medical images. The system uses modern Python packaging with Poetry and integrates with cloud services for scalable training.

## 🏗️ Core Architecture

### Package Structure

```
ImgAE-Dx/
├── pyproject.toml              # Poetry configuration & dependencies
├── README.md                   # Project documentation
├── QUICK_START.md             # Getting started guide
├── CLAUDE.md                  # AI assistant instructions
│
├── src/imgae_dx/              # Main package (Poetry src layout)
│   ├── __init__.py
│   ├── models/                # Model architectures
│   │   ├── __init__.py
│   │   ├── base.py           # Base model interface
│   │   ├── unet.py           # U-Net implementation (55M params)
│   │   └── reversed_ae.py    # Reversed Autoencoder (273M params)
│   │
│   ├── data/                 # Data processing pipeline
│   │   ├── __init__.py
│   │   ├── streaming_dataset.py    # Kaggle streaming dataset
│   │   ├── transforms.py          # Medical image transforms
│   │   └── __init__.py
│   │
│   ├── streaming/            # Data streaming from Kaggle
│   │   ├── __init__.py
│   │   ├── kaggle_client.py       # Kaggle API integration
│   │   └── memory_manager.py      # Memory management
│   │
│   ├── training/             # Training system
│   │   ├── __init__.py
│   │   ├── trainer.py             # Core training logic
│   │   ├── evaluator.py           # Model evaluation
│   │   └── metrics.py             # Custom metrics (AUC, etc.)
│   │
│   ├── utils/                # Utilities
│   │   ├── __init__.py
│   │   └── config_manager.py      # Configuration management
│   │
│   └── cli/                  # Command-line interfaces
│       ├── __init__.py
│       ├── train.py               # Training CLI
│       └── evaluate.py            # Evaluation CLI
│
├── configs/                   # Configuration files
│   ├── project_config.yaml        # Main project configuration
│   ├── kaggle.json.template       # Kaggle API template
│   └── wandb.json.template        # W&B API template
│
├── scripts/                   # Automation scripts
│   ├── setup.sh                   # Project setup
│   ├── train.sh                   # Model training
│   ├── test.sh                    # Testing suite
│   ├── jupyter.sh                 # Jupyter launcher
│   └── validate_project.py        # Project validation
│
├── tests/                     # Test suite (25 tests)
│   ├── conftest.py
│   ├── unit/                      # Unit tests
│   ├── integration/               # Integration tests
│   └── fixtures/                  # Test data
│
├── checkpoints/               # Model checkpoints & learning curves
├── logs/                     # Training logs
└── notebooks/                # Research notebooks
    ├── Anomaly_Detection_Research.ipynb
    └── Anomaly_Detection_Research_Colab.ipynb
```

## 🔧 Key Components

### 1. Model Architectures

#### U-Net (Baseline)

- **Parameters**: 54,986,305 (~55M)
- **Architecture**: Encoder-decoder with skip connections
- **Purpose**: Baseline model for comparison
- **Expected AUC**: 0.85-0.90

```python
from imgae_dx.models import UNet

model = UNet(
    input_channels=1,
    input_size=128,
    latent_dim=512
)
```

#### Reversed Autoencoder (Experimental)

- **Parameters**: 272,717,697 (~273M)
- **Architecture**: Asymmetric encoder-decoder without skip connections
- **Purpose**: Experimental "pseudo-healthy" reconstruction approach
- **Expected AUC**: 0.80-0.85

```python
from imgae_dx.models import ReversedAutoencoder

model = ReversedAutoencoder(
    input_channels=1,
    input_size=128,
    latent_dim=512
)
```

### 2. Data Pipeline

#### Kaggle Streaming Integration

- **Dataset**: NIH Chest X-ray dataset
- **Streaming**: Direct download from Kaggle API
- **Memory Management**: Efficient caching and cleanup
- **Fallback**: Dummy data generation for testing

```python
from imgae_dx.streaming import KaggleStreamClient
from imgae_dx.data import create_streaming_dataloaders

client = KaggleStreamClient("bachrr/covid-chest-xray")
train_loader, val_loader, info = create_streaming_dataloaders(
    kaggle_client=client,
    batch_size=32,
    max_samples=2000
)
```

#### Medical Image Transforms

- **Preprocessing**: Grayscale conversion, resizing, normalization
- **Augmentation**: Rotation, flipping, brightness adjustment
- **Validation**: Consistent evaluation transforms

### 3. Training System

#### Trainer Class

- **Multi-device Support**: CPU, CUDA, MPS (Apple Silicon)
- **Checkpointing**: Best model and epoch-based saving
- **Early Stopping**: Configurable patience and delta
- **Memory Management**: Automatic cleanup and monitoring

```python
from imgae_dx.training import Trainer

trainer = Trainer(
    model=model,
    config=config,
    device="auto",
    wandb_project="imgae-dx"
)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    checkpoint_dir="checkpoints"
)
```

#### Weights & Biases Integration

- **Experiment Tracking**: Loss curves, metrics, model comparison
- **Configuration**: Automatic API key loading from `wandb.json`
- **Optional**: Can be disabled with `--no-wandb` flag

### 4. Configuration Management

#### Project Configuration (`project_config.yaml`)

```yaml
model:
  input_channels: 1
  input_size: 128
  latent_dim: 512

training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-4
  optimizer: "adam"

data:
  num_normal_samples: 2000
  image_size: 128
  enable_augmentation: true

streaming:
  memory_limit_gb: 4
  dataset_name: "bachrr/covid-chest-xray"
  dataset_stages: ["images"]
```

#### API Keys Management

- **Kaggle**: `configs/kaggle.json` for dataset access
- **W&B**: `configs/wandb.json` for experiment tracking
- **Security**: Files excluded from git, environment variable fallbacks

## 🚀 Usage Patterns

### 1. Training Workflow

```bash
# Quick training (development)
./scripts/train.sh unet --samples 100 --epochs 5 --no-wandb

# Production training (with tracking)
./scripts/train.sh both --samples 2000 --epochs 25 --wandb-project production

# Resume from checkpoint
./scripts/train.sh unet --resume checkpoints/UNet_best.pth --epochs 10
```

### 2. Development Workflow

```bash
# Setup project
./scripts/setup.sh

# Run tests
./scripts/test.sh --all

# Start Jupyter for analysis
./scripts/jupyter.sh

# Validate project
poetry run python scripts/validate_project.py
```

### 3. Research Workflow

```python
# Programmatic usage
from imgae_dx import create_model, load_config
from imgae_dx.training import Trainer

config = load_config("configs/project_config.yaml")
model = create_model("unet", config.model)
trainer = Trainer(model, config)

# Training with custom parameters
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20
)

# Model comparison
unet_results = trainer.evaluate(test_loader)
ra_results = trainer.evaluate(test_loader)  # After loading RA model
```

## 📊 Performance Characteristics

### Memory Usage

- **Streaming Dataset**: ~500MB peak memory usage
- **U-Net Training**: 2-3GB GPU/system memory
- **Reversed AE Training**: 3-4GB GPU/system memory
- **Batch Processing**: Automatic batch size adjustment for available memory

### Training Performance

- **U-Net**: ~15-30 minutes for 2000 samples, 10 epochs (Apple Silicon M1)
- **Reversed AE**: ~30-60 minutes for 2000 samples, 10 epochs
- **Streaming Speed**: ~10-50 samples/second depending on network and hardware

### Model Performance

- **U-Net AUC**: Typically 0.85-0.90 on test data
- **Reversed AE AUC**: Typically 0.80-0.85 on test data
- **Reconstruction Quality**: Measured via MSE and visual inspection

## 🔧 Integration Points

### Poetry Package Manager

- **Dependency Groups**: `dev`, `cloud`, `viz`, `performance`
- **CLI Commands**: `poetry run imgae-train`, `poetry run imgae-evaluate`
- **Virtual Environment**: Isolated dependencies
- **Distribution**: PyPI-ready package structure

### Cloud Integration

- **Google Colab**: Jupyter notebooks with GPU support
- **Kaggle API**: Direct dataset streaming
- **Google Drive**: Checkpoint persistence (in Colab)
- **W&B Cloud**: Experiment tracking and model comparison

### Testing Framework

- **Unit Tests**: Model architectures, data processing, utilities
- **Integration Tests**: Training pipeline, streaming, CLI
- **Coverage**: Full test coverage with pytest
- **Validation**: Project setup and configuration validation

## 🔒 Security & Best Practices

### API Key Management

- **Separation**: Keys stored in separate JSON files
- **Git Ignore**: Sensitive files excluded from version control
- **Templates**: Template files for easy setup
- **Validation**: Automatic key validation and error handling

### Code Quality

- **Type Hints**: Full type annotation throughout codebase
- **Documentation**: Comprehensive docstrings and examples
- **Error Handling**: Graceful failure and informative error messages
- **Memory Safety**: Automatic cleanup and resource management

### Production Readiness

- **Logging**: Structured logging with timestamps and levels
- **Monitoring**: Memory usage monitoring and alerts
- **Checkpointing**: Robust model saving and resumption
- **Configuration**: Environment-based configuration override

## 🎯 Design Principles

1. **Research-First**: Easy experimentation with comprehensive tracking
2. **Memory Efficient**: Stream large datasets without local storage requirements
3. **Production Ready**: Professional packaging and deployment capabilities
4. **Cloud Native**: Built for Google Colab and cloud environments
5. **Reproducible**: Deterministic results with comprehensive configuration management
6. **Extensible**: Modular architecture supporting new models and datasets

This architecture provides a solid foundation for medical image anomaly detection research while maintaining production-grade code quality and cloud integration capabilities.
