# ImgAE-Dx Poetry Architecture

## Modern Python Package Structure with Poetry

### 🏗️ Core Architecture (Updated for Poetry)

```
imgae-dx/                        # Poetry project root
├── pyproject.toml              # Poetry configuration
├── README.md                   # Project documentation
├── CHANGELOG.md                # Version history
│
├── src/
│   └── imgae_dx/               # Main package (importable)
│       ├── __init__.py         # Package initialization
│       ├── cli/                # Command-line interface
│       │   ├── __init__.py
│       │   ├── train.py        # Training commands
│       │   ├── evaluate.py     # Evaluation commands
│       │   ├── config.py       # Config management commands
│       │   └── notebook.py     # Notebook utilities
│       ├── streaming/          # Cloud-native data streaming
│       │   ├── __init__.py
│       │   ├── kaggle_client.py      # Kaggle API wrapper
│       │   ├── stream_loader.py      # Streaming data loader
│       │   └── memory_manager.py     # Smart caching system
│       ├── data/               # Data processing pipeline
│       │   ├── __init__.py
│       │   ├── streaming_dataset.py  # PyTorch Dataset with streaming
│       │   ├── transforms.py         # Image preprocessing
│       │   └── batch_processor.py    # Memory-efficient batching
│       ├── models/             # Model architectures
│       │   ├── __init__.py
│       │   ├── base.py         # Base model interface
│       │   ├── unet.py         # U-Net architecture
│       │   └── reversed_ae.py  # Reversed Autoencoder
│       ├── training/           # Training system
│       │   ├── __init__.py
│       │   ├── trainer.py      # Base trainer class
│       │   ├── streaming_trainer.py # Cloud-native trainer
│       │   ├── evaluator.py    # Model evaluation
│       │   └── metrics.py      # Custom metrics (AUC, etc.)
│       ├── utils/              # Utilities and helpers
│       │   ├── __init__.py
│       │   ├── config_manager.py     # Configuration management (existing)
│       │   ├── checkpoint_manager.py # Model checkpointing
│       │   └── logging_utils.py      # Logging and W&B integration
│       └── visualization/      # Plotting and visualization
│           ├── __init__.py
│           ├── plots.py        # Standard plotting utilities
│           ├── streaming_plots.py    # Real-time visualization
│           └── medical_viz.py  # Medical imaging specific plots
│
├── configs/                    # Configuration files (existing)
│   ├── project_config.yaml     # Main project configuration
│   ├── kaggle.json            # Kaggle API credentials
│   ├── wandb.json             # W&B API credentials  
│   └── model_configs/         # Model-specific configurations
│       ├── unet.yaml
│       └── reversed_ae.yaml
│
├── scripts/                   # Executable scripts
│   ├── __init__.py
│   ├── train_streaming.py     # Main training script
│   ├── evaluate_models.py     # Model evaluation script
│   ├── setup_environment.py   # Environment setup
│   └── generate_notebooks.py  # Auto-generate research notebooks
│
├── notebooks/                 # Research notebooks (existing)
│   ├── research/              # Original research notebooks
│   │   ├── Anomaly_Detection_Research.ipynb
│   │   └── Anomaly_Detection_Research_Colab.ipynb
│   ├── examples/             # Example notebooks
│   │   ├── quick_start.ipynb
│   │   └── advanced_usage.ipynb
│   └── generated/            # Auto-generated notebooks
│       └── streaming_demo.ipynb
│
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── conftest.py          # Pytest configuration
│   ├── unit/                # Unit tests
│   │   ├── test_models.py
│   │   ├── test_data.py
│   │   └── test_utils.py
│   ├── integration/         # Integration tests
│   │   ├── test_streaming.py
│   │   └── test_training.py
│   └── fixtures/           # Test data and fixtures
│       ├── sample_images/
│       └── test_configs/
│
├── docs/                   # Project documentation (existing)
│   ├── api/               # Auto-generated API docs
│   ├── tutorials/         # User tutorials
│   ├── DEVELOPMENT_TODO.md # Development roadmap
│   ├── PROJECT_JOURNEY.md  # Research methodology
│   └── ...               # Other existing docs
│
└── .github/              # GitHub workflows (optional)
    └── workflows/
        ├── test.yml      # CI/CD testing
        ├── publish.yml   # Package publishing
        └── docs.yml      # Documentation building
```

---

## 🔄 Streaming Data Flow

### 1. **Kaggle API Integration**

```python
class KaggleStreamClient:
    def __init__(self, dataset="nih-chest-xray/data"):
        self.api = kaggle.api
        self.dataset = dataset
        
    def stream_zip_content(self, filename="images_001.zip"):
        """Stream ZIP content directly to memory"""
        # Download to temporary buffer
        temp_buffer = self.api.dataset_download_file(
            self.dataset, filename, 
            path="memory://", unzip=False
        )
        
        # Return ZipFile object for streaming access
        return zipfile.ZipFile(io.BytesIO(temp_buffer))
        
    def get_progressive_streams(self):
        """Get multiple stage streams progressively"""
        stages = ["images_001.zip", "images_002.zip", "images_003.zip"]
        for stage in stages:
            yield self.stream_zip_content(stage)
```

### 2. **Streaming Dataset Implementation**

```python
class StreamingNIHDataset(Dataset):
    def __init__(self, kaggle_client, stage="images_001", transform=None):
        self.kaggle_client = kaggle_client
        self.stage = stage
        self.transform = transform
        
        # Stream metadata first
        self.metadata = self._load_metadata_streaming()
        self.image_list = self._build_image_index()
        
        # Initialize streaming ZIP
        self.current_zip = None
        self.zip_cache = {}  # Smart caching for frequently accessed images
        
    def __getitem__(self, idx):
        img_info = self.image_list[idx]
        
        # Get image from streaming source
        image = self._get_image_streaming(img_info['filename'])
        label = img_info['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
        
    def _get_image_streaming(self, filename):
        """Stream individual image from Kaggle ZIP"""
        if self.current_zip is None:
            self.current_zip = self.kaggle_client.stream_zip_content(
                f"{self.stage}.zip"
            )
            
        # Extract image from ZIP stream
        with self.current_zip.open(filename) as img_file:
            image = Image.open(img_file).convert('L')
            
        return image
```

### 3. **Memory-Efficient Training Loop**

```python
class StreamingTrainer:
    def __init__(self, model, kaggle_client, config):
        self.model = model
        self.kaggle_client = kaggle_client
        self.config = config
        self.memory_manager = StreamingMemoryManager()
        
    def train_progressive_stages(self):
        stages = ["images_001", "images_002", "images_003"]
        
        for stage_idx, stage in enumerate(stages):
            print(f"Training Stage {stage_idx + 1}: {stage}")
            
            # Create streaming dataset for current stage
            dataset = StreamingNIHDataset(
                self.kaggle_client, 
                stage=stage,
                transform=self.get_transforms()
            )
            
            dataloader = DataLoader(
                dataset, 
                batch_size=self.config.batch_size,
                num_workers=0  # Single process for streaming
            )
            
            # Train with current stage
            stage_metrics = self.train_single_stage(
                dataloader, 
                stage_name=stage,
                resume_from=f"stage_{stage_idx}.pth" if stage_idx > 0 else None
            )
            
            # Save stage checkpoint to cloud
            self.save_cloud_checkpoint(f"stage_{stage_idx + 1}.pth", stage_metrics)
            
            # Memory cleanup
            self.memory_manager.cleanup_stage(stage)
            
    def train_single_stage(self, dataloader, stage_name, resume_from=None):
        if resume_from and self.cloud_checkpoint_exists(resume_from):
            self.load_cloud_checkpoint(resume_from)
            
        for epoch in range(self.config.epochs_per_stage):
            for batch_idx, (images, labels) in enumerate(dataloader):
                # Standard training loop
                loss = self.train_batch(images, labels)
                
                # Real-time logging to W&B
                self.log_streaming_metrics(loss, batch_idx, stage_name)
                
                # Memory management
                if batch_idx % 100 == 0:
                    self.memory_manager.check_memory_usage()
                    
        return {"stage": stage_name, "final_loss": loss}
```

---

## ☁️ Cloud-Native Features

### 1. **Google Colab Integration**

```python
class ColabStreamingSetup:
    def __init__(self):
        self.setup_kaggle_auth()
        self.setup_drive_integration()
        self.setup_gpu_optimization()
        
    def setup_kaggle_auth(self):
        """Setup Kaggle API in Colab"""
        from google.colab import files
        
        # Auto-upload kaggle.json if not exists
        if not os.path.exists('/root/.kaggle/kaggle.json'):
            print("Please upload your kaggle.json file:")
            uploaded = files.upload()
            
        # Configure permissions
        os.chmod('/root/.kaggle/kaggle.json', 600)
        
    def setup_drive_integration(self):
        """Mount Google Drive for checkpoints"""
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Create checkpoint directory
        self.checkpoint_dir = '/content/drive/MyDrive/ImgAE-Dx-Checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
```

### 2. **Real-Time Visualization**

```python
class StreamingVisualizer:
    def __init__(self, wandb_project="imgae-dx-streaming"):
        import wandb
        self.wandb = wandb
        self.wandb.init(project=wandb_project)
        
    def log_streaming_progress(self, metrics, stage, batch_idx):
        """Log real-time training progress"""
        self.wandb.log({
            f"{stage}/loss": metrics['loss'],
            f"{stage}/batch": batch_idx,
            "memory_usage": self.get_memory_usage(),
            "streaming_speed": metrics.get('samples_per_sec', 0)
        })
        
    def visualize_streaming_comparison(self, unet_metrics, ra_metrics):
        """Real-time model comparison"""
        comparison_table = self.wandb.Table(
            columns=["Stage", "Model", "AUC", "Loss"],
            data=[
                ["Current", "U-Net", unet_metrics['auc'], unet_metrics['loss']],
                ["Current", "RA", ra_metrics['auc'], ra_metrics['loss']]
            ]
        )
        self.wandb.log({"model_comparison": comparison_table})
```

### 3. **Automatic Notebook Generation**

```python
class StreamingNotebookGenerator:
    def generate_colab_notebook(self):
        """Generate research-ready Colab notebook"""
        notebook = {
            "cells": [
                self.create_setup_cell(),
                self.create_streaming_demo_cell(),
                self.create_training_cell(),
                self.create_evaluation_cell(),
                self.create_visualization_cell()
            ]
        }
        
        # Save to notebooks/generated/
        with open("notebooks/generated/streaming_research.ipynb", "w") as f:
            json.dump(notebook, f, indent=2)
            
    def create_streaming_demo_cell(self):
        return {
            "cell_type": "code",
            "source": [
                "# Streaming Demo: Direct Kaggle Access",
                "kaggle_client = KaggleStreamClient('nih-chest-xray/data')",
                "",
                "# Stream images_001 directly",
                "dataset = StreamingNIHDataset(kaggle_client, stage='images_001')",
                "print(f'Streaming {len(dataset)} images from Kaggle')",
                "",
                "# Show streaming progress",
                "for i in tqdm(range(0, 100, 10)):",
                "    img, label = dataset[i]",
                "    print(f'Streamed image {i}: shape={img.shape}, label={label}')"
            ]
        }
```

---

## 🚀 Poetry Usage Examples

### Development Workflow

```bash
# Install project with Poetry
poetry install --with dev,cloud,viz

# Activate virtual environment
poetry shell

# CLI commands through Poetry
poetry run imgae-train --config configs/project_config.yaml --stage images_001
poetry run imgae-evaluate --model checkpoints/unet_best.pth
poetry run imgae-config --validate-apis

# Run tests
poetry run pytest tests/ -v --cov

# Code quality checks
poetry run black src/ tests/
poetry run isort src/ tests/
poetry run mypy src/

# Build and publish
poetry build
poetry publish --repository pypi
```

### Google Colab with Poetry

```python
# Install from PyPI (after publishing)
!pip install imgae-dx[cloud]

# Or install from GitHub (development)
!pip install git+https://github.com/user/imgae-dx.git

from imgae_dx.streaming import ColabStreamingSetup, StreamingTrainer
from imgae_dx.cli import setup_colab_environment

# Auto-setup Colab environment
setup_colab_environment()

# Stream and train directly  
trainer = StreamingTrainer.from_config("configs/project_config.yaml")
results = trainer.train_progressive_stages()

print(f"Training complete! AUC scores: {results}")
```

### Package Import Structure

```python
# Clean imports with Poetry structure
from imgae_dx.models import UNet, ReversedAutoencoder
from imgae_dx.data import StreamingNIHDataset
from imgae_dx.training import StreamingTrainer, Evaluator
from imgae_dx.utils import ConfigManager
from imgae_dx.visualization import plot_roc_curve, visualize_anomalies

# Or use factory patterns
from imgae_dx import create_model, create_trainer, load_config

config = load_config("configs/project_config.yaml")
model = create_model("unet", config.model)
trainer = create_trainer("streaming", model, config.training)
```

---

## 📊 Poetry Architecture Benefits

### Package Management

- **Dependency Resolution**: Automatic conflict resolution with lock file
- **Virtual Environments**: Isolated environments per project
- **Version Pinning**: Reproducible builds across environments
- **Optional Dependencies**: Flexible installation (cloud, viz, performance groups)

### Development Experience

- **CLI Integration**: Built-in commands via `poetry run imgae-train`
- **Testing Framework**: Integrated pytest with coverage
- **Code Quality**: Black, isort, mypy integration
- **Documentation**: Sphinx auto-generation from docstrings

### Distribution & Publishing

- **PyPI Ready**: One-command publishing to package indexes
- **GitHub Integration**: Direct pip install from repository
- **Semantic Versioning**: Automated version management
- **Cross-Platform**: Works on Windows, macOS, Linux

### Cloud Integration

- **Colab Compatibility**: Easy pip install in notebooks
- **Memory Efficiency**: ~500MB peak memory usage with streaming
- **GPU Optimization**: Continuous GPU utilization
- **Checkpoint Management**: Auto-sync with Google Drive

### Professional Development

- **CI/CD Ready**: GitHub Actions integration
- **Type Safety**: Full mypy type checking
- **Test Coverage**: Automated coverage reporting  
- **Documentation**: Auto-generated API docs

This modern Poetry architecture provides enterprise-grade package management while maintaining the streaming performance and cloud-native features!
