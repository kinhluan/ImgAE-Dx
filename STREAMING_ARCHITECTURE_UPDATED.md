# ImgAE-Dx Streaming Architecture (Updated)

## 🔑 Secure Configuration Management

### **Existing API Keys Integration**

The project already has API keys configured in:

- `configs/kaggle.json` - Kaggle API credentials
- `configs/wandb.md` - Weights & Biases API key

### **Secure Configuration System**

```python
# Centralized config management with existing keys
from imgae_dx.utils.config_manager import get_config_manager

config_manager = get_config_manager()

# Auto-loads existing API keys
kaggle_auth = config_manager.setup_kaggle_auth()  # Uses configs/kaggle.json
wandb_auth = config_manager.setup_wandb_auth()    # Uses configs/wandb.md

# Project configuration from YAML
config = config_manager.config
print(f"Dataset: {config.dataset_name}")
print(f"Stages: {config.dataset_stages}")
```

---

## 🏗️ Updated Architecture with Existing Keys

### **Project Structure**

```
src/imgae_dx/
├── streaming/
│   ├── kaggle_client.py        # Uses existing kaggle.json
│   ├── stream_loader.py        # Memory-efficient streaming
│   └── memory_manager.py       # Smart caching
├── utils/
│   ├── config_manager.py       # Secure config with existing keys
│   └── auth_handler.py         # Authentication management
├── training/
│   ├── streaming_trainer.py    # W&B integration with existing key
│   └── cloud_evaluator.py      # Cloud-native evaluation
└── colab/
    ├── setup_helper.py         # Auto Colab setup with keys
    └── notebook_generator.py   # Generate notebooks with auth
```

### **Configuration Architecture**

```yaml
# configs/project_config.yaml
api_keys:
  kaggle_config_path: "configs/kaggle.json"     # Existing file
  wandb_key_path: "configs/wandb.md"            # Existing file

dataset:
  name: "nih-chest-xray/data"
  stages: ["images_001.zip", "images_002.zip", "images_003.zip"]

streaming:
  memory_limit_gb: 4
  cache_frequently_accessed: true
  cleanup_after_stage: true
```

---

## 🔄 Streaming Implementation with Existing Keys

### **1. Kaggle Authentication**

```python
class KaggleStreamClient:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
        # Auto-setup using existing kaggle.json
        if not self.config_manager.setup_kaggle_auth():
            raise ValueError("Kaggle authentication failed")
            
        import kaggle
        self.api = kaggle.api
        
    def stream_zip_content(self, filename="images_001.zip"):
        """Stream ZIP directly using authenticated API"""
        return self.api.dataset_download_file(
            self.config_manager.config.dataset_name,
            filename,
            path="memory://", 
            unzip=False
        )
```

### **2. W&B Integration**  

```python
class StreamingTrainer:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
        # Auto-setup W&B using existing key
        if not self.config_manager.setup_wandb_auth():
            raise ValueError("W&B authentication failed")
            
        import wandb
        self.wandb = wandb
        self.wandb.init(project="imgae-dx-streaming")
        
    def train_with_streaming(self):
        # Training loop with automatic W&B logging
        for batch_idx, (images, labels) in enumerate(self.dataloader):
            loss = self.train_batch(images, labels)
            
            # Auto-log using existing W&B key
            self.wandb.log({"loss": loss, "batch": batch_idx})
```

### **3. Google Colab Auto-Setup**

```python
class ColabStreamingSetup:
    def __init__(self):
        self.config_manager = get_config_manager()
        
    def auto_setup_colab(self):
        """Complete Colab setup using existing keys"""
        
        # 1. Copy existing keys to Colab environment
        self._copy_keys_to_colab()
        
        # 2. Mount Google Drive
        from google.colab import drive
        drive.mount('/content/drive')
        
        # 3. Setup authentications
        kaggle_ok = self.config_manager.setup_kaggle_auth()
        wandb_ok = self.config_manager.setup_wandb_auth()
        
        # 4. Validate setup
        return {
            "kaggle_auth": kaggle_ok,
            "wandb_auth": wandb_ok,
            "drive_mounted": True,
            "gpu_available": torch.cuda.is_available()
        }
        
    def _copy_keys_to_colab(self):
        """Copy existing keys to Colab filesystem"""
        
        # Create Kaggle config directory
        os.makedirs('/root/.kaggle', exist_ok=True)
        
        # Copy kaggle.json to standard location  
        shutil.copy(
            'configs/kaggle.json',
            '/root/.kaggle/kaggle.json'
        )
        os.chmod('/root/.kaggle/kaggle.json', 600)
        
        # Set W&B key as environment variable
        with open('configs/wandb.md', 'r') as f:
            wandb_key = f.read().strip()
        os.environ['WANDB_API_KEY'] = wandb_key
```

---

## ⚡ Usage Examples with Existing Keys

### **Local Development**

```python
# Auto-loads existing keys from configs/
from imgae_dx.utils.config_manager import get_config_manager
from imgae_dx.streaming import KaggleStreamClient, StreamingTrainer

# Setup with existing credentials
config_manager = get_config_manager()
validation = config_manager.validate_setup()
print(f"Setup valid: {all(validation.values())}")

# Train with streaming
kaggle_client = KaggleStreamClient(config_manager)
trainer = StreamingTrainer(config_manager)
trainer.train_progressive_stages()
```

### **Google Colab (One Cell Setup)**

```python
# Upload project files including configs/ to Colab
# Then run:

from imgae_dx.colab import ColabStreamingSetup
from imgae_dx.streaming import StreamingTrainer

# Auto-setup everything using existing keys
setup = ColabStreamingSetup()
status = setup.auto_setup_colab()
print(f"Colab setup complete: {status}")

# Start training immediately
trainer = StreamingTrainer.from_colab_config()
results = trainer.train_progressive_stages()
```

### **Generated Research Notebook**

```python
# Auto-generate notebook with existing keys embedded
from imgae_dx.colab import NotebookGenerator

generator = NotebookGenerator()
notebook = generator.create_research_notebook(
    include_auth_setup=True,  # Include cells for key setup
    embed_credentials=False,  # Security: don't embed keys in notebook
    auto_mount_drive=True     # Auto Google Drive integration
)

# Save generated notebook
generator.save_notebook(notebook, "notebooks/generated/research_colab.ipynb")
```

---

## 🔒 Security Best Practices

### **Key Management**

- ✅ Keys stored in separate files (existing `configs/kaggle.json`, `configs/wandb.md`)
- ✅ `.gitignore` configured to exclude sensitive files
- ✅ Environment variable fallbacks for cloud deployment
- ✅ Validation and error handling for missing keys

### **Safe Configuration Loading**

```python
# Automatic validation and error handling
config_manager = get_config_manager()

if not config_manager.api_keys.validate():
    raise ValueError("Missing required API keys. Please check configs/")

# Safe authentication with error handling
try:
    kaggle_auth = config_manager.setup_kaggle_auth()
    wandb_auth = config_manager.setup_wandb_auth()
except Exception as e:
    print(f"Authentication failed: {e}")
    print("Please verify your API keys in configs/")
```

### **Colab Security**

```python
# Never embed keys in notebooks
# Instead, upload configs/ folder to Colab and load securely
def secure_colab_setup():
    """Secure setup that doesn't expose keys in notebook"""
    
    # Check if configs/ exists in Colab
    if not os.path.exists('configs/kaggle.json'):
        print("Please upload configs/ folder to Colab")
        return False
        
    # Load and validate without displaying keys
    config_manager = get_config_manager()
    return config_manager.validate_setup()
```

---

## 🎯 Implementation Timeline (Updated)

### **Day 1 (4-5 hours): Core with Existing Keys**

- ✅ Setup secure config management using existing keys
- ✅ Implement Kaggle streaming client with authentication  
- ✅ Basic streaming dataset implementation
- ✅ Simple training loop with W&B integration

### **Day 2 (3-4 hours): Advanced Features**

- ✅ Memory management and optimization
- ✅ Progressive stage training
- ✅ Cloud checkpointing system
- ✅ Colab auto-setup helper

### **Day 3 (2-3 hours): Research Integration**

- ✅ Generate research notebook with secure auth
- ✅ End-to-end validation and testing
- ✅ Model comparison and visualization

**Total: 9-12 hours** with existing key integration

This architecture leverages your existing API key setup while providing a secure, production-ready streaming system!
