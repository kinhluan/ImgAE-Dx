#!/bin/bash

# ImgAE-Dx Google Colab T4 Setup Script
# One-command setup for T4 GPU training environment

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_info() { echo -e "${BLUE}[ℹ]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_t4() { echo -e "${CYAN}[T4]${NC} $1"; }

# Default values
HF_TOKEN=""
SKIP_DRIVE_MOUNT="false"
INSTALL_DEPS="true"
SETUP_WANDB="true"
WANDB_KEY=""

show_help() {
    cat << EOF
${CYAN}ImgAE-Dx T4 Colab Setup${NC}
========================

Usage: $0 [OPTIONS]

${GREEN}OPTIONS:${NC}
    --hf-token TOKEN      HuggingFace authentication token
    --wandb-key KEY       Weights & Biases API key  
    --skip-drive-mount    Skip Google Drive mounting
    --skip-deps          Skip dependency installation
    --no-wandb           Skip W&B setup
    --help, -h           Show this help

${GREEN}EXAMPLES:${NC}
    # Complete setup with tokens
    $0 --hf-token "hf_your_token" --wandb-key "your_wandb_key"
    
    # Quick setup without tokens
    $0
    
    # Setup without Google Drive
    $0 --skip-drive-mount

${GREEN}WHAT THIS SCRIPT DOES:${NC}
    1. Detects T4 GPU and optimizes environment
    2. Mounts Google Drive for persistent storage
    3. Installs optimized dependencies for T4
    4. Sets up HuggingFace and W&B authentication
    5. Configures T4-specific environment variables
    6. Creates directory structure
    7. Downloads and installs ImgAE-Dx package
EOF
}

# Parse arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --hf-token)
                HF_TOKEN="$2"
                shift 2
                ;;
            --wandb-key)
                WANDB_KEY="$2"
                shift 2
                ;;
            --skip-drive-mount)
                SKIP_DRIVE_MOUNT="true"
                shift
                ;;
            --skip-deps)
                INSTALL_DEPS="false"
                shift
                ;;
            --no-wandb)
                SETUP_WANDB="false"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Check if running in Colab
check_colab_environment() {
    print_info "Checking Colab environment..."
    
    if python3 -c "import google.colab" 2>/dev/null; then
        print_status "Google Colab environment detected"
        return 0
    else
        print_error "This script is designed for Google Colab"
        exit 1
    fi
}

# Detect T4 GPU
detect_and_optimize_t4() {
    print_info "Detecting GPU configuration..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits | head -1)
        GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1 | xargs)
        GPU_MEMORY=$(echo $GPU_INFO | cut -d',' -f2 | xargs)
        DRIVER_VERSION=$(echo $GPU_INFO | cut -d',' -f3 | xargs)
        
        print_status "GPU: $GPU_NAME"
        print_status "VRAM: ${GPU_MEMORY}MB"
        print_status "Driver: $DRIVER_VERSION"
        
        # T4-specific optimizations
        if [[ "$GPU_NAME" == *"T4"* ]]; then
            print_t4 "Tesla T4 detected! Setting up T4 optimizations..."
            
            # T4-optimized environment variables
            export CUDA_LAUNCH_BLOCKING=0
            export CUDNN_BENCHMARK=1
            export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024"
            export OMP_NUM_THREADS=4
            
            # Create T4 config
            mkdir -p /content/configs
            cat > /content/configs/t4_optimized.yaml << EOF
# T4 GPU Optimized Configuration
device: cuda
gpu_optimization:
  enable_amp: true
  memory_fraction: 0.85
  max_split_size_mb: 1024
  
training:
  batch_size: 48
  memory_limit_gb: 14
  prefetch_factor: 3
  num_workers: 2
  pin_memory: true
  
model:
  compile: true  # PyTorch 2.0 optimization
  mixed_precision: true
EOF
            
            print_t4 "T4 configuration saved"
            return 0
        else
            print_warning "Non-T4 GPU detected: $GPU_NAME"
            print_warning "T4 optimizations may not be optimal"
            return 1
        fi
    else
        print_error "No CUDA GPU detected!"
        exit 1
    fi
}

# Mount Google Drive
setup_google_drive() {
    if [ "$SKIP_DRIVE_MOUNT" = "true" ]; then
        print_info "Skipping Google Drive mount"
        return 0
    fi
    
    print_info "Mounting Google Drive..."
    
    python3 << EOF
from google.colab import drive
import os

try:
    drive.mount('/content/drive', force_remount=True)
    
    # Create necessary directories
    directories = [
        '/content/drive/MyDrive/imgae_dx_checkpoints',
        '/content/drive/MyDrive/imgae_dx_configs',
        '/content/drive/MyDrive/imgae_dx_logs',
        '/content/drive/MyDrive/imgae_dx_results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    print("✅ Google Drive setup complete")
    
except Exception as e:
    print(f"❌ Drive mount failed: {e}")
    raise
EOF
    
    print_status "Google Drive mounted and directories created"
}

# Install optimized dependencies
install_dependencies() {
    if [ "$INSTALL_DEPS" = "false" ]; then
        print_info "Skipping dependency installation"
        return 0
    fi
    
    print_info "Installing T4-optimized dependencies..."
    
    # Update pip and install optimized PyTorch
    python3 -m pip install --upgrade pip
    
    # Install PyTorch with CUDA 11.8 (optimized for T4)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install other ML dependencies
    pip install -q \
        transformers \
        datasets \
        accelerate \
        wandb \
        pillow \
        pandas \
        numpy \
        matplotlib \
        seaborn \
        tqdm \
        scikit-learn \
        psutil
    
    print_status "Dependencies installed"
}

# Setup authentication
setup_authentication() {
    print_info "Setting up authentication..."
    
    # HuggingFace setup
    if [ -n "$HF_TOKEN" ]; then
        echo "$HF_TOKEN" > /content/.hf_token
        export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
        
        python3 -c "
from huggingface_hub import login
with open('/content/.hf_token', 'r') as f:
    token = f.read().strip()
login(token=token)
print('✅ HuggingFace authentication setup')
"
        print_status "HuggingFace token configured"
    else
        print_warning "No HuggingFace token provided"
        print_info "Some datasets may not be accessible"
    fi
    
    # W&B setup
    if [ "$SETUP_WANDB" = "true" ]; then
        if [ -n "$WANDB_KEY" ]; then
            python3 -c "
import wandb
wandb.login(key='$WANDB_KEY')
print('✅ W&B authentication setup')
"
            print_status "W&B authentication configured"
        else
            print_warning "No W&B key provided"
            print_info "Run 'wandb login' manually if needed"
        fi
    fi
}

# Install ImgAE-Dx package
install_imgae_dx() {
    print_info "Installing ImgAE-Dx package..."
    
    # Clone repository
    cd /content
    if [ -d "ImgAE-Dx" ]; then
        rm -rf ImgAE-Dx
    fi
    
    git clone https://github.com/luanbhk/imgae-dx.git ImgAE-Dx
    cd ImgAE-Dx
    
    # Install in development mode
    pip install -e .
    
    # Make scripts executable
    chmod +x scripts/*.sh
    
    print_status "ImgAE-Dx package installed"
}

# Create configuration files
create_colab_configs() {
    print_info "Creating Colab-specific configurations..."
    
    mkdir -p /content/ImgAE-Dx/configs
    
    # T4 training config
    cat > /content/ImgAE-Dx/configs/colab_t4.yaml << EOF
# Google Colab T4 Configuration
project_name: "imgae-dx-colab"

# Device settings
device: "cuda"
mixed_precision: true
compile_model: true

# T4-optimized training
training:
  batch_size: 48
  learning_rate: 1e-4
  epochs: 20
  memory_limit_gb: 14
  gradient_clip_val: 1.0
  
# Data settings
data:
  source: "huggingface"
  streaming: true
  cache_size_mb: 512
  prefetch_factor: 3
  num_workers: 2
  pin_memory: true
  
# Model settings
model:
  image_size: 128
  channels: 1
  
# Checkpointing
checkpointing:
  save_frequency: 2  # Save every 2 epochs
  drive_backup: true
  local_dir: "./outputs/checkpoints"
  drive_dir: "/content/drive/MyDrive/imgae_dx_checkpoints"
  
# W&B settings  
wandb:
  project: "imgae-dx-t4-colab"
  save_artifacts: true
  log_frequency: 10
EOF
    
    # Quick start script
    cat > /content/ImgAE-Dx/quick_start_t4.sh << 'EOF'
#!/bin/bash

echo "🚀 ImgAE-Dx T4 Quick Start"
echo "=========================="

# Quick training commands
echo ""
echo "Quick Commands:"
echo "• Fast training (20 min):    ./scripts/train_colab_t4.sh unet --samples 1500 --epochs 10"
echo "• Standard training (45 min): ./scripts/train_colab_t4.sh unet --samples 3000 --epochs 20"  
echo "• Research training (90 min): ./scripts/train_colab_t4.sh unet --samples 5000 --epochs 30"
echo "• Both models (2 hours):     ./scripts/train_colab_t4.sh both --samples 3000"
echo ""

# Show GPU status
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    echo ""
fi

echo "Ready to train! 🎯"
EOF
    
    chmod +x /content/ImgAE-Dx/quick_start_t4.sh
    
    print_status "Configuration files created"
}

# Validate installation
validate_installation() {
    print_info "Validating T4 setup..."
    
    cd /content/ImgAE-Dx
    
    # Test imports
    python3 -c "
import torch
import torchvision
from transformers import __version__ as hf_version
from datasets import __version__ as ds_version
import wandb

print('✅ PyTorch:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
    print('✅ CUDA version:', torch.version.cuda)
print('✅ HuggingFace Transformers:', hf_version)
print('✅ Datasets:', ds_version)
print('✅ W&B available')

# Test T4 optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print('✅ cuDNN benchmark enabled')
    
    # Test mixed precision
    scaler = torch.cuda.amp.GradScaler()
    print('✅ Mixed precision ready')
"
    
    # Test package import
    python3 -c "
from imgae_dx.models import UNet, ReversedAutoencoder
from imgae_dx.training import Trainer
from imgae_dx.streaming import HuggingFaceStreamClient
print('✅ ImgAE-Dx package imports successful')
"
    
    print_status "Installation validation complete"
}

# Show setup summary
show_setup_summary() {
    echo ""
    echo "${CYAN}🎯 T4 Setup Summary${NC}"
    echo "==================="
    echo "Environment: Google Colab with T4 GPU"
    echo "Package: ImgAE-Dx installed at /content/ImgAE-Dx"
    echo "Configs: T4-optimized configurations created"
    
    if [ "$SKIP_DRIVE_MOUNT" = "false" ]; then
        echo "Storage: Google Drive mounted for persistence"
    fi
    
    if [ -n "$HF_TOKEN" ]; then
        echo "HuggingFace: Authenticated ✅"
    else
        echo "HuggingFace: No token provided ⚠️"
    fi
    
    if [ "$SETUP_WANDB" = "true" ] && [ -n "$WANDB_KEY" ]; then
        echo "W&B: Authenticated ✅"  
    else
        echo "W&B: Manual setup needed ⚠️"
    fi
    
    echo ""
    echo "${GREEN}🚀 Ready to Train!${NC}"
    echo ""
    echo "Quick Start:"
    echo "  cd /content/ImgAE-Dx"
    echo "  ./quick_start_t4.sh"
    echo ""
    echo "Training Commands:"
    echo "  ./scripts/train_colab_t4.sh unet --samples 2000"
    echo "  ./scripts/train_colab_t4.sh both --samples 3000 --epochs 25"
    echo ""
    echo "Documentation:"
    echo "  ./scripts/train_colab_t4.sh --help"
    echo ""
}

# Main execution
main() {
    echo "${CYAN}"
    echo "🚀 ImgAE-Dx T4 GPU Setup for Google Colab"
    echo "=========================================="  
    echo "${NC}"
    
    # Parse arguments
    parse_args "$@"
    
    # Setup steps
    check_colab_environment
    detect_and_optimize_t4
    setup_google_drive
    install_dependencies
    setup_authentication
    install_imgae_dx
    create_colab_configs
    validate_installation
    
    # Show summary
    show_setup_summary
    
    print_status "T4 GPU setup completed successfully! 🎉"
}

# Execute if called directly
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi