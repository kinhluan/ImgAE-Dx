#!/bin/bash

# ImgAE-Dx T4 GPU Optimized Training Script for Google Colab
# Specialized for T4 GPU (16GB VRAM) with HuggingFace streaming

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

# T4-optimized defaults
MODEL_TYPE="unet"
SAMPLES=3000
EPOCHS=20
BATCH_SIZE=48  # T4-optimized with mixed precision
HF_DATASET="keremberke/chest-xray-classification"  # Reliable for Colab
HF_SPLIT="train"
HF_TOKEN=""
MIXED_PRECISION="true"
T4_OPTIMIZATIONS="true"
DRIVE_BACKUP="true"
COLAB_SETUP="false"
MEMORY_LIMIT=14  # Leave 2GB for system

# Colab directories
COLAB_CHECKPOINT_DIR="/content/drive/MyDrive/imgae_dx_checkpoints"
COLAB_CONFIG_DIR="/content/drive/MyDrive/imgae_dx_configs"
LOCAL_CHECKPOINT_DIR="./outputs/checkpoints"

show_help() {
    cat << EOF
${CYAN}ImgAE-Dx T4 GPU Training for Google Colab${NC}
=============================================

Usage: $0 MODEL_TYPE [OPTIONS]

${GREEN}MODEL_TYPE:${NC}
    unet          Train U-Net autoencoder (recommended for T4)
    reversed_ae   Train Reversed Autoencoder  
    both          Train both models (2-3 hours total)

${GREEN}T4-OPTIMIZED OPTIONS:${NC}
    --samples NUM         Samples (default: 3000, T4-optimized)
    --epochs NUM          Epochs (default: 20, T4-optimized)
    --batch-size NUM      Batch size (default: 48, T4 + AMP)
    --hf-dataset NAME     HF dataset (default: keremberke/chest-xray-classification)
    --hf-token TOKEN      HuggingFace auth token
    --colab-setup         Auto-setup Colab environment
    --drive-backup        Backup to Google Drive (default: enabled)
    --memory-limit GB     Memory limit (default: 14GB for T4)

${GREEN}T4 PERFORMANCE OPTIONS:${NC}
    --no-mixed-precision  Disable AMP (not recommended for T4)
    --no-t4-optimizations Disable T4-specific optimizations
    --aggressive-memory   Use 15GB memory limit (risky)
    --conservative        Use safe settings (batch-32, 12GB limit)

${GREEN}COLAB EXAMPLES:${NC}
    # Quick T4 training (25-35 minutes)
    $0 unet --samples 2000 --epochs 15

    # Research-quality training (45-75 minutes)  
    $0 unet --samples 5000 --epochs 25 --hf-token your_token

    # Both models training (1.5-2 hours)
    $0 both --samples 3000 --epochs 20

    # Custom dataset with full setup
    $0 unet --colab-setup --hf-dataset "alkzar90/NIH-Chest-X-ray-dataset" --samples 4000

    # Conservative mode (for unstable Colab)
    $0 unet --conservative --samples 2000

${GREEN}T4 GPU SPECIFICATIONS:${NC}
    GPU Memory: 16GB GDDR6
    Optimal Batch Size: 32-64 (with AMP)
    Memory Limit: 14GB (safe) / 15GB (aggressive)
    Training Speed: ~850 samples/sec (optimized)
EOF
}

# Parse arguments with T4-specific handling
parse_args() {
    if [ $# -eq 0 ] || [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        show_help
        exit 0
    fi

    MODEL_TYPE="$1"
    shift

    while [[ $# -gt 0 ]]; do
        case $1 in
            --samples)
                SAMPLES="$2"
                shift 2
                ;;
            --epochs)
                EPOCHS="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --hf-dataset)
                HF_DATASET="$2"
                shift 2
                ;;
            --hf-token)
                HF_TOKEN="$2"
                shift 2
                ;;
            --colab-setup)
                COLAB_SETUP="true"
                shift
                ;;
            --drive-backup)
                DRIVE_BACKUP="true"
                shift
                ;;
            --no-drive-backup)
                DRIVE_BACKUP="false"
                shift
                ;;
            --memory-limit)
                MEMORY_LIMIT="$2"
                shift 2
                ;;
            --no-mixed-precision)
                MIXED_PRECISION="false"
                BATCH_SIZE=32  # Reduce batch size without AMP
                shift
                ;;
            --no-t4-optimizations)
                T4_OPTIMIZATIONS="false"
                shift
                ;;
            --aggressive-memory)
                MEMORY_LIMIT=15
                print_warning "Aggressive memory mode: Risk of OOM"
                shift
                ;;
            --conservative)
                BATCH_SIZE=32
                MEMORY_LIMIT=12
                SAMPLES=2000
                print_info "Conservative mode enabled"
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

# Detect and optimize for T4 GPU
detect_t4_gpu() {
    print_info "Detecting GPU configuration..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1)
        GPU_NAME=$(echo $GPU_INFO | cut -d',' -f1 | xargs)
        GPU_MEMORY=$(echo $GPU_INFO | cut -d',' -f2 | xargs)
        
        print_status "GPU: $GPU_NAME"
        print_status "GPU Memory: ${GPU_MEMORY}MB"
        
        # T4 Detection and optimization
        if [[ "$GPU_NAME" == *"T4"* ]]; then
            print_t4 "Tesla T4 detected! Applying T4 optimizations..."
            
            # T4-specific optimizations
            if [ "$MIXED_PRECISION" = "true" ]; then
                print_t4 "AMP enabled: Batch size optimized to $BATCH_SIZE"
            fi
            
            export CUDA_LAUNCH_BLOCKING=0
            export CUDNN_BENCHMARK=1
            
            # Memory fraction for T4
            if [ "$MEMORY_LIMIT" -gt 15 ]; then
                print_warning "Memory limit too high for T4, adjusting to 15GB"
                MEMORY_LIMIT=15
            fi
            
            return 0
        else
            print_warning "Non-T4 GPU detected. T4 optimizations may not be optimal."
            return 1
        fi
    else
        print_error "No GPU detected! T4 optimizations require CUDA GPU."
        exit 1
    fi
}

# Setup Colab environment
setup_colab_environment() {
    if [ "$COLAB_SETUP" = "true" ]; then
        print_info "Setting up Colab environment..."
        
        # Check if in Colab
        if python3 -c "import google.colab" 2>/dev/null; then
            print_status "Google Colab detected"
            
            # Mount Google Drive
            python3 -c "
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
print('✅ Google Drive mounted')
"
            
            # Create directories
            mkdir -p "$COLAB_CHECKPOINT_DIR"
            mkdir -p "$COLAB_CONFIG_DIR"
            mkdir -p "$LOCAL_CHECKPOINT_DIR"
            
            print_status "Colab directories created"
        else
            print_warning "Not in Colab environment, skipping Colab setup"
            COLAB_SETUP="false"
        fi
    fi
}

# T4-optimized training execution
train_t4_optimized() {
    local model_type=$1
    local model_name=$(echo "$model_type" | tr '_' '-')
    
    print_t4 "Starting T4-optimized training for $model_name"
    
    # Build T4-optimized command
    local cmd="python -m imgae_dx.cli.train"
    cmd="$cmd --model $model_type"
    cmd="$cmd --samples $SAMPLES"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --memory-limit $MEMORY_LIMIT"
    cmd="$cmd --data-source huggingface"
    cmd="$cmd --hf-dataset $HF_DATASET"
    cmd="$cmd --hf-streaming"
    
    # T4-specific optimizations
    if [ "$MIXED_PRECISION" = "true" ]; then
        cmd="$cmd --mixed-precision"
    fi
    
    if [ "$T4_OPTIMIZATIONS" = "true" ]; then
        cmd="$cmd --optimize-for-t4"
    fi
    
    # HuggingFace token
    if [ -n "$HF_TOKEN" ]; then
        cmd="$cmd --hf-token $HF_TOKEN"
    fi
    
    # Output directory
    cmd="$cmd --output-dir $LOCAL_CHECKPOINT_DIR"
    
    # W&B project
    export WANDB_PROJECT="imgae-dx-t4-colab"
    
    # Log file
    local log_file="./outputs/logs/t4_${model_name}_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p ./outputs/logs
    
    print_info "T4 Command: $cmd"
    print_info "Log file: $log_file"
    
    # Set T4-specific environment variables
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024"
    export OMP_NUM_THREADS=4
    
    # Execute training with T4 optimizations
    local start_time=$(date +%s)
    
    if eval "$cmd 2>&1 | tee $log_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        
        print_t4 "$model_name training completed in ${minutes}m!"
        
        # Backup to Drive if enabled
        if [ "$DRIVE_BACKUP" = "true" ] && [ "$COLAB_SETUP" = "true" ]; then
            backup_to_drive "$model_name"
        fi
        
        return 0
    else
        print_error "$model_name training failed!"
        return 1
    fi
}

# Backup to Google Drive
backup_to_drive() {
    local model_name=$1
    
    print_info "Backing up to Google Drive..."
    
    # Find latest checkpoints
    local checkpoints=($(ls ${LOCAL_CHECKPOINT_DIR}/${model_name}*.pth 2>/dev/null || true))
    
    for checkpoint in "${checkpoints[@]}"; do
        local filename=$(basename "$checkpoint")
        local drive_path="${COLAB_CHECKPOINT_DIR}/${filename}"
        
        cp "$checkpoint" "$drive_path"
        print_status "Backed up: $filename"
    done
    
    # Backup logs
    local latest_log=$(ls -t ./outputs/logs/t4_${model_name}*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        local log_filename=$(basename "$latest_log")
        cp "$latest_log" "${COLAB_CONFIG_DIR}/${log_filename}"
        print_status "Log backed up: $log_filename"
    fi
}

# Show T4 performance summary
show_t4_summary() {
    local model_type=$1
    local duration=$2
    
    echo ""
    echo "${CYAN}🎯 T4 GPU Performance Summary${NC}"
    echo "================================"
    echo "Model: $model_type"
    echo "Duration: ${duration}m"
    echo "Samples: $SAMPLES"
    echo "Batch Size: $BATCH_SIZE"
    echo "Mixed Precision: $MIXED_PRECISION"
    echo "Memory Limit: ${MEMORY_LIMIT}GB"
    echo "Dataset: $HF_DATASET"
    
    if [ -f "${LOCAL_CHECKPOINT_DIR}/${model_type}_best.pth" ]; then
        local file_size=$(du -h "${LOCAL_CHECKPOINT_DIR}/${model_type}_best.pth" | cut -f1)
        echo "Best Model Size: $file_size"
    fi
    
    echo ""
    echo "${GREEN}Next Steps:${NC}"
    echo "• Evaluate model: ./scripts/evaluate.sh ${LOCAL_CHECKPOINT_DIR}/${model_type}_best.pth"
    echo "• Compare models: ./scripts/compare.sh"
    echo "• View results in Colab notebook"
    
    if [ "$DRIVE_BACKUP" = "true" ]; then
        echo "• Models saved to: $COLAB_CHECKPOINT_DIR"
    fi
}

# Main execution
main() {
    echo "${CYAN}"
    echo "🚀 ImgAE-Dx T4 GPU Training for Colab"
    echo "======================================"
    echo "${NC}"
    
    # Parse arguments
    parse_args "$@"
    
    # Setup environment
    setup_colab_environment
    
    # Detect T4
    detect_t4_gpu
    
    # Show configuration
    echo ""
    print_t4 "T4 Training Configuration:"
    echo "  Model(s): $MODEL_TYPE"
    echo "  Dataset: $HF_DATASET"
    echo "  Samples: $SAMPLES"
    echo "  Epochs: $EPOCHS" 
    echo "  Batch Size: $BATCH_SIZE (T4-optimized)"
    echo "  Memory Limit: ${MEMORY_LIMIT}GB"
    echo "  Mixed Precision: $MIXED_PRECISION"
    echo "  Drive Backup: $DRIVE_BACKUP"
    echo ""
    
    # Training execution
    local overall_start=$(date +%s)
    local success=true
    
    case "$MODEL_TYPE" in
        unet)
            train_t4_optimized "unet" || success=false
            ;;
        reversed_ae)
            train_t4_optimized "reversed_ae" || success=false
            ;;
        both)
            print_t4 "Training both models on T4..."
            train_t4_optimized "unet" || success=false
            if [ "$success" = true ]; then
                sleep 10  # Let GPU cool down
                train_t4_optimized "reversed_ae" || success=false
            fi
            ;;
        *)
            print_error "Invalid model type: $MODEL_TYPE"
            exit 1
            ;;
    esac
    
    # Final summary
    local overall_end=$(date +%s)
    local total_duration=$(((overall_end - overall_start) / 60))
    
    if [ "$success" = true ]; then
        show_t4_summary "$MODEL_TYPE" "$total_duration"
        print_status "T4 training completed successfully! 🎉"
    else
        print_error "T4 training failed!"
        exit 1
    fi
}

# Execute if called directly
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi