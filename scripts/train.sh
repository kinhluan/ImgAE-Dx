#!/bin/bash

# ImgAE-Dx Training Script
# Train models with various configurations

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_info() { echo -e "${BLUE}[ℹ]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }

# Default values
MODEL_TYPE="unet"
SAMPLES=2000
EPOCHS=10
BATCH_SIZE=32
CONFIG_FILE="configs/project_config.yaml"
RESUME=""
OUTPUT_DIR="checkpoints"
WANDB_PROJECT="imgae-dx"

# Help message
show_help() {
    cat << EOF
Usage: $0 MODEL_TYPE [OPTIONS]

Train ImgAE-Dx models for medical image anomaly detection.

MODEL_TYPE:
    unet          Train U-Net autoencoder (baseline)
    reversed_ae   Train Reversed Autoencoder
    both          Train both models sequentially

OPTIONS:
    --samples NUM         Number of samples to use (default: 2000)
    --epochs NUM          Number of training epochs (default: 10)
    --batch-size NUM      Batch size (default: 32)
    --config FILE         Configuration file (default: configs/project_config.yaml)
    --resume PATH         Resume from checkpoint
    --output-dir DIR      Output directory for checkpoints (default: checkpoints)
    --wandb-project NAME  W&B project name (default: imgae-dx)
    --no-wandb           Disable W&B logging
    --gpu                Force GPU usage
    --cpu                Force CPU usage
    --memory-limit GB    Memory limit in GB (default: 4)
    --help, -h           Show this help message

EXAMPLES:
    # Quick training on small dataset
    $0 unet --samples 100 --epochs 5

    # Full training with GPU
    $0 reversed_ae --samples 2000 --epochs 20 --gpu

    # Resume training from checkpoint
    $0 unet --resume checkpoints/unet_epoch_10.pth

    # Train both models
    $0 both --samples 1000 --epochs 15

    # Custom config file
    $0 unet --config configs/custom.yaml
EOF
}

# Parse arguments
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
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --resume)
                RESUME="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --wandb-project)
                WANDB_PROJECT="$2"
                shift 2
                ;;
            --no-wandb)
                export WANDB_MODE=disabled
                shift
                ;;
            --gpu)
                export CUDA_VISIBLE_DEVICES=0
                shift
                ;;
            --cpu)
                export CUDA_VISIBLE_DEVICES=""
                shift
                ;;
            --memory-limit)
                MEMORY_LIMIT="$2"
                shift 2
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

# Validate inputs
validate_inputs() {
    if [ ! -f "$CONFIG_FILE" ]; then
        print_warning "Config file not found: $CONFIG_FILE"
        print_info "Using default configuration"
        CONFIG_FILE=""
    fi

    case "$MODEL_TYPE" in
        unet|reversed_ae|both)
            ;;
        *)
            print_error "Invalid model type: $MODEL_TYPE"
            print_error "Valid options: unet, reversed_ae, both"
            exit 1
            ;;
    esac

    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    mkdir -p logs
}

# Check system resources
check_resources() {
    print_info "Checking system resources..."
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | head -1)
        print_status "GPU available: $GPU_INFO"
    else
        print_warning "No GPU detected. Training will use CPU (slower)"
    fi
    
    # Check memory
    if command -v free &> /dev/null; then
        TOTAL_RAM=$(free -g | awk 'NR==2{print $2}')
        print_info "System RAM: ${TOTAL_RAM}GB"
    elif command -v vm_stat &> /dev/null; then
        # macOS
        TOTAL_RAM=$(echo "$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d .) * 4096 / 1024 / 1024 / 1024" | bc)
        print_info "System RAM: ~${TOTAL_RAM}GB"
    fi
}

# Train single model
train_model() {
    local model_type=$1
    local model_name=$(echo "$model_type" | tr '_' '-')
    
    print_info "Training $model_name model..."
    print_info "Samples: $SAMPLES, Epochs: $EPOCHS, Batch size: $BATCH_SIZE"
    
    # Build command
    local cmd="poetry run python -m imgae_dx.cli.train"
    cmd="$cmd --model $model_type"
    cmd="$cmd --samples $SAMPLES"
    cmd="$cmd --epochs $EPOCHS"
    cmd="$cmd --batch-size $BATCH_SIZE"
    cmd="$cmd --output-dir $OUTPUT_DIR"
    
    if [ -n "$CONFIG_FILE" ]; then
        cmd="$cmd --config $CONFIG_FILE"
    fi
    
    if [ -n "$RESUME" ]; then
        cmd="$cmd --resume $RESUME"
    fi
    
    if [ -n "$MEMORY_LIMIT" ]; then
        cmd="$cmd --memory-limit $MEMORY_LIMIT"
    fi
    
    # Set W&B project
    export WANDB_PROJECT="$WANDB_PROJECT"
    
    # Create log file
    local log_file="logs/${model_name}_$(date +%Y%m%d_%H%M%S).log"
    
    print_info "Command: $cmd"
    print_info "Log file: $log_file"
    
    # Execute training
    if eval "$cmd 2>&1 | tee $log_file"; then
        print_status "$model_name training completed successfully!"
        
        # Show results
        local checkpoint_pattern="$OUTPUT_DIR/${model_name}*.pth"
        local checkpoints=($(ls $checkpoint_pattern 2>/dev/null || true))
        
        if [ ${#checkpoints[@]} -gt 0 ]; then
            print_status "Checkpoints created:"
            for checkpoint in "${checkpoints[@]}"; do
                echo "  - $checkpoint"
            done
        fi
        
        return 0
    else
        print_error "$model_name training failed!"
        print_error "Check log file: $log_file"
        return 1
    fi
}

# Main training flow
main() {
    echo "🧠 ImgAE-Dx Model Training"
    echo "=========================="
    echo ""
    
    # Parse and validate
    parse_args "$@"
    validate_inputs
    check_resources
    
    echo ""
    print_info "Training configuration:"
    echo "  Model(s): $MODEL_TYPE"
    echo "  Samples: $SAMPLES"
    echo "  Epochs: $EPOCHS"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Output dir: $OUTPUT_DIR"
    echo "  Config file: ${CONFIG_FILE:-"default"}"
    echo "  W&B project: $WANDB_PROJECT"
    echo ""
    
    # Start training
    local start_time=$(date +%s)
    local success=true
    
    case "$MODEL_TYPE" in
        unet)
            train_model "unet" || success=false
            ;;
        reversed_ae)
            train_model "reversed_ae" || success=false
            ;;
        both)
            print_info "Training both models sequentially..."
            train_model "unet" || success=false
            if [ "$success" = true ]; then
                print_info "Starting Reversed AE training..."
                train_model "reversed_ae" || success=false
            fi
            ;;
    esac
    
    # Summary
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    echo ""
    echo "📊 Training Summary"
    echo "==================="
    echo "Duration: ${minutes}m ${seconds}s"
    echo "Model(s): $MODEL_TYPE"
    echo "Samples: $SAMPLES"
    echo "Epochs: $EPOCHS"
    
    if [ "$success" = true ]; then
        print_status "All training completed successfully! 🎉"
        echo ""
        echo "Next steps:"
        echo "  - Evaluate models: ./scripts/evaluate.sh"
        echo "  - Compare results: ./scripts/compare.sh"
        echo "  - View in notebook: ./scripts/jupyter.sh"
    else
        print_error "Training failed! Check the logs above."
        exit 1
    fi
}

# Handle direct execution
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi