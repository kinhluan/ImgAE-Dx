#!/bin/bash

# ImgAE-Dx Jupyter Lab Launcher
# Start Jupyter Lab with project-specific configuration

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[✓]${NC} $1"; }
print_info() { echo -e "${BLUE}[ℹ]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[⚠]${NC} $1"; }

# Default values
PORT=8888
NOTEBOOK_DIR="notebooks"
OPEN_BROWSER=true

# Help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Launch Jupyter Lab for ImgAE-Dx development.

OPTIONS:
    --port NUM          Port number (default: 8888)
    --dir PATH          Working directory (default: notebooks)
    --no-browser        Don't open browser automatically
    --lab               Force Jupyter Lab (default)
    --notebook          Use classic Jupyter Notebook
    --help, -h          Show this help message

EXAMPLES:
    # Standard launch
    $0

    # Custom port
    $0 --port 9999

    # Work from project root
    $0 --dir .

    # Classic notebook interface
    $0 --notebook
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --dir)
            NOTEBOOK_DIR="$2"
            shift 2
            ;;
        --no-browser)
            OPEN_BROWSER=false
            shift
            ;;
        --lab)
            JUPYTER_TYPE="lab"
            shift
            ;;
        --notebook)
            JUPYTER_TYPE="notebook"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set default if not specified
JUPYTER_TYPE=${JUPYTER_TYPE:-lab}

# Validate notebook directory
if [ ! -d "$NOTEBOOK_DIR" ]; then
    print_warning "Directory $NOTEBOOK_DIR doesn't exist. Creating it..."
    mkdir -p "$NOTEBOOK_DIR"
fi

# Check if Poetry environment is available
check_environment() {
    print_info "Checking Poetry environment..."
    
    if ! command -v poetry &> /dev/null; then
        print_warning "Poetry not found. Make sure it's installed and in PATH."
        exit 1
    fi
    
    # Check if virtual environment exists
    if poetry env info --path &> /dev/null; then
        VENV_PATH=$(poetry env info --path)
        print_status "Virtual environment found: $VENV_PATH"
    else
        print_warning "Virtual environment not found. Run 'poetry install' first."
        exit 1
    fi
}

# Setup Jupyter configuration
setup_jupyter_config() {
    print_info "Setting up Jupyter configuration..."
    
    # Create Jupyter config directory if it doesn't exist
    JUPYTER_CONFIG_DIR="$HOME/.jupyter"
    mkdir -p "$JUPYTER_CONFIG_DIR"
    
    # Create custom Jupyter config for this project
    cat > "$JUPYTER_CONFIG_DIR/jupyter_lab_config.py" << 'EOF'
# ImgAE-Dx Jupyter Lab Configuration

# Allow all IPs for remote access (be careful in production)
c.ServerApp.ip = '127.0.0.1'

# Increase max buffer size for large images
c.ServerApp.max_buffer_size = 268435456  # 256MB

# Enable extensions
c.ServerApp.jpserver_extensions = {
    'jupyterlab': True,
}

# Custom CSS for better visualization
c.ServerApp.extra_static_paths = []

# Kernel settings
c.ServerApp.kernel_manager_class = 'jupyter_server.services.kernels.kernelmanager.MappingKernelManager'
EOF

    print_status "Jupyter configuration updated"
}

# Create sample notebooks if they don't exist
create_sample_notebooks() {
    if [ ! -f "$NOTEBOOK_DIR/Quick_Start.ipynb" ]; then
        print_info "Creating sample notebooks..."
        
        # Create Quick Start notebook
        cat > "$NOTEBOOK_DIR/Quick_Start.ipynb" << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# ImgAE-Dx Quick Start\n",
    "\n",
    "Welcome to ImgAE-Dx! This notebook will help you get started with medical image anomaly detection."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import core modules\n",
    "from imgae_dx.models import UNet, ReversedAutoencoder\n",
    "from imgae_dx.utils import ConfigManager\n",
    "from imgae_dx.data import MedicalImageTransforms\n",
    "\n",
    "print(\"✅ ImgAE-Dx modules imported successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a simple U-Net model\n",
    "model = UNet(input_channels=1, input_size=128, latent_dim=512)\n",
    "print(f\"Model created: {model.count_parameters():,} parameters\")\n",
    "print(model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load configuration\n",
    "config_manager = ConfigManager()\n",
    "try:\n",
    "    config = config_manager.load_config('../configs/project_config.yaml')\n",
    "    print(\"✅ Configuration loaded successfully\")\n",
    "    print(f\"Project: {config.project_name}\")\n",
    "    print(f\"Version: {config.version}\")\n",
    "except Exception as e:\n",
    "    print(f\"⚠️ Could not load config: {e}\")\n",
    "    print(\"Using default settings\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Next Steps\n",
    "\n",
    "1. **Explore the models**: Check out `imgae_dx.models`\n",
    "2. **Data processing**: Look at `imgae_dx.data` and `imgae_dx.streaming`\n",
    "3. **Training**: Use `imgae_dx.training` modules\n",
    "4. **Visualization**: Explore `imgae_dx.visualization`\n",
    "\n",
    "Happy experimenting! 🚀"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
        
        print_status "Sample notebook created: $NOTEBOOK_DIR/Quick_Start.ipynb"
    fi
}

# Launch Jupyter
launch_jupyter() {
    print_info "Launching Jupyter $JUPYTER_TYPE..."
    print_info "Port: $PORT"
    print_info "Directory: $NOTEBOOK_DIR"
    print_info "Browser: $OPEN_BROWSER"
    
    # Build command
    local cmd="poetry run jupyter $JUPYTER_TYPE"
    cmd="$cmd --port=$PORT"
    cmd="$cmd --notebook-dir=$NOTEBOOK_DIR"
    
    if [ "$OPEN_BROWSER" = false ]; then
        cmd="$cmd --no-browser"
    fi
    
    # Additional Jupyter Lab options
    if [ "$JUPYTER_TYPE" = "lab" ]; then
        cmd="$cmd --ServerApp.terminado_settings='{\"shell_command\":[\"/bin/bash\"]}'"
    fi
    
    print_status "Starting Jupyter..."
    print_info "Command: $cmd"
    print_info "Access URL: http://localhost:$PORT"
    
    echo ""
    echo "🚀 Jupyter $JUPYTER_TYPE is starting..."
    echo "📂 Working directory: $NOTEBOOK_DIR"
    echo "🌐 URL: http://localhost:$PORT"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    # Execute the command
    eval "$cmd"
}

# Main execution
main() {
    echo "📓 ImgAE-Dx Jupyter Launcher"
    echo "============================="
    echo ""
    
    check_environment
    setup_jupyter_config
    create_sample_notebooks
    
    echo ""
    launch_jupyter
}

# Handle script termination
cleanup() {
    print_info "Shutting down Jupyter..."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Run main function
main "$@"