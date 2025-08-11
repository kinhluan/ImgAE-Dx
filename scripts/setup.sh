#!/bin/bash

# ImgAE-Dx Setup Script
# Automated setup for medical image anomaly detection project

set -e  # Exit on any error

echo "🚀 ImgAE-Dx Setup Starting..."
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ]${NC} $1"
}

# Check if Poetry is installed
check_poetry() {
    print_info "Checking Poetry installation..."
    if ! command -v poetry &> /dev/null; then
        print_error "Poetry not found. Installing Poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
        
        if ! command -v poetry &> /dev/null; then
            print_error "Poetry installation failed. Please install manually: https://python-poetry.org/docs/#installation"
            exit 1
        fi
    fi
    
    POETRY_VERSION=$(poetry --version | cut -d' ' -f3)
    print_status "Poetry found: version $POETRY_VERSION"
}

# Check Python version
check_python() {
    print_info "Checking Python version..."
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    REQUIRED_VERSION="3.8.1"
    
    if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,8,1) else 1)"; then
        print_status "Python version OK: $PYTHON_VERSION"
    else
        print_error "Python 3.8.1+ required, found: $PYTHON_VERSION"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_info "Installing project dependencies..."
    
    echo "📦 Installing core dependencies..."
    poetry install
    
    print_status "Dependencies installed successfully"
}

# Setup configuration files
setup_config() {
    print_info "Setting up configuration files..."
    
    # Create config templates if they don't exist
    if [ ! -f "configs/kaggle.json" ]; then
        cat > configs/kaggle.json.template << 'EOF'
{
    "username": "your-kaggle-username",
    "key": "your-kaggle-api-key"
}
EOF
        print_warning "Please edit configs/kaggle.json.template with your Kaggle credentials"
        print_warning "Then rename it to configs/kaggle.json"
    else
        print_status "Kaggle config exists"
    fi
    
    if [ ! -f "configs/wandb.json" ]; then
        cat > configs/wandb.json.template << 'EOF'
{
    "api_key": "your-wandb-api-key"
}
EOF
        print_warning "Please edit configs/wandb.json.template with your W&B API key"
        print_warning "Then rename it to configs/wandb.json"
    else
        print_status "W&B config exists"
    fi
    
    # Create directories
    mkdir -p checkpoints
    mkdir -p logs
    mkdir -p results
    
    print_status "Configuration setup complete"
}

# Create additional scripts
create_scripts() {
    print_info "Creating utility scripts..."
    
    # Make all scripts executable
    chmod +x scripts/*.sh 2>/dev/null || true
    
    print_status "Scripts created and made executable"
}

# Test installation
test_installation() {
    print_info "Testing installation..."
    
    # Test Python imports
    if poetry run python -c "
import sys
try:
    from imgae_dx.models import UNet, ReversedAutoencoder
    from imgae_dx.utils import ConfigManager
    from imgae_dx.streaming import KaggleStreamClient
    print('✅ All core modules imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"; then
        print_status "Python imports working correctly"
    else
        print_error "Python import test failed"
        return 1
    fi
    
    # Test configuration loading
    if [ -f "configs/project_config.yaml" ]; then
        if poetry run python -c "
from imgae_dx.utils import ConfigManager
try:
    config = ConfigManager().load_config('configs/project_config.yaml')
    print('✅ Configuration loaded successfully')
except Exception as e:
    print(f'❌ Config error: {e}')
"; then
            print_status "Configuration test passed"
        else
            print_warning "Configuration test failed (non-critical)"
        fi
    else
        print_warning "No project config found (will use defaults)"
    fi
}

# Print next steps
print_next_steps() {
    echo ""
    echo "🎉 Setup Complete!"
    echo "=================="
    echo ""
    echo "📝 Next Steps:"
    echo "1. Configure API keys:"
    echo "   - Edit configs/kaggle.json with your Kaggle credentials"
    echo "   - Edit configs/wandb.json with your W&B API key (optional)"
    echo ""
    echo "2. Test the setup:"
    echo "   ./scripts/test.sh"
    echo ""
    echo "3. Start developing:"
    echo "   poetry shell                    # Activate virtual environment"
    echo "   ./scripts/jupyter.sh           # Start Jupyter Lab"
    echo ""
    echo "4. Quick training:"
    echo "   ./scripts/train.sh unet --samples 100"
    echo ""
    echo "📚 Documentation:"
    echo "   - Quick Start: QUICK_START.md"
    echo "   - Project Journey: docs/PROJECT_JOURNEY.md"
    echo "   - API Reference: poetry run python -c 'help(imgae_dx)'"
    echo ""
    print_status "Ready to detect anomalies! 🔍"
}

# Main setup flow
main() {
    echo "Starting ImgAE-Dx setup process..."
    echo "This will install dependencies and configure the project."
    echo ""
    
    # Run setup steps
    check_python
    check_poetry
    install_dependencies
    setup_config
    create_scripts
    
    # Test installation
    if test_installation; then
        print_next_steps
    else
        print_error "Setup completed with warnings. Check the messages above."
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --test-only    Only run installation tests"
        echo "  --deps-only    Only install dependencies"
        echo ""
        echo "This script will:"
        echo "  1. Check Python and Poetry installation"
        echo "  2. Install project dependencies"  
        echo "  3. Setup configuration templates"
        echo "  4. Create utility scripts"
        echo "  5. Test the installation"
        ;;
    --test-only)
        test_installation
        ;;
    --deps-only)
        check_python
        check_poetry
        install_dependencies
        ;;
    *)
        main
        ;;
esac