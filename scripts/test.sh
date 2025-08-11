#!/bin/bash

# ImgAE-Dx Test Suite Runner
# Run tests and validate project setup

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

# Test configuration
RUN_UNIT_TESTS=true
RUN_INTEGRATION_TESTS=false
RUN_IMPORTS=true
RUN_CONFIG=true
RUN_MODELS=true
VERBOSE=false

# Help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run ImgAE-Dx test suite and validate project setup.

OPTIONS:
    --unit              Run unit tests only
    --integration       Run integration tests only  
    --all               Run all tests
    --imports           Test imports only
    --config            Test configuration only
    --models            Test model creation only
    --verbose, -v       Verbose output
    --help, -h          Show this help message

EXAMPLES:
    # Quick validation
    $0

    # Full test suite
    $0 --all --verbose

    # Just test imports
    $0 --imports

    # Test specific component
    $0 --models --verbose
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            RUN_UNIT_TESTS=true
            RUN_INTEGRATION_TESTS=false
            shift
            ;;
        --integration)
            RUN_UNIT_TESTS=false
            RUN_INTEGRATION_TESTS=true
            shift
            ;;
        --all)
            RUN_UNIT_TESTS=true
            RUN_INTEGRATION_TESTS=true
            shift
            ;;
        --imports)
            RUN_IMPORTS=true
            RUN_CONFIG=false
            RUN_MODELS=false
            RUN_UNIT_TESTS=false
            shift
            ;;
        --config)
            RUN_IMPORTS=false
            RUN_CONFIG=true
            RUN_MODELS=false
            RUN_UNIT_TESTS=false
            shift
            ;;
        --models)
            RUN_IMPORTS=false
            RUN_CONFIG=false
            RUN_MODELS=true
            RUN_UNIT_TESTS=false
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
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

# Test imports
test_imports() {
    print_info "Testing Python imports..."
    
    local import_script=$(cat << 'EOF'
import sys
import traceback

def test_import(module_name, description):
    try:
        exec(f"import {module_name}")
        print(f"✅ {description}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {e}")
        if "--verbose" in sys.argv or "-v" in sys.argv:
            traceback.print_exc()
        return False

success = True

# Core modules
success &= test_import("imgae_dx", "Core package")
success &= test_import("imgae_dx.models", "Models module")  
success &= test_import("imgae_dx.utils", "Utils module")
success &= test_import("imgae_dx.streaming", "Streaming module")
success &= test_import("imgae_dx.data", "Data module")

# Specific imports
try:
    from imgae_dx.models import UNet, ReversedAutoencoder, BaseAutoencoder
    print("✅ Model classes")
except ImportError as e:
    print(f"❌ Model classes: {e}")
    success = False

try:
    from imgae_dx.utils import ConfigManager
    print("✅ Configuration manager")
except ImportError as e:
    print(f"❌ Configuration manager: {e}")
    success = False

try:
    from imgae_dx.streaming import KaggleStreamClient, StreamingMemoryManager
    print("✅ Streaming components")
except ImportError as e:
    print(f"❌ Streaming components: {e}")
    success = False

try:
    from imgae_dx.data import MedicalImageTransforms
    print("✅ Data transforms")
except ImportError as e:
    print(f"❌ Data transforms: {e}")
    success = False

# Dependencies
success &= test_import("torch", "PyTorch")
success &= test_import("torchvision", "TorchVision")
success &= test_import("numpy", "NumPy")
success &= test_import("pandas", "Pandas")
success &= test_import("sklearn", "Scikit-learn")
success &= test_import("PIL", "Pillow")
success &= test_import("yaml", "PyYAML")
success &= test_import("tqdm", "TQDM")

if success:
    print("\n🎉 All imports successful!")
    sys.exit(0)
else:
    print("\n💥 Some imports failed!")
    sys.exit(1)
EOF
)
    
    if [ "$VERBOSE" = true ]; then
        poetry run python -c "$import_script" --verbose
    else
        poetry run python -c "$import_script"
    fi
    
    return $?
}

# Test configuration
test_config() {
    print_info "Testing configuration system..."
    
    local config_script=$(cat << 'EOF'
import sys
from pathlib import Path
from imgae_dx.utils import ConfigManager

try:
    # Test config manager initialization
    config_manager = ConfigManager()
    print("✅ ConfigManager initialization")
    
    # Test device detection
    device = config_manager.get_device()
    print(f"✅ Device detection: {device}")
    
    # Test environment detection
    is_colab = config_manager.is_colab_environment()
    print(f"✅ Environment detection: {'Colab' if is_colab else 'Local'}")
    
    # Test config loading (if config exists)
    config_file = Path("configs/project_config.yaml")
    if config_file.exists():
        try:
            config = config_manager.load_config(config_file)
            print(f"✅ Config loading: {config.project_name} v{config.version}")
        except Exception as e:
            print(f"⚠️  Config loading warning: {e}")
    else:
        print("ℹ️  No project config found (using defaults)")
    
    # Test API key structure (without validation)
    api_keys = config_manager.api_keys
    print(f"✅ API keys structure initialized")
    
    print("\n🎉 Configuration tests passed!")
    
except Exception as e:
    print(f"❌ Configuration test failed: {e}")
    if "--verbose" in sys.argv or "-v" in sys.argv:
        import traceback
        traceback.print_exc()
    sys.exit(1)
EOF
)

    if [ "$VERBOSE" = true ]; then
        poetry run python -c "$config_script" --verbose
    else
        poetry run python -c "$config_script"
    fi
    
    return $?
}

# Test model creation
test_models() {
    print_info "Testing model creation..."
    
    local model_script=$(cat << 'EOF'
import sys
import torch
from imgae_dx.models import UNet, ReversedAutoencoder

try:
    # Test UNet creation
    unet = UNet(input_channels=1, input_size=128, latent_dim=512)
    print(f"✅ U-Net creation: {unet.count_parameters():,} parameters")
    
    # Test UNet forward pass
    x = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        output = unet(x)
    print(f"✅ U-Net forward pass: {x.shape} -> {output.shape}")
    
    # Test Reversed AE creation
    rae = ReversedAutoencoder(input_channels=1, input_size=128, latent_dim=512)
    print(f"✅ Reversed AE creation: {rae.count_parameters():,} parameters")
    
    # Test Reversed AE forward pass
    with torch.no_grad():
        output = rae(x)
    print(f"✅ Reversed AE forward pass: {x.shape} -> {output.shape}")
    
    # Test encode/decode
    with torch.no_grad():
        latent = unet.encode(x)
        reconstructed = unet.decode(latent)
    print(f"✅ Encode/decode: {x.shape} -> {latent.shape} -> {reconstructed.shape}")
    
    # Test anomaly scoring
    with torch.no_grad():
        scores = unet.compute_anomaly_score(x)
    print(f"✅ Anomaly scoring: {scores.shape}")
    
    # Test model info
    info = unet.get_model_info()
    print(f"✅ Model info: {info['model_name']}")
    
    print("\n🎉 Model tests passed!")
    
except Exception as e:
    print(f"❌ Model test failed: {e}")
    if "--verbose" in sys.argv or "-v" in sys.argv:
        import traceback
        traceback.print_exc()
    sys.exit(1)
EOF
)

    if [ "$VERBOSE" = true ]; then
        poetry run python -c "$model_script" --verbose
    else
        poetry run python -c "$model_script"
    fi
    
    return $?
}

# Run unit tests
run_unit_tests() {
    print_info "Running unit tests..."
    
    if [ -d "tests" ]; then
        if [ "$VERBOSE" = true ]; then
            poetry run pytest tests/unit/ -v
        else
            poetry run pytest tests/unit/ -q
        fi
    else
        print_warning "No tests directory found. Creating basic test structure..."
        mkdir -p tests/unit
        
        # Create a simple test file
        cat > tests/unit/test_models.py << 'EOF'
"""Basic model tests."""

import torch
import pytest
from imgae_dx.models import UNet, ReversedAutoencoder


def test_unet_creation():
    """Test UNet model creation."""
    model = UNet(input_channels=1, input_size=64, latent_dim=256)
    assert model is not None
    assert model.count_parameters() > 0


def test_unet_forward():
    """Test UNet forward pass."""
    model = UNet(input_channels=1, input_size=64, latent_dim=256)
    x = torch.randn(2, 1, 64, 64)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == x.shape


def test_reversed_ae_creation():
    """Test Reversed Autoencoder creation."""
    model = ReversedAutoencoder(input_channels=1, input_size=64, latent_dim=256)
    assert model is not None
    assert model.count_parameters() > 0


def test_reversed_ae_forward():
    """Test Reversed AE forward pass."""
    model = ReversedAutoencoder(input_channels=1, input_size=64, latent_dim=256)
    x = torch.randn(2, 1, 64, 64)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == x.shape
EOF
        
        # Run the new tests
        if [ "$VERBOSE" = true ]; then
            poetry run pytest tests/unit/test_models.py -v
        else
            poetry run pytest tests/unit/test_models.py -q
        fi
    fi
    
    return $?
}

# Run integration tests
run_integration_tests() {
    print_info "Running integration tests..."
    
    if [ -d "tests/integration" ]; then
        if [ "$VERBOSE" = true ]; then
            poetry run pytest tests/integration/ -v
        else
            poetry run pytest tests/integration/ -q
        fi
    else
        print_warning "No integration tests found. Skipping..."
        return 0
    fi
    
    return $?
}

# Main test execution
main() {
    echo "🧪 ImgAE-Dx Test Suite"
    echo "======================"
    echo ""
    
    local overall_success=true
    local tests_run=0
    local tests_passed=0
    
    # Run selected tests
    if [ "$RUN_IMPORTS" = true ]; then
        echo "📦 Import Tests"
        echo "---------------"
        if test_imports; then
            tests_passed=$((tests_passed + 1))
        else
            overall_success=false
        fi
        tests_run=$((tests_run + 1))
        echo ""
    fi
    
    if [ "$RUN_CONFIG" = true ]; then
        echo "⚙️  Configuration Tests"
        echo "----------------------"
        if test_config; then
            tests_passed=$((tests_passed + 1))
        else
            overall_success=false
        fi
        tests_run=$((tests_run + 1))
        echo ""
    fi
    
    if [ "$RUN_MODELS" = true ]; then
        echo "🧠 Model Tests"
        echo "-------------"
        if test_models; then
            tests_passed=$((tests_passed + 1))
        else
            overall_success=false
        fi
        tests_run=$((tests_run + 1))
        echo ""
    fi
    
    if [ "$RUN_UNIT_TESTS" = true ]; then
        echo "🔬 Unit Tests"
        echo "------------"
        if run_unit_tests; then
            tests_passed=$((tests_passed + 1))
        else
            overall_success=false
        fi
        tests_run=$((tests_run + 1))
        echo ""
    fi
    
    if [ "$RUN_INTEGRATION_TESTS" = true ]; then
        echo "🔗 Integration Tests"
        echo "------------------"
        if run_integration_tests; then
            tests_passed=$((tests_passed + 1))
        else
            overall_success=false
        fi
        tests_run=$((tests_run + 1))
        echo ""
    fi
    
    # Summary
    echo "📊 Test Summary"
    echo "==============="
    echo "Tests run: $tests_run"
    echo "Tests passed: $tests_passed"
    echo "Tests failed: $((tests_run - tests_passed))"
    echo ""
    
    if [ "$overall_success" = true ]; then
        print_status "All tests passed! 🎉"
        echo ""
        echo "✨ Your ImgAE-Dx setup is working correctly!"
        echo ""
        echo "Next steps:"
        echo "  - Train a model: ./scripts/train.sh unet --samples 100"
        echo "  - Start Jupyter: ./scripts/jupyter.sh"
        echo "  - Read docs: QUICK_START.md"
        exit 0
    else
        print_error "Some tests failed! 💥"
        echo ""
        echo "🔧 Troubleshooting:"
        echo "  - Check Poetry installation: poetry --version"
        echo "  - Reinstall dependencies: poetry install --no-cache"
        echo "  - Check Python version: python --version"
        echo "  - Run setup again: ./scripts/setup.sh"
        exit 1
    fi
}

# Run main function
main "$@"