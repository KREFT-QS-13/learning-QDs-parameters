#!/bin/bash
# Setup script for learning_parameters repository

set -e  # Exit on error

echo "========================================="
echo "Setting up learning_parameters repository"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CPU version by default, user can modify for CUDA)
echo ""
echo "Installing PyTorch (CPU version)..."
echo "Note: For CUDA support, install PyTorch separately:"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
pip install torch torchvision torchaudio

# Install other requirements
echo ""
echo "Installing other requirements..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import torch; import numpy; import h5py; import qdarts; print('✓ All core packages imported successfully!')" || {
    echo "✗ Installation verification failed!"
    exit 1
}

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""

