#!/bin/bash
# Setup script for learning_parameters repository

# Don't exit on error - we want to track failures and report them at the end
set +e

echo "========================================="
echo "Setting up learning_parameters repository"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
# Try python3 first, then python (Windows often uses just 'python')
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

python_version=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Warn about Python 3.14 compatibility issues
python_major=$(echo "$python_version" | cut -d. -f1)
python_minor=$(echo "$python_version" | cut -d. -f2)
if [ "$python_major" -eq 3 ] && [ "$python_minor" -ge 14 ]; then
    echo ""
    echo "⚠️  WARNING: Python 3.14+ may have compatibility issues with some packages."
    echo "   Some packages (cvxpy, osqp) may not have pre-built wheels and will require"
    echo "   compilation. Consider using Python 3.11 or 3.12 for better compatibility."
    echo ""
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
# Detect venv structure (Windows uses Scripts, Unix uses bin)
if [ -d "venv/Scripts" ]; then
    # Windows venv structure
    export PATH="$(pwd)/venv/Scripts:$PATH"
    # Try python.exe first, then python (for Git Bash compatibility)
    if [ -f "venv/Scripts/python.exe" ]; then
        VENV_PYTHON="venv/Scripts/python.exe"
        VENV_PIP="venv/Scripts/pip.exe"
    else
        VENV_PYTHON="venv/Scripts/python"
        VENV_PIP="venv/Scripts/pip"
    fi
    echo "Using Windows venv structure"
elif [ -d "venv/bin" ]; then
    # Unix venv structure
    source venv/bin/activate 2>/dev/null || export PATH="$(pwd)/venv/bin:$PATH"
    VENV_PYTHON="venv/bin/python"
    VENV_PIP="venv/bin/pip"
    echo "Using Unix venv structure"
else
    echo "Warning: Could not detect venv structure"
    VENV_PYTHON="$PYTHON_CMD"
    VENV_PIP="pip"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
if ! $VENV_PIP install --upgrade pip 2>&1; then
    echo "⚠️  Warning: Failed to upgrade pip, continuing anyway..."
fi

# Install PyTorch (CPU version by default, user can modify for CUDA)
echo ""
echo "Installing PyTorch (CPU version)..."
echo "Note: For CUDA support, install PyTorch separately:"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
if ! $VENV_PIP install torch torchvision torchaudio 2>&1; then
    echo "❌ Failed to install PyTorch packages"
    FAILED_PACKAGES+=("torch" "torchvision" "torchaudio")
fi

# Initialize failed packages tracking
FAILED_PACKAGES=()

# Install other requirements
echo ""
echo "Installing other requirements..."
echo "Note: Some packages (cvxpy, osqp) may require C++ build tools."
echo "If installation fails on Windows, install Microsoft C++ Build Tools from:"
echo "  https://visualstudio.microsoft.com/visual-cpp-build-tools/"

# Try to install requirements
echo ""
echo "Attempting to install all packages from requirements.txt..."
if ! $VENV_PIP install -r requirements.txt 2>&1; then
    echo ""
    echo "⚠️  Standard installation failed. Attempting to install packages individually..."
    echo ""
    
    # Read requirements.txt and try installing each package individually
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
        
        # Extract package name (handle version specifiers like ==, >=, <=, !=, >, <)
        # Trim whitespace and remove everything from first version specifier onwards
        package_name=$(echo "$line" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/[<>=!].*$//')
        
        if [ -z "$package_name" ]; then
            continue
        fi
        
        echo "Installing $package_name..."
        if ! $VENV_PIP install "$line" 2>&1; then
            echo "❌ Failed to install: $package_name"
            FAILED_PACKAGES+=("$package_name")
        else
            echo "✓ Successfully installed: $package_name"
        fi
    done < requirements.txt
else
    echo "✓ All packages installed successfully from requirements.txt"
fi

# Verify installation
echo ""
echo "Verifying installation..."
VERIFICATION_FAILED=0
VERIFICATION_ERRORS=()

# Check core packages
for pkg in torch numpy h5py; do
    if ! $VENV_PYTHON -c "import $pkg" 2>/dev/null; then
        VERIFICATION_FAILED=1
        VERIFICATION_ERRORS+=("$pkg")
    fi
done

# Check qdarts (may fail if cvxpy is missing, but that's okay)
if ! $VENV_PYTHON -c "import qdarts" 2>/dev/null; then
    echo "⚠️  Warning: qdarts import failed (may be due to missing cvxpy/osqp)"
    VERIFICATION_ERRORS+=("qdarts (may require cvxpy)")
fi

# Check optional packages that might have failed
for pkg in cvxpy osqp; do
    if ! $VENV_PYTHON -c "import $pkg" 2>/dev/null; then
        if [[ ! " ${FAILED_PACKAGES[@]} " =~ " ${pkg} " ]]; then
            # Package was supposed to be installed but import failed
            FAILED_PACKAGES+=("$pkg")
        fi
    fi
done

if [ $VERIFICATION_FAILED -eq 0 ] && [ ${#VERIFICATION_ERRORS[@]} -eq 0 ]; then
    echo "✓ Core packages verified successfully!"
elif [ ${#VERIFICATION_ERRORS[@]} -gt 0 ]; then
    echo "⚠️  Verification warnings:"
    for err in "${VERIFICATION_ERRORS[@]}"; do
        echo "   - $err"
    done
fi

echo ""
echo "========================================="
if [ ${#FAILED_PACKAGES[@]} -eq 0 ]; then
    echo "Setup complete! ✓"
else
    echo "Setup completed with warnings ⚠️"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Report failed packages
if [ ${#FAILED_PACKAGES[@]} -gt 0 ]; then
    echo "❌ The following packages failed to install:"
    echo ""
    for pkg in "${FAILED_PACKAGES[@]}"; do
        echo "   - $pkg"
    done    
    echo "To manually install failed packages, activate the venv and run:"
    if [ -d "venv/Scripts" ]; then
        echo "   venv\\Scripts\\activate"
    else
        echo "   source venv/bin/activate"
    fi
    echo "   pip install " "${FAILED_PACKAGES[@]}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi  

echo "To activate the virtual environment in the future, run:"
if [ -d "venv/Scripts" ]; then
    echo "  source venv/Scripts/activate  (Git Bash/MSYS)"
    echo "  or: venv\\Scripts\\activate.bat  (Command Prompt)"
    echo "  or: venv\\Scripts\\Activate.ps1  (PowerShell)"
else
    echo "  source venv/bin/activate"
fi
echo ""

