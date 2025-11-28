#!/bin/bash
# Setup script for GNN Sparsification Research project

echo "=========================================="
echo "GNN Sparsification Research - Setup"
echo "=========================================="

# Check Python version
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_CMD="python3"
else
    python_version=$(python --version 2>&1 | awk '{print $2}')
    PYTHON_CMD="python"
fi
echo "✓ Python version: $python_version"

# Check if we're already in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✓ Already in virtual environment: $VIRTUAL_ENV"
    PIP_CMD="pip"
    PYTHON_CMD="python"
else
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        $PYTHON_CMD -m venv .venv
        echo "✓ Virtual environment created"
    else
        echo "✓ Virtual environment already exists"
    fi
    
    # Activate virtual environment
    echo "Activating virtual environment..."
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    else
        source venv/bin/activate
    fi
    PIP_CMD="pip"
    PYTHON_CMD="python"
fi

# Upgrade pip
echo "Upgrading pip..."
$PIP_CMD install --upgrade pip setuptools wheel > /dev/null 2>&1 || {
    echo "  Upgrading with --quiet flag..."
    $PIP_CMD install --quiet --upgrade pip setuptools wheel
}

# Install PyTorch first (with CUDA support if available)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "  CUDA detected, installing PyTorch with CUDA support"
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "  No CUDA detected, installing CPU-only PyTorch"
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Get PyTorch version for PyG
TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "2.0.0")
echo "✓ PyTorch ${TORCH_VERSION} installed"

# Install PyG dependencies with correct torch version
echo "Installing PyTorch Geometric and extensions..."
$PIP_CMD install torch-geometric

# Try to install PyG extensions, but don't fail if they don't work
echo "Installing PyG extensions (torch-scatter, torch-sparse, torch-cluster)..."
$PIP_CMD install pyg-lib torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html 2>&1 | grep -v "error\|ERROR" || {
    echo "⚠️  Warning: Some PyG extensions failed to install. This is OK for basic usage."
    echo "   You can install them manually later if needed."
}

# Install remaining requirements
echo "Installing other requirements..."
$PIP_CMD install -r requirements.txt

# Install package in editable mode
echo "Installing package in editable mode..."
$PIP_CMD install -e .

# Create necessary directories
echo "Creating output directories..."
mkdir -p data/raw data/processed outputs/checkpoints outputs/logs outputs/results

echo ""
echo "=========================================="
echo "✓ Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Implement model classes in src/models/"
echo ""
echo "3. Test sparsification methods:"
echo "   pytest tests/"
echo ""
echo "4. Run an experiment:"
echo "   python scripts/train.py"
echo ""
echo "See docs/QUICKSTART.md for more information."
