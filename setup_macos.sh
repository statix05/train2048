#!/bin/bash
# 🍎 Setup script for 2048 AI on macOS Apple Silicon

echo "🎮 2048 AI Setup for macOS"
echo "=========================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
echo "✓ Python: $PYTHON_VERSION"

# Check if we're on Apple Silicon
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    echo "✓ Architecture: Apple Silicon (arm64)"
else
    echo "⚠ Architecture: $ARCH (not Apple Silicon)"
fi

echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo ""
echo "📥 Installing PyTorch with MPS support..."
pip install --upgrade pip
pip install torch torchvision torchaudio

echo ""
echo "📥 Installing numpy..."
pip install numpy

echo ""
echo "🔍 Checking MPS availability..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    x = torch.zeros(1, device='mps')
    print('✓ MPS test passed!')
else:
    print('⚠ MPS not available, will use CPU')
"

echo ""
echo "🔍 Checking Tkinter (for GUI)..."
python3 -c "import tkinter; print('✓ Tkinter available')" 2>/dev/null || echo "⚠ Tkinter not found (GUI may not work)"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start using:"
echo "  source venv/bin/activate"
echo "  python main.py play          # Manual game (GUI)"
echo "  python main.py play --ai     # Watch AI play"
echo "  python main.py train --quick # Quick training (5 min)"
echo "  python main.py train         # Full training"
echo "  python main.py demo          # Console demo"
echo ""
