#!/bin/bash
# Doodle Recognition Web App Launcher
# This script launches the web-based doodle recognition application

set -e

cd "$(dirname "$0")"

echo "🎨 Doodle Recognition Web App"
echo "=============================="

# Check for virtual environment
if [ -d "venv" ]; then
    echo "✓ Found virtual environment"
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "✓ Found virtual environment"
    source .venv/bin/activate
elif [ -n "$CONDA_PREFIX" ]; then
    echo "✓ Using conda environment: $CONDA_DEFAULT_ENV"
elif [ -n "$VIRTUAL_ENV" ]; then
    echo "✓ Using active virtual environment: $VIRTUAL_ENV"
else
    echo "⚠ No virtual environment found."
    echo "  Consider creating one with: uv venv"
fi

# Install dependencies using uv
echo "📦 Installing dependencies with uv (this may take a while for PyTorch)..."
if command -v uv &> /dev/null; then
    uv pip install flask torch torchvision opencv-python pillow numpy scikit-image scikit-learn
else
    echo "⚠ uv not found, using pip..."
    pip install flask torch torchvision opencv-python pillow numpy scikit-image scikit-learn
fi

echo "✓ Dependencies OK"
echo ""

# Kill any existing process on port 5001
lsof -ti:5001 | xargs kill -9 2>/dev/null || true

# Open browser after a short delay (in background)
(sleep 2 && open http://localhost:5001 2>/dev/null || xdg-open http://localhost:5001 2>/dev/null || echo "Open http://localhost:5001 in your browser") &

# Run the web app
echo "🚀 Starting server on http://localhost:5001"
echo "   Press Ctrl+C to stop"
echo ""
python3 web_app.py

