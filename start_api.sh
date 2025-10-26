#!/bin/bash

# Start the Video Re-rendering API Server
# This script sets up environment and starts the FastAPI server

set -e

echo "Starting Video Re-rendering API Server..."

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/ComfyUI"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Check if GPU is available
if command -v nvidia-smi > /dev/null 2>&1; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

# Check if required directories exist
if [ ! -d "ComfyUI" ]; then
    echo "Error: ComfyUI directory not found. Please ensure ComfyUI is properly installed."
    exit 1
fi

if [ ! -d "models" ]; then
    echo "Warning: models directory not found. Please ensure models are properly downloaded."
fi

# Create necessary directories
mkdir -p temp api_outputs logs

# Install API requirements if not already installed
if [ ! -f ".api_requirements_installed" ]; then
    echo "Installing additional API requirements..."
    pip install -r api_requirements.txt
    touch .api_requirements_installed
    echo "API requirements installed."
fi

# Start the server
echo "Starting FastAPI server on port $PORT..."
echo "API will be available at: http://localhost:$PORT"
echo "API documentation at: http://localhost:$PORT/docs"
echo ""

# Run the server with appropriate settings
uvicorn video_api_server:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1 \
    --timeout-keep-alive 300 \
    --access-log \
    --log-level info \
    --reload-delay 2 \
    "$@"