---
description: Initialize SAM3 development environment
---

# SAM3 Project Initialization

## Prerequisites
- Python 3.12 or higher
- **CUDA-compatible GPU with CUDA 12.6+** (required for Triton kernels)
- Conda package manager

> ⚠️ **Important**: SAM3 requires an NVIDIA GPU with Triton support. It will NOT work on macOS (Apple Silicon) or CPU-only systems due to the Triton dependency for EDT (Euclidean Distance Transform) kernels.

## Setup Steps

### 1. Create Conda Environment
```bash
conda create -n sam3 python=3.12 -y
conda activate sam3
```

### 2. Install PyTorch with CUDA Support
```bash
# For CUDA 12.6
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 3. Install SAM3 Package
```bash
# Basic installation
pip install -e .

# With notebook dependencies (for running examples)
pip install -e ".[notebooks]"

# For development
pip install -e ".[train,dev]"
```

### 4. Hugging Face Authentication
SAM3 checkpoints require access approval:
1. Request access at https://huggingface.co/facebook/sam3
2. Generate an access token at https://huggingface.co/settings/tokens
3. Run `huggingface-cli login` and paste your token

## Running Examples

```bash
# Install notebook dependencies first
pip install -e ".[notebooks]"

# Start Jupyter
jupyter notebook examples/sam3_image_predictor_example.ipynb
```

## Example Notebooks
- `sam3_image_predictor_example.ipynb` - Text and visual box prompts on images
- `sam3_video_predictor_example.ipynb` - Text prompts on videos with refinements
- `sam3_image_batched_inference.ipynb` - Batched inference on images
- `sam3_agent.ipynb` - SAM 3 Agent for complex text prompts

## Development Commands
```bash
# Format code
ufmt format .

# Run tests
pytest

# Install dev dependencies
pip install -e ".[dev,train]"
```
