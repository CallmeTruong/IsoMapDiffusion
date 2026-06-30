#!/bin/bash
# Train LoRA with config
# Usage: bash training/train.sh [config_path]

CONFIG_PATH="${1:-training/config.yaml}"

# Get absolute path to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT" || exit 1

# Set environment variables for GPU training
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

echo "=========================================="
echo "LoRA Training - Isometric Map Infilling"
echo "=========================================="
echo "Project: $PROJECT_ROOT"
echo "Config: $CONFIG_PATH"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "=========================================="

python -m training.lora_train --config "$CONFIG_PATH"
