#!/bin/bash
# Verify training setup
# Usage: bash training/verify.sh

cd "$(dirname "$0")/.." || exit 1

echo "=== Training Setup Verification ==="
echo ""

echo "1. Checking dataset structure..."
if [ -d "lora_dataset/templates" ] && [ -d "lora_dataset/targets" ]; then
    TEMPLATES=$(ls lora_dataset/templates/*.png 2>/dev/null | wc -l)
    TARGETS=$(ls lora_dataset/targets/*_target.png 2>/dev/null | wc -l)
    echo "   ✓ templates/: $TEMPLATES files"
    echo "   ✓ targets/: $TARGETS files"
else
    echo "   ✗ Dataset not found. Run: python prepare_dataset.py"
fi
echo ""

echo "2. Checking config..."
if [ -f "training/config.yaml" ]; then
    echo "   ✓ config.yaml exists"
else
    echo "   ✗ config.yaml not found"
fi
echo ""

echo "3. Checking Python dependencies..."
python -c "import torch; import diffusers; import peft; print('   ✓ torch, diffusers, peft installed')" 2>/dev/null || echo "   ✗ Missing dependencies"
echo ""

echo "=== Ready to train ==="
echo "Run: python training/lora_train.py --config training/config.yaml"
