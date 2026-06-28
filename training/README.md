# LoRA Training Module

Train LoRA cho Qwen-Image-Edit trên local hoặc GPU cloud.

## Cấu trúc

```
training/
├── __init__.py         # Package init
├── config.py           # Configuration management
├── dataset.py          # Dataset handling
├── train.py            # Main training script
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Cài đặt

```bash
cd isometric-map
pip install -r training/requirements.txt
```

## Dataset Format

Dataset cần có format như sau:

### CSV Format

```csv
image,control_image,prompt
output_000.png,input_000.png,Fill in the outlined section with coherent pixels matching the <isometric pixel art> style...
output_001.png,input_001.png,Fill in the outlined section with coherent pixels matching the <isometric pixel art> style...
```

### Thư mục

```
my_dataset/
├── metadata.csv
├── images/           # Target/output images
│   ├── image_000.png
└── controls/        # Input/control images
    ├── input_000.png
```

## Sử dụng

### 1. Train Local

```bash
# Train với config mặc định
python -m training.train --dataset-path ./my_dataset/metadata.csv --output ./output/lora

# Train với overrides
python -m training.train \
    --dataset-path ./my_dataset/metadata.csv \
    --epochs 5 \
    --lora-rank 32 \
    --lr 1e-4 \
    --output ./output/my_lora
```

### 2. Sử dụng như Module

```python
from training import train_lora, TrainingConfig

config = TrainingConfig(
    dataset_metadata_path="./my_dataset/metadata.csv",
    output_dir="./output/lora",
    num_epochs=5,
    lora_rank=32,
)

trainer = train_lora(config)
trainer.export_lora("./final_lora.safetensors")
```

## Configuration

### TrainingConfig

| Parameter | Default | Mô tả |
|-----------|---------|--------|
| `num_epochs` | 5 | Số epochs |
| `learning_rate` | 1e-4 | Learning rate |
| `train_batch_size` | 1 | Batch size |
| `gradient_accumulation_steps` | 4 | Gradient accumulation |
| `mixed_precision` | bfloat16 | Mixed precision |
| `use_gradient_checkpointing` | True | Gradient checkpointing |

### LoRAConfig

| Parameter | Default | Mô tả |
|-----------|---------|--------|
| `rank` | 32 | LoRA rank (higher = better quality, more VRAM) |
| `alpha` | 64 | LoRA alpha (usually 2x rank) |
| `target_modules` | (see code) | Target layers |

### Presets

```python
from training.config import get_preset

config = get_preset("fast")     # 3 epochs, rank 16
config = get_preset("quality")  # 10 epochs, rank 64
config = get_preset("low_vram") # rank 16, fp16
```

## Cloud Training (Git Clone)

```bash
# 1. Push code lên GitHub
git add .
git commit -m "ready for training"
git push origin main

# 2. Trên GPU server (RunPod/Vast.ai/etc)
git clone https://github.com/your-username/isometric-map.git
cd isometric-map

# Upload dataset (nếu chưa push lên git)
# scp -r ./my_dataset root@<server>:/workspace/dataset

# Cài đặt
pip install -r training/requirements.txt

# Train
python -m training.train \
    --dataset-path /workspace/my_dataset/metadata.csv \
    --epochs 5 \
    --lora-rank 32 \
    --output /workspace/output

# Download kết quả
# scp -r root@<server>:/workspace/output ./my_lora
```

## VRAM Requirements

| Rank | VRAM (Training) | Notes |
|------|-----------------|-------|
| 16 | ~20GB | Fast, OK quality |
| 32 | ~35GB | Balanced |
| 64 | ~50GB | High quality |

## Troubleshooting

### Out of Memory
- Giảm `train_batch_size` xuống 1
- Tăng `gradient_accumulation_steps`
- Giảm `lora_rank`

### Loss NaN
- Kiểm tra images có valid không
- Giảm `learning_rate`
- Thử `mixed_precision: bfloat16`

## Reference

- [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit)
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [PEFT LoRA](https://huggingface.co/docs/peft/en/conceptual_guides/lora)
