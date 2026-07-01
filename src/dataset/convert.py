"""
Convert dataset to Oxen format for fine-tuning.

Oxen requires a CSV with columns:
- image: target output image (relative path from repo root)
- control_image: input image with control signal (relative path)
- prompt: description of the edit

Usage:
    python src/dataset/convert_oxen.py
"""

import json
import csv
from pathlib import Path

from ..constants import DEFAULT_PROMPT


def convert_to_format(
    dataset_dir: str = None,
    output_csv: str = "dataset.csv",
    prompt: str = DEFAULT_PROMPT,
):
    """
    Convert existing dataset to Oxen format.

    Args:
        dataset_dir: Path to lora_dataset folder (default: PROJECT_ROOT/lora_dataset)
        output_csv: Output CSV filename
        prompt: Default prompt for all samples
    """
    project_root = Path(__file__).parent.parent.parent.resolve()
    dataset_dir = Path(dataset_dir or project_root / "lora_dataset")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    # Load metadata
    meta_path = dataset_dir / "dataset_metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    print(f"Converting {metadata['total_samples']} samples to Oxen format...")

    rows = []
    for sample in metadata["samples"]:
        # Get relative paths from dataset root
        template_abs = Path(sample["template_path"])
        target_abs = Path(sample["target_path"])
        
        # Use relative paths from project root
        template_path = template_abs.relative_to(project_root)
        target_path = target_abs.relative_to(project_root)

        rows.append({
            "image": str(target_path),           # Target (output)
            "control_image": str(template_path),  # Template (input)
            "prompt": prompt,
        })

    # Write CSV
    output_path = project_root / output_csv
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "control_image", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Created: {output_path}")
    print(f"Samples: {len(rows)}")
    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert dataset to format")
    parser.add_argument("--dataset", type=str, help="Dataset directory")
    parser.add_argument("--output", type=str, default="dataset.csv", help="Output CSV")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Training prompt")

    args = parser.parse_args()

    output = convert_to_format(
        dataset_dir=args.dataset,
        output_csv=args.output,
        prompt=args.prompt,
    )

    print(f"\nDataset ready for Oxen!")
    print(f"Upload to OxenHub: {output.parent}")
    print(f"CSV file: {output.name}")


if __name__ == "__main__":
    main()
