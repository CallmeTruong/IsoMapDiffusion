"""
Validate dataset structure for training.

NEW STRUCTURE (v2.0):
    dataset/
        images/
            image_001.jpg
            image_001.txt
            ...
        control/
            image_001.jpg
            ...
        prompts/
            prompt.txt
        dataset_mapping.csv
        dataset_metadata.json
"""

import os
import argparse
from pathlib import Path

import pandas as pd


def validate_new_structure(data_path: str) -> bool:
    """
    Validate new dataset structure (v2.0).

    Args:
        data_path: Path to dataset directory (contains images/, control/, prompts/)

    Returns:
        bool: True if valid, False otherwise
    """
    data_dir = Path(data_path)
    valid = True

    print(f"Validating dataset at: {data_dir}")
    print("=" * 50)

    # Check directories exist
    images_dir = data_dir / 'images'
    control_dir = data_dir / 'control'
    prompts_dir = data_dir / 'prompts'

    if not images_dir.exists():
        print(f"[ERROR] Missing: {images_dir}")
        valid = False
    else:
        print(f"[OK] images/ exists")

    if not control_dir.exists():
        print(f"[ERROR] Missing: {control_dir}")
        valid = False
    else:
        print(f"[OK] control/ exists")

    if not prompts_dir.exists():
        print(f"[WARN] Missing: {prompts_dir} (optional)")
    else:
        print(f"[OK] prompts/ exists")
        prompt_file = prompts_dir / 'prompt.txt'
        if prompt_file.exists():
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"[OK] prompts/prompt.txt exists ({len(content)} chars)")
        else:
            print(f"[WARN] No prompt.txt in prompts/")

    # Check images/ structure
    if images_dir.exists():
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        caption_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.txt')])

        print(f"[INFO] images/: {len(image_files)} .jpg files, {len(caption_files)} .txt files")

        # Check if captions match images
        jpg_stems = {f.rsplit('.', 1)[0] for f in image_files}
        txt_stems = {f.rsplit('.', 1)[0] for f in caption_files}

        if jpg_stems != txt_stems:
            missing_txt = jpg_stems - txt_stems
            missing_jpg = txt_stems - jpg_stems
            if missing_txt:
                print(f"[WARN] Missing .txt for: {list(missing_txt)[:5]}...")
            if missing_jpg:
                print(f"[WARN] Missing .jpg for: {list(missing_jpg)[:5]}...")
        else:
            print(f"[OK] All images have matching captions")

    # Check control/ structure
    if control_dir.exists():
        control_files = sorted([f for f in os.listdir(control_dir) if f.endswith('.jpg')])
        print(f"[INFO] control/: {len(control_files)} .jpg files")

    # Check CSV mapping
    csv_path = data_dir / 'dataset_mapping.csv'
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            print(f"[OK] dataset_mapping.csv exists ({len(df)} rows)")
            print(f"[INFO] Columns: {list(df.columns)}")
        except Exception as e:
            print(f"[ERROR] Failed to read CSV: {e}")
            valid = False
    else:
        print(f"[INFO] dataset_mapping.csv not found (optional)")

    # Check metadata JSON
    json_path = data_dir / 'dataset_metadata.json'
    if json_path.exists():
        import json
        try:
            with open(json_path, 'r') as f:
                meta = json.load(f)
            print(f"[OK] dataset_metadata.json exists ({meta.get('total_samples', '?')} samples)")
        except Exception as e:
            print(f"[ERROR] Failed to read JSON: {e}")
            valid = False
    else:
        print(f"[INFO] dataset_metadata.json not found (optional)")

    print("=" * 50)
    if valid:
        print("Dataset structure is VALID!")
    else:
        print("Dataset structure has ERRORS!")

    return valid


def main():
    parser = argparse.ArgumentParser(
        description="Validate dataset structure for training."
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the dataset directory to validate."
    )
    args = parser.parse_args()

    valid = validate_new_structure(args.path)
    return 0 if valid else 1


if __name__ == "__main__":
    exit(main())
