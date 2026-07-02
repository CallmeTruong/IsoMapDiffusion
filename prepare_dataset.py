import sys
import json
import csv
import argparse
import shutil
from pathlib import Path
from typing import Optional, List, Tuple

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.prepare import OmniMasker, DEFAULT_PROMPT
from src.constants import DEFAULT_PROMPT as CONSTANTS_PROMPT

# Use the same prompt consistently
COMMON_PROMPT = CONSTANTS_PROMPT

CONFIG = {
    'renders': PROJECT_ROOT / 'output' / 'renders',
    'generations': PROJECT_ROOT / 'generate',
    'output': PROJECT_ROOT / 'dataset',
    'size': 1024,
    'variants': 5,
    'max_pairs': None,
    'seed': 42,
    'prompt': COMMON_PROMPT,
}


def get_tile_coords(filename: str) -> Tuple[int, int]:
    """Extract x, y coordinates from tile filename."""
    parts = filename.replace('.png', '').split('_')
    return int(parts[1]), int(parts[2])


def get_tile_hash(filename: str) -> str:
    """Extract hash from tile filename."""
    return filename.rsplit('_', 1)[-1].replace('.png', '')


def find_matching_pair(renders_dir: Path, generations_dir: Path, render_file: Path) -> Optional[Path]:
    """Find the matching pixel art generation for a render."""
    render_hash = get_tile_hash(render_file.name)
    for gen_file in generations_dir.glob('*_*.png'):
        if gen_file.name.endswith(f'_{render_hash}.png'):
            return gen_file
        if get_tile_hash(gen_file.name) == render_hash:
            return gen_file
    return None


def prepare_new_dataset(
    renders_dir: Path,
    generations_dir: Path,
    output_dir: Path,
    resize_to: Tuple[int, int] = (1024, 1024),
    variants_per_pair: int = 5,
    max_pairs: Optional[int] = None,
    seed: int = 42,
    prompt: str = COMMON_PROMPT,
) -> dict:

    from PIL import Image
    import random
    random.seed(seed)
    
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images'
    control_dir = output_dir / 'control'
    prompts_dir = output_dir / 'prompts'
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize OmniMasker
    omni = OmniMasker(seed=seed)
    
    # Find valid render/pixel-art pairs
    renders = list(renders_dir.glob('tile_*.png'))
    pairs = []
    for render_path in renders:
        if render_path.stat().st_size < 50 * 1024:  # Skip tiny files
            continue
        pixel_art_path = find_matching_pair(renders_dir, generations_dir, render_path)
        if pixel_art_path is None:
            continue
        pairs.append((render_path, pixel_art_path))
    
    if max_pairs:
        pairs = pairs[:max_pairs]
    
    print(f"Found {len(pairs)} valid pairs")
    
    # Create COMMON prompt file
    common_prompt_path = prompts_dir / 'prompt.txt'
    with open(common_prompt_path, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"Created common prompt: {common_prompt_path}")
    
    # Process pairs and create samples
    samples = []
    image_index = 1
    
    for render_path, pixel_art_path in pairs:
        try:
            # Load and resize images
            render = Image.open(render_path).resize(resize_to, Image.Resampling.LANCZOS)
            pixel_art = Image.open(pixel_art_path).resize(resize_to, Image.Resampling.LANCZOS)
            
            coords = get_tile_coords(render_path.name)
            
            # Select diverse template types
            from src.dataset.omni import ALL_TYPES
            template_types = random.sample(ALL_TYPES, min(variants_per_pair, len(ALL_TYPES)))
            
            for variant_index, template_type in enumerate(template_types):
                # Get the input region (what part of pixel_art to show with render)
                region = omni.get_input_region(resize_to[0], resize_to[1], template_type)
                
                # Create control image (template with red border)
                control_image = omni.create_infill_template(pixel_art, render, region)
                
                # Create target image (full pixel art)
                target_image = pixel_art.copy()
                
                # Save files with sequential numbering
                img_name = f"image_{image_index:03d}"
                
                # Save target image (jpg) and caption (txt) to images/
                target_jpg_path = images_dir / f"{img_name}.jpg"
                target_image = target_image.convert('RGB')
                target_image.save(target_jpg_path, quality=95)
                
                # Caption = blank or minimal (based on image name)
                caption_path = images_dir / f"{img_name}.txt"
                with open(caption_path, 'w', encoding='utf-8') as f:
                    f.write(f"isometric pixel art tile at x={coords[0]}, y={coords[1]}")
                
                # Save control image to control/
                control_jpg_path = control_dir / f"{img_name}.jpg"
                control_image = control_image.convert('RGB')
                control_image.save(control_jpg_path, quality=95)
                
                # Store sample info
                samples.append({
                    'sample_id': img_name,
                    'caption_path': f"images/{img_name}.txt",
                    'target_path': f"images/{img_name}.jpg",
                    'control_path': f"control/{img_name}.jpg",
                    'prompt_path': "prompts/prompt.txt",
                    'mask_type': template_type,
                    'tile_coords': f"{coords[0]},{coords[1]}",
                })
                
                image_index += 1
                
        except Exception as e:
            print(f"Error processing pair: {e}")
            continue
    
    # Create mapping CSV
    mapping_csv_path = output_dir / 'dataset_mapping.csv'
    with open(mapping_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['caption_path', 'control_path', 'prompt_path'])
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                'caption_path': sample['caption_path'],
                'control_path': sample['control_path'],
                'prompt_path': sample['prompt_path'],
            })
    
    # Create metadata JSON
    metadata = {
        'total_samples': len(samples),
        'total_pairs': len(pairs),
        'variants_per_pair': variants_per_pair,
        'resize_to': list(resize_to),
        'seed': seed,
        'prompt': prompt,
        'common_prompt_file': 'prompts/prompt.txt',
        'structure_version': '2.0',
        'samples': samples,
    }
    
    metadata_json_path = output_dir / 'dataset_metadata.json'
    with open(metadata_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("Dataset created successfully!")
    print(f"{'='*60}")
    print(f"Dataset root: {output_dir}")
    print(f"Total samples: {len(samples)}")
    print(f"  - images/: {len(list(images_dir.glob('*.jpg')))} target images")
    print(f"  - control/: {len(list(control_dir.glob('*.jpg')))} control images")
    print(f"  - prompts/prompt.txt: Common prompt")
    print(f"  - dataset_mapping.csv: {len(samples)} rows")
    print(f"  - dataset_metadata.json: Full metadata")
    print(f"{'='*60}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Prepare LoRA dataset (NEW STRUCTURE)')
    parser.add_argument('--renders', type=str, help='Path to renders directory')
    parser.add_argument('--generations', type=str, help='Path to generations directory')
    parser.add_argument('--output', type=str, help='Path to output dataset directory')
    parser.add_argument('--size', type=int, default=1024, help='Image size (default: 1024)')
    parser.add_argument('--variants', type=int, default=5, help='Variants per pair (default: 5)')
    parser.add_argument('--max-pairs', type=int, default=None, help='Max pairs to process')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--prompt', type=str, default=None, help='Custom prompt text')
    
    args = parser.parse_args()
    
    renders_dir = Path(args.renders) if args.renders else CONFIG['renders']
    generations_dir = Path(args.generations) if args.generations else CONFIG['generations']
    output_dir = Path(args.output) if args.output else CONFIG['output']
    prompt = args.prompt if args.prompt else CONFIG['prompt']
    
    print("=" * 60)
    print("LoRA Dataset Preparation (NEW STRUCTURE)")
    print("=" * 60)
    print(f"  renders:      {renders_dir}")
    print(f"  generations:  {generations_dir}")
    print(f"  output:       {output_dir}")
    print(f"  size:        {args.size}x{args.size}")
    print(f"  variants:    {args.variants}")
    print(f"  max_pairs:   {args.max_pairs or 'all'}")
    print(f"  seed:        {args.seed}")
    print("=" * 60)
    
    # Validate input directories
    if not renders_dir.exists():
        print(f"ERROR: Renders directory not found: {renders_dir}")
        print("Please run render pipeline first or specify --renders")
        return
    
    if not generations_dir.exists():
        print(f"WARNING: Generations directory not found: {generations_dir}")
        print("Will try to find matching pairs anyway...")
    
    result = prepare_new_dataset(
        renders_dir=renders_dir,
        generations_dir=generations_dir,
        output_dir=output_dir,
        resize_to=(args.size, args.size),
        variants_per_pair=args.variants,
        max_pairs=args.max_pairs,
        seed=args.seed,
        prompt=prompt,
    )
    
    print(f"\nDataset ready at: {output_dir}")
    print("Next steps:")
    print("  1. Review images/ and control/ folders")
    print("  2. Edit prompts/prompt.txt if needed")
    print("  3. Use dataset_mapping.csv for training")


if __name__ == '__main__':
    main()
