import os
import gc

os.environ['VIPS_PROGRESS'] = '1'
os.environ['VIPS_CONCURRENCY'] = '2'

vips_bin_path = r"D:\isometric-map\vips\vips-dev-8.18\bin"
os.environ['PATH'] = vips_bin_path + ';' + os.environ.get('PATH', '')
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(vips_bin_path)

import pyvips
import json
import sys

pyvips.cache_set_max_mem(2000 * 1024 * 1024)
pyvips.cache_set_max(500)
pyvips.cache_set_max_files(500)

STRIP_HEIGHT = 4096

def composite_strip(tiles_info, canvas_w, strip_y0, strip_y1):
    strip_h = strip_y1 - strip_y0

    strip_tiles = [
        t for t in tiles_info
        if os.path.exists(t['path'])
        and t['y'] < strip_y1
        and t['y'] + t.get('height', 512) > strip_y0
    ]

    background = pyvips.Image.black(canvas_w, strip_h, bands=4).copy(interpretation='srgb')

    if not strip_tiles:
        return background

    images, xs, ys = [], [], []
    for t in strip_tiles:
        img = pyvips.Image.new_from_file(t['path'], access='sequential')
        if img.bands < 4:
            img = img.addalpha()
        if img.interpretation != 'srgb':
            img = img.colourspace('srgb')
        images.append(img)
        xs.append(t['x'])
        ys.append(t['y'] - strip_y0)

    return background.composite(images, [2] * len(images), x=xs, y=ys)

def build_dzi_from_plan(json_path, output_dzi_name):
    if not os.path.exists(json_path):
        print(f"Missing file at:\n   {json_path}")
        sys.exit(1)

    print(f"Reading:\n   {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        plan = json.load(f)

    canvas_w = plan['canvasWidth']
    canvas_h = plan['canvasHeight']
    tiles_info = plan['tiles']

    print(f"Canvas size: {canvas_w} x {canvas_h}")
    print(f"Tiles: {len(tiles_info)}")

    temp_tiff = output_dzi_name + "_temp.tif"

    if os.path.exists(temp_tiff):
        print(f"\nFound existing TIFF, skipping Step 1...")
    else:
        # --- strip to TIF
        import math
        num_strips = math.ceil(canvas_h / STRIP_HEIGHT)
        strip_tiffs = []

        print(f"\nStep 1: Writing {num_strips} strips to TIFF...")

        for i in range(num_strips):
            strip_y0 = i * STRIP_HEIGHT
            strip_y1 = min(strip_y0 + STRIP_HEIGHT, canvas_h)
            print(f"  Strip {i+1}/{num_strips}: y={strip_y0}~{strip_y1}")

            strip_img = composite_strip(tiles_info, canvas_w, strip_y0, strip_y1)

            strip_path = output_dzi_name + f"_strip_{i}.tif"
            strip_img.tiffsave(
                strip_path,
                tile=True,
                tile_width=256,
                tile_height=256,
                compression='lzw',
                bigtiff=True,
            )
            strip_tiffs.append(strip_path)
            del strip_img

        # --- Merge strips
        print(f"\nMerging strips into single TIFF...")
        strip_images = [
            pyvips.Image.new_from_file(p, access='sequential')
            for p in strip_tiffs
        ]
        # Merge all
        print(f"\nMerging {len(strip_tiffs)} strips into single TIFF...")
        strip_images = []
        for p in strip_tiffs:
            img = pyvips.Image.new_from_file(p, access='sequential')
            strip_images.append(img)

        full_image = pyvips.Image.arrayjoin(strip_images, across=1)

        full_image.tiffsave(
            temp_tiff,
            tile=True,
            tile_width=256,
            tile_height=256,
            compression='lzw',
            bigtiff=True,
        )
        del full_image, strip_images
        gc.collect()

        for p in strip_tiffs:
            try:
                os.remove(p)
            except PermissionError:
                print(f"   Warning: cannot delete {p}, pls remove by hand")

        print(f"  TIFF done.")

    # ---dzsave from TIFF ---
    print(f"\nStep 2: Exporting DZI from TIFF...")
    tiff_img = pyvips.Image.new_from_file(temp_tiff, access='sequential')
    tiff_img.dzsave(
        output_dzi_name,
        overlap=0,
        tile_size=512,
        suffix='.jpg[Q=95]',
    )

    # Clean
    del tiff_img
    gc.collect()

    print(f"\nCleaning up temp file...")
    try:
        os.remove(temp_tiff)
    except PermissionError:
        print(f"   Warning: cannot delete temp file, del by hand at:\n   {temp_tiff}")

    print(f"\nComplete! DZI saved at:\n   {output_dzi_name}.dzi")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(src_dir)

    output_folder = os.path.join(project_root, "output")
    input_json = os.path.join(output_folder, "map_plan.json")
    output_name = os.path.join(output_folder, "gigapixel_map")

    build_dzi_from_plan(input_json, output_name)