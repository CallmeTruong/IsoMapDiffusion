import os

os.environ['VIPS_PROGRESS'] = '1'

#worker
os.environ['VIPS_CONCURRENCY'] = '3'

vips_bin_path = r"vips\vips-dev-8.18\bin"
os.environ['PATH'] = vips_bin_path + ';' + os.environ.get('PATH', '')
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(vips_bin_path)

import pyvips
import json
import sys

pyvips.cache_set_max_mem(4000 * 1024 * 1024) 
pyvips.cache_set_max(2000)
pyvips.cache_set_max_files(1000)

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

    tiles_info.sort(key=lambda t: (t['y'], t['x']))

    images = []
    xs = []
    ys = []

    for t in tiles_info:
        if not os.path.exists(t['path']):
            continue
            
        img = pyvips.Image.new_from_file(t['path'], access='sequential')
        
        if img.bands < 4:
            img = img.addalpha()
        if img.interpretation != 'srgb':
            img = img.colourspace('srgb')

        images.append(img)
        xs.append(t['x'])
        ys.append(t['y'])

    if len(images) == 0:
        print("images not foud")
        sys.exit(1)

    background = pyvips.Image.black(canvas_w, canvas_h, bands=4).copy(interpretation='srgb')
    
    blend_modes = [2] * len(images)
    
    stitched_image = background.composite(images, blend_modes, x=xs, y=ys)

    print(f"\nStart Export DZI")
    try:
        stitched_image.dzsave(
            output_dzi_name, 
            overlap=0,          
            tile_size=512,      
            suffix='.webp[Q=95]'       
        )
        print("\nComplete! dzi saved at:\n   " + output_dzi_name + ".dzi")
    except pyvips.Error as e:
        print(f"\nError during execute: {e}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(src_dir)

    output_folder = os.path.join(project_root, "output")
    input_json = os.path.join(output_folder, "map_plan.json")
    output_name = os.path.join(output_folder, "gigapixel_map")
    
    build_dzi_from_plan(input_json, output_name)