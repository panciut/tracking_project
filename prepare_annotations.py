# prepare_annotations.py

import json
import os
from collections import defaultdict
import pickle  # <-- Add this import

COCO_PATH = 'data/annotations/_annotations.coco.json'

# Load COCO JSON
with open(COCO_PATH, 'r') as f:
    coco = json.load(f)

# Build image_id -> info mapping
image_id_to_info = {}
for img in coco['images']:
    image_id_to_info[img['id']] = {
        'file_name': img['file_name'],
        'height': img['height'],
        'width': img['width']
    }

# Build camera name -> {frame_index: [object_dicts]} mapping
camera_data = defaultdict(lambda: defaultdict(list))
category_id_to_name = {cat['id']: cat['name'] for cat in coco['categories']}
for ann in coco['annotations']:
    img_info = image_id_to_info[ann['image_id']]
    fname = img_info['file_name']
    # Parse camera name and frame index (assume filename: out13_frame_0094_png.rf.xxxx.jpg)
    base = os.path.basename(fname)
    parts = base.split('_frame_')
    cam_name = parts[0]  # e.g., 'out13'
    frame_part = parts[1].split('_')[0]  # e.g., '0094'
    frame_idx = int(frame_part)
    # Store info
    obj_dict = {
        'category_id': ann['category_id'],
        'category_name': category_id_to_name[ann['category_id']],
        'bbox': ann['bbox'],
        'annotation_id': ann['id'],
        'file_name': fname,
        'image_id': ann['image_id']
    }
    camera_data[cam_name][frame_idx].append(obj_dict)

# Example output (for sanity check)
print(f"Annotated frames for out4: {sorted(camera_data['out4'].keys())}")
print(f"Annotated frames for out13: {sorted(camera_data['out13'].keys())}")

# --- Save camera_data as a pickle file ---
output_dir = 'output/processed'
os.makedirs(output_dir, exist_ok=True)
pickle_path = os.path.join(output_dir, 'camera_data.pkl')

with open(pickle_path, 'wb') as f:
    pickle.dump(dict(camera_data), f)

print(f"\nSaved camera_data dictionary to {pickle_path}")