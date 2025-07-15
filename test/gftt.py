# test/gftt.py

import pickle
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# CONFIGURATION - Change as needed
FEATURES_PICKLE = 'output/processed/initial_features.pkl'
CAM = 'out13'  # 'out4' or 'out13'
ANNOT_IDX = 6 # Annotation index to visualize
FRAME_DIR = 'data/frames'

# These should match your feature extraction config!
FIRST_FRAME_25FPS = 2
FRAME_STEP_25FPS = 5

# --- Load detected features ---
with open(FEATURES_PICKLE, 'rb') as f:
    features = pickle.load(f)

if CAM not in features:
    raise ValueError(f"No features for camera {CAM}")
if ANNOT_IDX not in features[CAM]:
    raise ValueError(f"No features for camera {CAM}, annotation index {ANNOT_IDX}")

# --- Map annotation index to frame index ---
frame_idx = FIRST_FRAME_25FPS + (ANNOT_IDX - 1) * FRAME_STEP_25FPS
frame_path = os.path.join(FRAME_DIR, CAM, f"{CAM}_frame_{frame_idx:04d}.jpg")
img = cv2.imread(frame_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {frame_path}")

img_vis = img.copy()

for obj in features[CAM][ANNOT_IDX]:
    x, y, w, h = map(int, obj['bbox'])
    # Draw bounding box
    cv2.rectangle(img_vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Draw GFTT keypoints
    if obj['keypoints'].shape[0] > 0:
        for pt in obj['keypoints'].reshape(-1, 2):
            cv2.circle(img_vis, (int(pt[0]), int(pt[1])), 3, (0,255,0), -1)
    print(f"BBox {obj['category_name']} at {x,y,w,h}: {obj['keypoints'].shape[0]} features")

plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
plt.title(f"{CAM} annotation {ANNOT_IDX} = frame {frame_idx}")
plt.axis('off')
plt.show()