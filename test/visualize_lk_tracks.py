# test/visualize_lk_tracks.py

import pickle
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def robust_expanded_bbox(kp_list, std_thresh=1.5, min_pts=3, expand_ratio=0.15, image_shape=None):
    """
    kp_list: list of (x, y) coordinates
    std_thresh: std threshold for outlier removal
    min_pts: fallback to all pts if not enough
    expand_ratio: percent to expand (e.g., 0.15 = +15%)
    image_shape: (height, width) for clipping
    Returns: (x1, y1, x2, y2)
    """
    if len(kp_list) == 0:
        return None
    pts = np.array([np.array(pt).flatten() for pt in kp_list])
    if pts.shape[0] < min_pts:
        filtered_pts = pts
    else:
        med = np.median(pts, axis=0)
        std = np.std(pts, axis=0)
        mask = (np.abs(pts - med) <= std_thresh * (std + 1e-6)).all(axis=1)
        filtered_pts = pts[mask]
        if filtered_pts.shape[0] < min_pts:
            filtered_pts = pts  # fallback
    x1, y1 = np.min(filtered_pts, axis=0).astype(int)
    x2, y2 = np.max(filtered_pts, axis=0).astype(int)
    # Expand the box
    w = x2 - x1
    h = y2 - y1
    x1_exp = int(x1 - w * expand_ratio / 2)
    y1_exp = int(y1 - h * expand_ratio / 2)
    x2_exp = int(x2 + w * expand_ratio / 2)
    y2_exp = int(y2 + h * expand_ratio / 2)
    # Optional: clip to image size
    if image_shape is not None:
        H, W = image_shape[:2]
        x1_exp = max(0, x1_exp)
        y1_exp = max(0, y1_exp)
        x2_exp = min(W-1, x2_exp)
        y2_exp = min(H-1, y2_exp)
    return (x1_exp, y1_exp, x2_exp, y2_exp)

# === CONFIGURATION ===
CAM = 'out13'    # 'out4' or 'out13'
FRAME_TO_SHOW = 70  # Set to the video frame index you want to visualize (e.g., 30)
TRACKS_PICKLE = 'output/processed/lk_tracks.pkl'
FRAME_DIR = 'data/frames'

# --- Load tracks ---
with open(TRACKS_PICKLE, 'rb') as f:
    tracks = pickle.load(f)

if CAM not in tracks:
    raise ValueError(f"No tracks found for camera {CAM}")

img_path = os.path.join(FRAME_DIR, CAM, f"{CAM}_frame_{FRAME_TO_SHOW:04d}.jpg")
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Frame not found: {img_path}")
img_vis = img.copy()
img_h, img_w = img_vis.shape[:2]

# --- Draw tracks and bounding boxes ---
for annotation_idx in tracks[CAM]:
    for obj in tracks[CAM][annotation_idx]:
        # Find if this object was tracked in the frame to show
        if FRAME_TO_SHOW in obj['frames']:
            frame_pos = obj['frames'].index(FRAME_TO_SHOW)
            bbox = obj['bbox']
            kp_list = obj['keypoints'][frame_pos]
            # Draw original bbox (blue, fixed)
            x, y, w, h = map(int, bbox)
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Draw tracked keypoints (green)
            for pt in kp_list:
                pt = np.array(pt).flatten()
                cv2.circle(img_vis, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
            # Draw robust bbox with outlier removal + expansion (yellow)
            bbox_filt_exp = robust_expanded_bbox(
                kp_list,
                std_thresh=1.5,
                min_pts=3,
                expand_ratio=0.15,
                image_shape=(img_h, img_w)
            )
            if bbox_filt_exp is not None:
                x1, y1, x2, y2 = bbox_filt_exp
                cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow
            # Put label
            cv2.putText(img_vis, obj['category_name'], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
plt.title(f"{CAM} tracked frame {FRAME_TO_SHOW}")
plt.axis('off')
plt.show()