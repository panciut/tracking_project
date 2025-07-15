# detect_gftt_features.py

import pickle
import cv2
import os
import numpy as np

# === CONFIGURATION ===
CAMERAS = ['out4', 'out13']
FRAME_DIR = 'data/frames'
ANNOTATION_PICKLE = 'output/processed/camera_data.pkl'
OUTPUT_PICKLE = 'output/processed/initial_features.pkl'

# Mapping from annotation index to video frame index:
FIRST_FRAME_25FPS = 2      # <--- Set to frame index at 25fps for your first annotation (e.g. 2)
FRAME_STEP_25FPS = 5       # <--- Step between annotated frames

# GFTT parameters (tune here!)
GFTT_PARAMS = {
    "maxCorners": 40,      # Max features per bbox
    "qualityLevel": 0.005,  # Lower = more (but weaker) corners
    "minDistance": 10,      # Minimum distance between corners
    "blockSize": 3         # Size of block for covariance calculation
}

# Load annotations
with open(ANNOTATION_PICKLE, 'rb') as f:
    camera_data = pickle.load(f)

features = {}

for cam in CAMERAS:
    features[cam] = {}
    for annotated_idx in sorted(camera_data[cam].keys()):
        annotated_objects = camera_data[cam][annotated_idx]
        # Map annotation index to video frame index
        frame_idx = FIRST_FRAME_25FPS + (annotated_idx - 1) * FRAME_STEP_25FPS
        frame_path = os.path.join(FRAME_DIR, cam, f"{cam}_frame_{frame_idx:04d}.jpg")
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Warning: Could not load {frame_path}")
            continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features[cam][annotated_idx] = []
        for obj in annotated_objects:
            x, y, w, h = map(int, obj['bbox'])
            roi = img_gray[y:y+h, x:x+w]
            if roi.shape[0] < 5 or roi.shape[1] < 5:
                kp_array = np.empty((0, 1, 2), dtype=np.float32)
            else:
                kp = cv2.goodFeaturesToTrack(
                    roi,
                    maxCorners=GFTT_PARAMS["maxCorners"],
                    qualityLevel=GFTT_PARAMS["qualityLevel"],
                    minDistance=GFTT_PARAMS["minDistance"],
                    blockSize=GFTT_PARAMS["blockSize"]
                )
                if kp is not None:
                    kp = kp.reshape(-1, 2)
                    kp[:, 0] += x
                    kp[:, 1] += y
                    kp_array = kp.reshape(-1, 1, 2)
                else:
                    kp_array = np.empty((0, 1, 2), dtype=np.float32)
            features[cam][annotated_idx].append({
                'category_id': obj['category_id'],
                'category_name': obj['category_name'],
                'bbox': obj['bbox'],
                'keypoints': kp_array
            })
        print(f"{cam}, annotated frame {annotated_idx} (video frame {frame_idx}): detected features for {len(features[cam][annotated_idx])} objects.")

os.makedirs(os.path.dirname(OUTPUT_PICKLE), exist_ok=True)
with open(OUTPUT_PICKLE, 'wb') as f:
    pickle.dump(features, f)

print(f"Saved initial features to {OUTPUT_PICKLE}")