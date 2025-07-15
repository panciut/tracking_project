# test/evaluate_tracks.py

import pickle
import os
import numpy as np

ANNOT_PICKLE = 'output/processed/camera_data.pkl'
TRACKS_PICKLE = 'output/processed/lk_tracks.pkl'

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
    return iou

with open(ANNOT_PICKLE, 'rb') as f:
    annotations = pickle.load(f)
with open(TRACKS_PICKLE, 'rb') as f:
    tracks = pickle.load(f)

CAM = 'out4'  # or 'out13'
all_ious = []
for frame_idx in sorted(annotations[CAM].keys()):
    ann_objs = annotations[CAM][frame_idx]
    # Find tracked objects present in this frame
    for tracked_ann_idx in tracks[CAM]:
        for obj in tracks[CAM][tracked_ann_idx]:
            if frame_idx in obj['frames']:
                pos = obj['frames'].index(frame_idx)
                keypoints = np.array(obj['keypoints'][pos])
                if keypoints.shape[0] < 2:
                    continue
                # Robust bbox (use your robust_expanded_bbox here)
                x1, y1 = np.min(keypoints, axis=0)
                x2, y2 = np.max(keypoints, axis=0)
                pred_bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
                # Find the best matching annotation
                best_iou = 0
                for gt in ann_objs:
                    gt_bbox = [int(gt['bbox'][0]), int(gt['bbox'][1]), int(gt['bbox'][2]), int(gt['bbox'][3])]
                    curr_iou = iou(pred_bbox, gt_bbox)
                    if curr_iou > best_iou:
                        best_iou = curr_iou
                all_ious.append(best_iou)
print(f"Mean IoU on annotated frames: {np.mean(all_ious):.3f}")