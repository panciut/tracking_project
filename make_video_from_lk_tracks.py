import pickle
import cv2
import os
import numpy as np

# ---- CONFIG ----
CAM = 'out13'
TRACKS_PICKLE = 'output/processed/resampled_tracks.pkl'
FRAME_DIR = 'data/frames'
START_FRAME = 2
END_FRAME = 499

def robust_expanded_bbox(kp_list, std_thresh=1.5, min_pts=3, expand_ratio=0.15, image_shape=None):
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
            filtered_pts = pts
    x1, y1 = np.min(filtered_pts, axis=0).astype(int)
    x2, y2 = np.max(filtered_pts, axis=0).astype(int)
    w = x2 - x1
    h = y2 - y1
    x1_exp = int(x1 - w * expand_ratio / 2)
    y1_exp = int(y1 - h * expand_ratio / 2)
    x2_exp = int(x2 + w * expand_ratio / 2)
    y2_exp = int(y2 + h * expand_ratio / 2)
    if image_shape is not None:
        H, W = image_shape[:2]
        x1_exp = max(0, x1_exp)
        y1_exp = max(0, y1_exp)
        x2_exp = min(W-1, x2_exp)
        y2_exp = min(H-1, y2_exp)
    return (x1_exp, y1_exp, x2_exp, y2_exp)

with open(TRACKS_PICKLE, 'rb') as f:
    tracks = pickle.load(f)

frame_idx = START_FRAME
while True:
    img_path = os.path.join(FRAME_DIR, CAM, f"{CAM}_frame_{frame_idx:04d}.jpg")
    img = cv2.imread(img_path)
    if img is None:
        print(f"Frame {img_path} not found, stopping.")
        break
    img_h, img_w = img.shape[:2]

    num_tracked = 0
    for obj in tracks:
        if frame_idx in obj['frames']:
            pos = obj['frames'].index(frame_idx)
            curr_kps = obj['keypoints'][pos]
            bbox_yellow = robust_expanded_bbox(curr_kps, image_shape=(img_h, img_w))
            feat_count = len(curr_kps)
            if bbox_yellow:
                x1, y1, x2, y2 = bbox_yellow
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                label_pos = (x1, max(15, y1 - 10))
            else:
                label_pos = (10, 20)
            # Show object name and number of features
            cv2.putText(img, f"{obj['category_name']} ({feat_count})", label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            for pt in curr_kps:
                pt = np.array(pt).flatten()
                cv2.circle(img, (int(pt[0]), int(pt[1])), 3, (0,255,0), -1)
            num_tracked += 1

    # Show frame/global stats
    cv2.putText(img, f"Frame {frame_idx}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
    cv2.putText(img, f"Tracked objects: {num_tracked}", (40, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    cv2.imshow('Tracked', img)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('d'):
        # Next frame
        frame_idx += 1
        if frame_idx > END_FRAME:
            frame_idx = END_FRAME
    elif key == ord('a'):
        # Previous frame
        frame_idx -= 1
        if frame_idx < START_FRAME:
            frame_idx = START_FRAME
    elif key == ord('q') or key == 27:  # 27 is ESC
        break
    else:
        # Any other key: do nothing, just re-display
        pass

cv2.destroyAllWindows()