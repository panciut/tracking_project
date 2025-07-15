# track_with_resampling.py

import pickle
import cv2
import os
import numpy as np

# ==== CONFIGURATION ====
CAMERA = 'out13'                 # 'out4' or 'out13'
START_ANNOT_IDX = 1              # Index of the annotated frame to start from
FIRST_FRAME_25FPS = 2            # Map annotated index to frame idx (should match pipeline)
FRAME_STEP_25FPS = 5
FRAME_DIR = 'data/frames'
INITIAL_FEATURES_PICKLE = 'output/processed/initial_features.pkl'
N_FRAMES_PER_VIDEO = 500         # Adjust for your video
RESAMPLE_INTERVAL = 5            # Frames between resampling
MIN_FEATURES = 30                # If fewer features, always resample
STATIC_FRAMES = 5                # Number of frames to check for static points
STATIC_THRESH = 2.0              # Max pixel movement to consider a point "static"

# GFTT parameters
GFTT_PARAMS = {
    "maxCorners": 40,
    "qualityLevel": 0.01,
    "minDistance": 10,
    "blockSize": 3
}
# LK params
LK_PARAMS = dict(
    winSize  = (15, 15),
    maxLevel = 2,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)
# Outlier and error threshold
LK_ERR_THRESH = 30.0     # Remove features with LK error above this
DISPLACEMENT_THRESH = 80 # Remove features that move too far

OUTPUT_TRACKS_PICKLE = 'output/processed/resampled_tracks.pkl'

def robust_bbox(kp_list, expand_ratio=0.15, image_shape=None):
    if len(kp_list) == 0:
        return None
    pts = np.array([np.array(pt).flatten() for pt in kp_list])
    x1, y1 = np.min(pts, axis=0).astype(int)
    x2, y2 = np.max(pts, axis=0).astype(int)
    # Expand box
    w = x2 - x1
    h = y2 - y1
    x1_exp = int(x1 - w * expand_ratio / 2)
    y1_exp = int(y1 - h * expand_ratio / 2)
    x2_exp = int(x2 + w * expand_ratio / 2)
    y2_exp = int(y2 + h * expand_ratio / 2)
    if image_shape:
        H, W = image_shape[:2]
        x1_exp = max(0, x1_exp)
        y1_exp = max(0, y1_exp)
        x2_exp = min(W-1, x2_exp)
        y2_exp = min(H-1, y2_exp)
    return (x1_exp, y1_exp, x2_exp-x1_exp, y2_exp-y1_exp)

def resample_gftt(gray, bbox, gftt_params):
    x, y, w, h = [int(v) for v in bbox]
    roi = gray[y:y+h, x:x+w]
    if roi.shape[0] < 5 or roi.shape[1] < 5:
        return np.empty((0, 1, 2), dtype=np.float32)
    kp = cv2.goodFeaturesToTrack(roi, **gftt_params)
    if kp is not None:
        kp = kp.reshape(-1, 2)
        kp[:, 0] += x
        kp[:, 1] += y
        return kp.reshape(-1, 1, 2)
    else:
        return np.empty((0, 1, 2), dtype=np.float32)

def get_displacements(history, n_last=STATIC_FRAMES):
    disp_list = []
    for traj in history:
        traj_arr = np.array(traj[-n_last:])
        if traj_arr.shape[0] < 2:
            disp_list.append(0)
        else:
            disp_list.append(np.max(np.linalg.norm(traj_arr - traj_arr[0], axis=1)))
    return np.array(disp_list)

def main():
    # Load initial features
    with open(INITIAL_FEATURES_PICKLE, 'rb') as f:
        all_features = pickle.load(f)
    if CAMERA not in all_features or START_ANNOT_IDX not in all_features[CAMERA]:
        raise ValueError("No initial features for specified camera/frame.")

    # Map annotation index to video frame index
    start_frame_idx = FIRST_FRAME_25FPS + (START_ANNOT_IDX - 1) * FRAME_STEP_25FPS
    print(f"Tracking objects from {CAMERA} annotation index {START_ANNOT_IDX} (frame {start_frame_idx})")

    # Load initial image
    img_path = os.path.join(FRAME_DIR, CAMERA, f"{CAMERA}_frame_{start_frame_idx:04d}.jpg")
    img0 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Initialize object tracks
    objects = []
    for obj in all_features[CAMERA][START_ANNOT_IDX]:
        kp = obj['keypoints']
        if kp.shape[0] == 0:
            continue
        track = {
            'category_id': obj['category_id'],
            'category_name': obj['category_name'],
            'frames': [start_frame_idx],
            'bboxes': [obj['bbox']],
            'keypoints': [kp.reshape(-1, 2).tolist()],
            'history': [[ [pt] for pt in kp.reshape(-1, 2).tolist() ]] # histories for each feature
        }
        objects.append(track)

    prev_gray = img0
    for f_idx in range(start_frame_idx + 1, N_FRAMES_PER_VIDEO):
        img_path = os.path.join(FRAME_DIR, CAMERA, f"{CAMERA}_frame_{f_idx:04d}.jpg")
        next_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if next_gray is None:
            break

        for obj in objects:
            pts_prev = np.array(obj['keypoints'][-1], dtype=np.float32).reshape(-1, 1, 2)
            prev_histories = obj['history'][-1] if len(obj['history']) > 0 else []
            if pts_prev.shape[0] == 0:
                obj['frames'].append(f_idx)
                obj['bboxes'].append(obj['bboxes'][-1])
                obj['keypoints'].append([])
                obj['history'].append([])
                continue

            # Optical flow tracking
            pts_next, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, pts_prev, None, **LK_PARAMS)
            # Outlier removal
            raw_pts = []
            prev_indices = []
            ref_pts = pts_prev.reshape(-1, 2)
            for idx, (p, s, e, p0) in enumerate(zip(pts_next.reshape(-1, 2), status.flatten(), err.flatten(), ref_pts)):
                if not s:
                    continue
                if e > LK_ERR_THRESH:
                    continue
                disp = np.linalg.norm(p - p0)
                if disp > DISPLACEMENT_THRESH:
                    continue
                raw_pts.append(p)
                prev_indices.append(idx)

            if len(raw_pts) > 0:
                pts_arr = np.array(raw_pts)
                centroid = np.mean(pts_arr, axis=0)
                dists = np.linalg.norm(pts_arr - centroid, axis=1)
                std = np.std(dists)
                mean_dist = np.mean(dists)
                good_mask = dists < mean_dist + 2*std
                filtered_pts = pts_arr[good_mask]
                filtered_indices = np.array(prev_indices)[good_mask]
            else:
                filtered_pts = np.empty((0,2))
                filtered_indices = np.array([], dtype=int)

            # --- Group-aware static removal ---
            curr_hist = []
            if filtered_pts.shape[0] > 0:
                # Update histories: assign by previous index
                for idx, pt in zip(filtered_indices, filtered_pts):
                    prev_traj = prev_histories[idx] if idx < len(prev_histories) else []
                    curr_hist.append(prev_traj + [pt.tolist()])
                # Compute displacements for all
                disps = get_displacements(curr_hist, n_last=STATIC_FRAMES)
                group_med = np.median(disps)
                # Remove static only if group median is high
                if group_med > STATIC_THRESH:
                    not_static_mask = disps > STATIC_THRESH
                else:
                    not_static_mask = np.ones_like(disps, dtype=bool)
                filtered_pts = filtered_pts[not_static_mask]
                curr_hist = [h for h, m in zip(curr_hist, not_static_mask) if m]
            else:
                filtered_pts = np.empty((0,2))
                curr_hist = []

            new_pts = filtered_pts.tolist()
            obj['frames'].append(f_idx)
            obj['keypoints'].append(new_pts)
            obj['history'].append(curr_hist)

            # Re-detect features if too few, or every RESAMPLE_INTERVAL
            need_resample = (len(new_pts) < MIN_FEATURES) or ((f_idx - start_frame_idx) % RESAMPLE_INTERVAL == 0)
            if need_resample:
                if len(new_pts):
                    bbox = robust_bbox(new_pts, expand_ratio=0.15, image_shape=next_gray.shape)
                else:
                    bbox = obj['bboxes'][-1]
                kp = resample_gftt(next_gray, bbox, GFTT_PARAMS)
                kp_list = [list(pt) for pt in kp.reshape(-1, 2)]
                obj['keypoints'][-1] = kp_list
                bbox = robust_bbox(kp_list, expand_ratio=0.15, image_shape=next_gray.shape)
                obj['bboxes'].append(bbox if bbox is not None else obj['bboxes'][-1])
                # Reset histories for new features
                obj['history'][-1] = [[pt] for pt in kp_list]
            else:
                bbox = robust_bbox(new_pts, expand_ratio=0.15, image_shape=next_gray.shape)
                obj['bboxes'].append(bbox if bbox is not None else obj['bboxes'][-1])

        prev_gray = next_gray

        if (f_idx - start_frame_idx) % 10 == 0:
            print(f"Tracked up to frame {f_idx}")

    os.makedirs(os.path.dirname(OUTPUT_TRACKS_PICKLE), exist_ok=True)
    with open(OUTPUT_TRACKS_PICKLE, 'wb') as f:
        pickle.dump(objects, f)
    print(f"\nSaved tracks with periodic resampling to {OUTPUT_TRACKS_PICKLE}")

if __name__ == "__main__":
    main()