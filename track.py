import pickle
import cv2
import os
import numpy as np
from utils import robust_bbox, resample_gftt, get_displacements
from visualize import draw_tracking_frame

# ==== CONFIGURATION ====
CAMERA = 'out13'
START_ANNOT_IDX = 1
FIRST_FRAME_25FPS = 2
FRAME_STEP_25FPS = 5
FRAME_DIR = 'data/frames'
INITIAL_FEATURES_PICKLE = 'output/processed/initial_features.pkl'
N_FRAMES_PER_VIDEO = 500
RESAMPLE_INTERVAL = 5
MIN_FEATURES = 30
STATIC_FRAMES = 5
STATIC_THRESH = 2.0

GFTT_PARAMS = {
    "maxCorners": 40,
    "qualityLevel": 0.01,
    "minDistance": 10,
    "blockSize": 3
}
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)
LK_ERR_THRESH = 30.0
DISPLACEMENT_THRESH = 80

OUTPUT_TRACKS_PICKLE = 'output/processed/resampled_tracks.pkl'

def main():
    with open(INITIAL_FEATURES_PICKLE, 'rb') as f:
        all_features = pickle.load(f)
    if CAMERA not in all_features or START_ANNOT_IDX not in all_features[CAMERA]:
        raise ValueError("No initial features for specified camera/frame.")

    start_frame_idx = FIRST_FRAME_25FPS + (START_ANNOT_IDX - 1) * FRAME_STEP_25FPS
    print(f"Tracking objects from {CAMERA} annotation index {START_ANNOT_IDX} (frame {start_frame_idx})")

    img_path = os.path.join(FRAME_DIR, CAMERA, f"{CAMERA}_frame_{start_frame_idx:04d}.jpg")
    img0 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

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
            'history': [[[pt] for pt in kp.reshape(-1, 2).tolist()]]
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

            pts_next, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, pts_prev, None, **LK_PARAMS)
            raw_pts = []
            prev_indices = []
            ref_pts = pts_prev.reshape(-1, 2)
            for idx, (p, s, e, p0) in enumerate(zip(pts_next.reshape(-1, 2), status.flatten(), err.flatten(), ref_pts)):
                if not s or e > LK_ERR_THRESH or np.linalg.norm(p - p0) > DISPLACEMENT_THRESH:
                    continue
                raw_pts.append(p)
                prev_indices.append(idx)

            if len(raw_pts) > 0:
                pts_arr = np.array(raw_pts)
                centroid = np.mean(pts_arr, axis=0)
                dists = np.linalg.norm(pts_arr - centroid, axis=1)
                std = np.std(dists)
                mean_dist = np.mean(dists)
                good_mask = dists < mean_dist + 2 * std
                filtered_pts = pts_arr[good_mask]
                filtered_indices = np.array(prev_indices)[good_mask]
            else:
                filtered_pts = np.empty((0, 2))
                filtered_indices = np.array([], dtype=int)

            curr_hist = []
            if filtered_pts.shape[0] > 0:
                for idx, pt in zip(filtered_indices, filtered_pts):
                    prev_traj = prev_histories[idx] if idx < len(prev_histories) else []
                    curr_hist.append(prev_traj + [pt.tolist()])
                disps = get_displacements(curr_hist, n_last=STATIC_FRAMES)
                group_med = np.median(disps)
                not_static_mask = disps > STATIC_THRESH if group_med > STATIC_THRESH else np.ones_like(disps, dtype=bool)
                filtered_pts = filtered_pts[not_static_mask]
                curr_hist = [h for h, m in zip(curr_hist, not_static_mask) if m]
            else:
                filtered_pts = np.empty((0, 2))
                curr_hist = []

            new_pts = filtered_pts.tolist()
            obj['frames'].append(f_idx)
            obj['keypoints'].append(new_pts)
            obj['history'].append(curr_hist)

            need_resample = (len(new_pts) < MIN_FEATURES) or ((f_idx - start_frame_idx) % RESAMPLE_INTERVAL == 0)
            if need_resample:
                bbox = robust_bbox(new_pts, expand_ratio=0.15, image_shape=next_gray.shape) if new_pts else obj['bboxes'][-1]
                kp = resample_gftt(next_gray, bbox, GFTT_PARAMS)
                kp_list = [list(pt) for pt in kp.reshape(-1, 2)]
                obj['keypoints'][-1] = kp_list
                bbox = robust_bbox(kp_list, expand_ratio=0.15, image_shape=next_gray.shape)
                obj['bboxes'].append(bbox if bbox is not None else obj['bboxes'][-1])
                obj['history'][-1] = [[pt] for pt in kp_list]
            else:
                bbox = robust_bbox(new_pts, expand_ratio=0.15, image_shape=next_gray.shape)
                obj['bboxes'].append(bbox if bbox is not None else obj['bboxes'][-1])

        prev_gray = next_gray

        vis = draw_tracking_frame(next_gray, objects)
        cv2.imshow("Tracking", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if (f_idx - start_frame_idx) % 10 == 0:
            print(f"Tracked up to frame {f_idx}")

    cv2.destroyAllWindows()
    os.makedirs(os.path.dirname(OUTPUT_TRACKS_PICKLE), exist_ok=True)
    with open(OUTPUT_TRACKS_PICKLE, 'wb') as f:
        pickle.dump(objects, f)
    print(f"\nSaved tracks with periodic resampling to {OUTPUT_TRACKS_PICKLE}")

if __name__ == "__main__":
    main()