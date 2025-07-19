import numpy as np
import cv2

def robust_bbox(kp_list, expand_ratio=0.15, image_shape=None):
    if len(kp_list) == 0:
        return None
    pts = np.array([np.array(pt).flatten() for pt in kp_list])
    x1, y1 = np.min(pts, axis=0).astype(int)
    x2, y2 = np.max(pts, axis=0).astype(int)
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

def get_displacements(history, n_last):
    disp_list = []
    for traj in history:
        traj_arr = np.array(traj[-n_last:])
        if traj_arr.shape[0] < 2:
            disp_list.append(0)
        else:
            disp_list.append(np.max(np.linalg.norm(traj_arr - traj_arr[0], axis=1)))
    return np.array(disp_list)