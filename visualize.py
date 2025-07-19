import cv2
import numpy as np

def draw_tracking_frame(gray, objects):
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for obj in objects:
        for pt in obj['keypoints'][-1]:
            cv2.circle(vis, tuple(np.round(pt).astype(int)), 2, (0, 255, 0), -1)
        if obj['bboxes'][-1] is not None:
            x, y, w, h = map(int, obj['bboxes'][-1])
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 1)
    return vis