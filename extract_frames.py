import cv2
import os

video_files = [
    ('out4', 'data/video/out4.mp4'),
    ('out13', 'data/video/out13.mp4')
]

output_root = 'data/frames'

for cam_name, video_path in video_files:
    output_dir = os.path.join(output_root, cam_name)
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{cam_name}: FPS detected: {fps}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, f"{cam_name}_frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path, frame)
        frame_idx += 1

    cap.release()
    print(f"Extracted {frame_idx} frames from {video_path} to {output_dir}")