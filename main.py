# main.py

import os
import sys

def run_script(script_name):
    print(f"\n===== Running: {script_name} =====")
    exit_code = os.system(f"python {script_name}")
    if exit_code != 0:
        print(f"Error: {script_name} failed (exit code {exit_code})")
        sys.exit(1)

def main():
    # Step 1: Extract frames from videos
    run_script('extract_frames.py')

    # Step 2: Prepare annotation pickle from COCO JSON
    run_script('prepare_annotations.py')

    # Step 3: Detect GFTT features in annotated frames
    run_script('detect_gftt_features.py')

    # Step 4: Track features with LK optical flow
    run_script('lk_track_features.py')

    #test on video frames
    run_script('make_video_from_lk_tracks.py')

    print("\n===== Pipeline complete! =====\n")

if __name__ == "__main__":
    main()