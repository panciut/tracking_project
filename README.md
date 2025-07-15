# 2D Tracking Project – Computer Vision @ UniTN

## Project Overview

This project implements 2D tracking of players, referees, and the ball in a sports scenario using **two synchronized camera views** (`out13` and `out4`).  
The workflow strictly follows the requirements for group projects in the Computer Vision course (A.Y. 2024/2025, UniTN), using classical feature-based tracking methods.

## Goals

- Track the motion of all annotated objects across video frames at the native frame rate (25fps) for **both camera views**
- Evaluate tracking performance **only** on manually annotated frames (ground truth at 5fps)
- Visualize and report numerical results for the annotated frames
- (Optionally) Prepare for 3D triangulation/trajectory reconstruction using multi-view tracking

## Dataset

- Videos and COCO-format bounding box annotations provided by course tutors
- Manual annotation at 5fps, via Roboflow, for both `out13` and `out4`
- Two camera views (as required for a two-person team project)

---

## Pipeline / Step-by-Step Plan

### 1. **Data Preparation**

- Extract all frames from `out13` and `out4` videos at 25fps and store as images
- Check that annotations are mapped correctly to frame indices for both views

### 2. **Tracking Algorithm**

#### a) **Good Features to Track (GFTT)**
- Use OpenCV’s `cv2.goodFeaturesToTrack` on annotated frames to detect strong, trackable points within each object’s bounding box (players, referees, ball), for each camera

#### b) **Lucas-Kanade Optical Flow (LK)**
- Track detected features frame-by-frame using OpenCV’s `cv2.calcOpticalFlowPyrLK`, independently for each camera view

#### c) **[Optional] Kalman Filter**
- Apply a Kalman filter to smooth/predict object locations between annotated frames
- Update (correct) the filter only when a manual annotation is available (every 5th frame)

### 3. **Evaluation**

- Compare tracking predictions to manual annotations only on annotated frames (in both cameras)
- Compute and report standard metrics: Intersection over Union (IoU), center point error, or other relevant tracking metrics

### 4. **Visualization**

- Overlay predictions and ground truth boxes on frames for both cameras
- Optionally, create short demo videos for the report

### 5. **Reporting**

- Document methods, results, and discussion in a concise report (3–6 pages, LNCS format)
- Prepare a demo video (max 5 minutes) explaining the approach and showing the tracking results

---

## Key Notes

- **Evaluation uses only manually annotated frames in both views**
- Interpolated boxes may be used for visualization but not for evaluation metrics
- No deep learning is required (classical tracking only)
- Code is implemented in Python with OpenCV and standard scientific libraries

---