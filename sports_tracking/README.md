# Multi-Object Detection and Tracking Pipeline

This repository contains a robust computer vision pipeline for detecting and tracking multiple moving subjects in sports or event footage. It was built using Python, OpenCV, and the Ultralytics YOLOv8 framework with ByteTrack.

## Approach
1.  **Detection:** YOLOv8 (nano) is used to detect "persons" (Class 0) and "sports balls" (Class 32) in each frame.
2.  **Tracking:** The detections are fed into **ByteTrack**, a multi-object tracker that excels at handling partial occlusions by associating bounding boxes with lower detection scores.
3.  **Enhancement (Trajectory):** The script maintains a history of the tracklets' bottom-center coordinates to draw a trailing trajectory behind each tracked subject, providing a visual representation of their path over the last 30 frames.
4.  **Enhancement (Count):** A running total of all unique IDs ever detected is displayed.

## Installation Steps
1.  Clone this repository or download the files.
2.  Ensure you have Python 3.8+ installed.
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Dependencies include `ultralytics`, `opencv-python`, `numpy`, and `pandas`.*

## How to Run the Pipeline
To run the tracker on a video file, execute the following command:
```bash
python main.py --source input.mp4 --output output_tracked.mp4
```

### Parameters:
*   `--source`: (Required) Path to the input video file (e.g., `football_clip.mp4`).
*   `--output`: (Optional) Output video file name. Default: `output.mp4`.
*   `--model`: (Optional) Which YOLO model to use. Default: `yolov8n.pt`. Note that larger models (like `yolov8s.pt` or `yolov8m.pt`) will run slower but might detect further players.
*   `--classes`: (Optional) Which COCO classes to filter for. Default is `0 32` (person and sports ball).

## Limitations and Assumptions
*   **Camera Motion:** The tracker works best on static or smoothly panning cameras. Extreme, jerky camera movement might cause ID switching because the bounding boxes jump too far between frames for the Kalman Filter to predict accurately.
*   **Scale Changes:** Subjects moving very close to the camera might be given multiple IDs if their bounding boxes suddenly enclose only their torso instead of their full body.
*   **Assumptions:** The subjects are assumed to be humans (players) and a sports ball. The script filters out other noise (cars, birds, etc.).
