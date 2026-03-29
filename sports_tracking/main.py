import cv2
import argparse
from ultralytics import YOLO
from collections import defaultdict
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Object Tracking Pipeline")
    parser.add_argument("--source", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output video file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model to use (default: yolov8n.pt)")
    parser.add_argument("--classes", nargs='+', type=int, default=[0, 32], help="Classes to detect (default: 0 for person, 32 for sports ball)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the YOLOv8 model - it automatically downloads the weights if not present
    print(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    
    # Open the video file
    print(f"Opening video source: {args.source}")
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {args.source}")
        return

    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    # Dictionary to store the history of trajectories for each track_id
    track_history = defaultdict(lambda: [])

    print(f"Processing {args.source}...")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run tracking on the current frame
        # persist=True ensures IDs match across frames
        # classes=args.classes filters the detections (e.g., person only)
        # tracker="bytetrack.yaml" uses the ByteTrack algorithm (BoT-SORT is default but ByteTrack handles occlusion well)
        results = model.track(frame, persist=True, classes=args.classes, tracker="bytetrack.yaml", verbose=False)
        
        # Check if any objects were detected and tracked
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box
                color = (0, 255, 0) if cls == 0 else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label and ID
                label = f"ID: {track_id} {model.names[cls]} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Highlight trajectory (tracking the bottom center of the bounding box - roughly the feet)
                center_x = (x1 + x2) // 2
                center_y = int(y2)
                track_history[track_id].append((center_x, center_y))
                
                # Retain only the last 30 points for the trajectory tail
                if len(track_history[track_id]) > 30:
                    track_history[track_id].pop(0)

                # Draw trajectory points
                if len(track_history[track_id]) > 1:
                    points = np.hstack(track_history[track_id]).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)
        
        # Display Total Unique Subjects Counter
        total_unique_ids = len(track_history)
        cv2.putText(frame, f"Total Unique Subjects Tracked: {total_unique_ids}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Write the annotated frame to output video
        out.write(frame)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    # Cleanup
    cap.release()
    out.release()
    print(f"Finished processing! Total frames: {frame_count}. Video saved to {args.output}")

if __name__ == "__main__":
    main()
