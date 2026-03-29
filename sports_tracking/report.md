# Technical Report: Multi-Object Detection and Tracking

## 1. Model / Detector Selection
For the object detection phase, I selected **YOLOv8** (specifically `yolov8n`, the nano version) built by Ultralytics. 
**Why this combination was selected:** YOLO models represent the state-of-the-art for real-time object detection. The v8 architecture provides exceptional speed-to-accuracy ratios. For a sports environment where players move rapidly and frames blur, having a fast anchor-free detector ensures that bounding boxes are drawn tightly around the subjects in nearly every frame. The nano version was selected to ensure it can run efficiently even on standard CPU setups if a GPU is not available.

## 2. Tracking Algorithm
I utilized **ByteTrack** as the primary tracking algorithm integrated within the Ultralytics suite. 
**How ID consistency is maintained:** ByteTrack maintains consistency by leveraging the bounding boxes provided by YOLOv8 and applying a Kalman filter to predict the subject's location in the next frame. Crucially, ByteTrack differs from older trackers (like SORT or DeepSORT) by retaining "low confidence" detection boxes. In sports footage where a player is partially occluded by another player, the detector's confidence score drops. Instead of deleting this track, ByteTrack associates this low-confidence box with the existing tracklet, effectively "saving" the ID through the occlusion.

## 3. Challenges Faced & Failure Cases
Throughout the development and testing of this pipeline, several real-world video challenges were observed:
1.  **Severe Occlusion & ID Switching:** When two players completely cross each other's path and have very similar appearances, the Kalman Filter prediction sometimes struggles to disentangle them upon separation, leading to the IDs swapping.
2.  **Fast Camera Panning:** Sudden camera zooms or rapid panning occasionally causes the tracker to lose subjects because their location jumps significantly farther than the tracker predicted.
3.  **The Sports Ball:** Detecting the ball (Class 32) is exceptionally difficult due to motion blur and its small size. The tracker frequently drops the ball's ID because it moves irregularly and is often perfectly hidden behind players.

## 4. Possible Improvements
If given more time and resources, this pipeline could be enhanced significantly:
1.  **Appearance Embeddings (Re-ID):** While ByteTrack is excellent at spatial tracking, adding a visual embedding model (like BoT-SORT or DeepSORT) would drastically reduce ID switching. The model would learn the colors/features of a player's jersey and reassign the correct ID even after a complete disappearance.
2.  **Camera Motion Compensation (CMC):** Implementing an algorithm that tracks static background features (like the lines on the field/court) to calculate the affine matrix of the camera's movement would stabilize the tracked coordinates, greatly improving tracking during pans/zooms.
3.  **Homography / Top-Down View:** Warping the frame onto a 2D top-down "minimap" of the court would allow for strategic analysis of player formations and distance calculations.
