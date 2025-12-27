from ultralytics import YOLO
import cv2
import numpy as np
from src.config.settings import MODEL_PATH

# Keypoint names corresponding to YOLO pose output indices
keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Load YOLO pose model
model = YOLO(str(MODEL_PATH))

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # Frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # Frame height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run pose inference
    results = model(frame, conf=0.5)
    annotated = results[0].plot()

    if results[0].keypoints is not None:
        # Convert keypoints to numpy arrays
        # keypoints shape: (num_people, num_keypoints, 2)
        keypoints = results[0].keypoints.xy.numpy()   # x, y coordinates
        confs = results[0].keypoints.conf.numpy()     # confidence scores

        for person_idx in range(keypoints.shape[0]):
            for k_idx in range(keypoints.shape[1]):
                x, y = keypoints[person_idx, k_idx]
                conf = confs[person_idx, k_idx]

                if conf > 0.3:
                    cv2.circle(
                        annotated,
                        (int(x), int(y)),
                        3,
                        (0, 0, 255),
                        -1
                    )
                    cv2.putText(
                        annotated,
                        keypoint_names[k_idx],
                        (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),  # Green color
                        1
                    )

    cv2.imshow("Elevator", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
