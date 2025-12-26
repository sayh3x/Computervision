from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-pose.pt")   #  yolov10s-pose.pt

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    annotated = results[0].plot()
    cv2.imshow("YOLO Pose", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
