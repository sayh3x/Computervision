import cv2
import time
import requests
from ultralytics import YOLO

# ================= CONFIG =================
MODEL_PATH = "best.pt"
# VIDEO_SOURCE = "input.mp4"
VIDEO_SOURCE = 0
FALL_CLASS_ID = 0
CONF_THRESHOLD = 0.6
API_URL = "http://127.0.0.1:8000/fall-event"
COOLDOWN_SECONDS = 10 
# ==========================================

model = YOLO(MODEL_PATH)

last_fall_time = 0

def send_fall_event(confidence):
    payload = {
        "event": "fall_detected",
        "confidence": float(confidence),
        "timestamp": time.time()
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=2)
        print(f"[API] Sent | Status: {response.status_code}")
    except Exception as e:
        print(f"[API ERROR] {e}")

cap = cv2.VideoCapture(VIDEO_SOURCE)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls == FALL_CLASS_ID:
                now = time.time()

                if now - last_fall_time > COOLDOWN_SECONDS:
                    print(f"[FALL] Detected | confidence={conf:.2f}")
                    send_fall_event(conf)
                    last_fall_time = now

                # Draw bbox
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"FALL {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
