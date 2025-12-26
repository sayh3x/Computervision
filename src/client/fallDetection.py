import cv2
import time
from ultralytics import YOLO

from src.config.settings import (
    MODEL_PATH,
    VIDEO_SOURCE,
    CONF_THRESHOLD,
    FALL_CLASS_ID,
    COOLDOWN_SECONDS,
    API_URL,
)

from src.client.notifier import send_fall_event
from src.utils.cooldown import Cooldown


# ================= INIT =================
model = YOLO(str(MODEL_PATH))
cooldown = Cooldown(COOLDOWN_SECONDS)

cap = cv2.VideoCapture(VIDEO_SOURCE)
# =======================================


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

            # ---------- FALL LOGIC ----------
            if cls == FALL_CLASS_ID and cooldown.ready():
                print(f"[FALL] Detected | confidence={conf:.2f}")
                send_fall_event(API_URL, conf)

            # ---------- DRAW ----------
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = "FALL" if cls == FALL_CLASS_ID else "NORMAL"
            color = (0, 0, 255) if cls == FALL_CLASS_ID else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
