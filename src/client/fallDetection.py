# src/client/fallDetection.py
import time
from ultralytics import YOLO
import cv2
from src.config.settings import MODEL_PATH, CONF_THRESHOLD, FALL_CLASS_ID, COOLDOWN_SECONDS, API_URL
from src.client.notifier import send_fall_event
from src.utils.cooldown import Cooldown

class FallDetector:
    def __init__(self):
        self.model = YOLO(str(MODEL_PATH))
        self.cooldown = Cooldown(COOLDOWN_SECONDS)
        self.last_status = {"fall": False, "timestamp": None}

    def process(self, frame):
        results = self.model(frame, conf=CONF_THRESHOLD, verbose=False)
        fall_detected = False

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # ---------- FALL LOGIC ----------
                if cls == FALL_CLASS_ID and self.cooldown.ready():
                    print(f"[FALL] Detected | confidence={conf:.2f}")
                    send_fall_event(API_URL, conf)
                    self.last_status = {"fall": True, "timestamp": time.time()}
                    fall_detected = True

                # ---------- DRAW BBOX ----------
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255) if cls == FALL_CLASS_ID else (0, 255, 0)
                label = "FALL" if cls == FALL_CLASS_ID else "NORMAL"

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

        if not fall_detected:
            self.last_status = {"fall": False, "timestamp": None}

        return frame
