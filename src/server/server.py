# src/server/server.py
import threading
import cv2
from fastapi import FastAPI
from pydantic import BaseModel
from src.client.fallDetection import FallDetector
from src.config.settings import Settings

class FallEvent(BaseModel):
    event: str
    confidence: float
    timestamp: float

class FallDetectionServer:
    def __init__(self, camera_id=0, host="0.0.0.0", port=8000):
        self.app = FastAPI()
        self.detector = FallDetector()
        self.camera_id = camera_id
        self.host = host
        self.port = port
        self.running = True

        # API endpoint نمونه
        @self.app.get("/status")
        def status():
            return self.detector.last_status

        @self.app.post("/fall-event")
        def receive_fall_event(event: FallEvent):
            print(f"[SERVER] Fall event received: {event}")
            return {"status": "ok"}

    def _detection_loop(self):
        cap = cv2.VideoCapture(self.camera_id)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = self.detector.process(frame)

            cv2.imshow("Fall Detection", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False
                break

        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        thread = threading.Thread(target=self._detection_loop)
        thread.start()

        import uvicorn
        uvicorn.run(self.app, host=self.host, port=self.port)
