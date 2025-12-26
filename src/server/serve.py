from fastapi import FastAPI
from pydantic import BaseModel
import datetime


app = FastAPI()


class FallEvent(BaseModel):
    event: str
    confidence: float
    timestamp: float


@app.post("/fall-event")
def fall_event(data: FallEvent):
    print("🚨 FALL EVENT 🚨")
    print("Confidence:", data.confidence)
    print("Time:", datetime.datetime.fromtimestamp(data.timestamp))
    return {"status": "ok"}
