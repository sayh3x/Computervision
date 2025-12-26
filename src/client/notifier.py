import requests
import time

def send_fall_event(api_url, confidence):
    payload = {
        "event": "fall_detected",
        "confidence": float(confidence),
        "timestamp": time.time()
    }
    try:
        r = requests.post(api_url, json=payload, timeout=2)
        print("[API] sent", r.status_code)
    except Exception as e:
        print("[API ERROR]", e)
