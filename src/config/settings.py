from pathlib import Path


# ========= PATHS =========
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "best.pt"


# ========= INFERENCE =========
VIDEO_SOURCE = 0 # 0 webcam | path | rtsp
CONF_THRESHOLD = 0.6
FALL_CLASS_ID = 0
COOLDOWN_SECONDS = 10


# ========= API =========
API_HOST = "127.0.0.1"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}/fall-event"
MODEL_PATH = "models/best.pt" 
VIDEO_SOURCE = "input.mp4" # 0 for webcam 
FALL_CLASS_ID = 0 
CONF_THRESHOLD = 0.6 
API_URL = "http://127.0.0.1:8000/fall-event" 
COOLDOWN_SECONDS = 10

