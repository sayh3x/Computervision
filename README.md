# Computervision
A project for fall Detection in elevator

pip install -r requirements.txt


# Run server
uvicorn server.server:app --host 0.0.0.0 --port 8000


# Run inference
python inference/fall_detector.py
