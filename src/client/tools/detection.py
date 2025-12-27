from ultralytics import YOLO
from src.config.settings import MODEL_PATH


model = YOLO(str(MODEL_PATH))

results = model('not fallen031.jpg')

for result in results:
    result.show() 
    
    result.save(filename='output_pose2.jpg')
