from ultralytics import YOLO


model = YOLO('yolov8n-pose.pt')

results = model('')

for result in results:
    result.show() 
    
    result.save(filename='output_pose.jpg')
