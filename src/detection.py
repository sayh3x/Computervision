from ultralytics import YOLO


model = YOLO('best.pt')

results = model('not fallen031.jpg')

for result in results:
    result.show() 
    
    result.save(filename='output_pose2.jpg')
