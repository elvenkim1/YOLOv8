from ultralytics import YOLO
import math
model = YOLO('best.pt')  # best result
#classNames = ["roof", "shipping-container", 'side-corner-cast', 'top-corner-cast']

img = "/home/rpic4b/Rpi5_yolov8/image/dawn29.jpg"  #modify the path according to image location
results = model.predict([img], conf = 0.25)

#to show all bounding boxes
for result in results:
    boxes = result.boxes
    result.show()
    
    for box in boxes:
        confidence = math.ceil((box.conf[0]*100))/100
        print(confidence)
    






