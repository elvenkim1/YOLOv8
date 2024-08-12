# source from https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993
from ultralytics import YOLO
import cv2
import math
import time
from threading import Thread
import importlib.util
# start webcam
cap = cv2.VideoCapture(0)
#cap.set(3, 640)
#cap.set(4, 480)

# model
model = YOLO("best_10Aug.pt")
#model = YOLO("best.pt")
# object classes
#classNames = ["container", "corner-cast-hole"]
classNames = ["container", "corner"]
#classNames = ["corner-cast-hole", "container"]
#classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat","traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella","handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
#validation_results = model.val(data="coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
#validation_results = model.val(data="coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6)

prev_frame_time = 0
new_frame_time = 0
font = cv2.FONT_HERSHEY_SIMPLEX 

while(cap.isOpened()): 
    success, img = cap.read()
    #results = model(img, stream=True)
    results = model.predict(img, conf = 0.20 )
    new_frame_time = time.time() 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    fps = str(int(fps))
    cv2.putText(img, fps, (7, 70), font, 2, (100, 255, 0), 2, cv2.LINE_AA)


    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
     #       print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
      #      print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            org1 = [x1-60, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.8
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            cv2.putText(img, str(confidence*100), org1, font, fontScale, color, thickness)
            
  
    #cv2.putText(img,'FPS: {0:.2f}'.format(fps),(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),1,cv2.LINE_AA)
    #print("FPS : ", frame_rate_calc)
            
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break
 

cap.release()
cv2.destroyAllWindows()