# source from https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993
from ultralytics import YOLO
import cv2
import math
import time
from threading import Thread
import importlib.util
# start webcam
#cap = cv2.VideoCapture(0)
#cap.set(3, 640)
#cap.set(4, 480)
#cap.set(3, 600)
#cap.set(4, 800)
# model
#model = YOLO("best_10Aug.pt")
#model = YOLO("best.pt")
# object classes
#classNames = ["container", "corner-cast-hole"]

model = YOLO("best_11Oct.pt")
# object classes
classNames = ["container", "corner"]
#classNames = ["hole", "plate"]


#classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat","traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella","handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]



frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream

time.sleep(1)


def holeDetection(img):
    #success, img = cap.read()
    #results = model(img, stream=True)
    #results = model.predict(img, conf = 0.30 )
    results = model(img, stream=True)
    t1 = cv2.getTickCount()


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
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            org1 = [x1-40, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
            cv2.putText(img, str(confidence), org1, font, fontScale, color, thickness)
            
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    cv2.putText(img,'FPS: {0:.2f}'.format(frame_rate_calc),(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),1,cv2.LINE_AA)
    print("FPS : ", frame_rate_calc)
            
    return img

