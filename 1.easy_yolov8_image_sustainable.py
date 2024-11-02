from ultralytics import YOLO
import cv2
import time
# Load the pre-trained model
#model = YOLO('yolov8n.pt')  # or another model variant
#model = YOLO('best_10Aug.pt')  # or another model variant
model = YOLO('best_11Oct.pt')  # best result
#model = YOLO('best_why.pt') #worst result

img = "/home/pi/Use_Me_rpi-bookworm-yolov8-main/Image/container7.jpg"
results = model.predict([img], conf = 0.29)
#results = model.predict(["27cm.png"], conf = 0.1)

prev_frame_time = 0
new_frame_time = 0


for result in results:
    boxes = result.boxes
    result.show()   
    
	# font which we will be using to display FPS 
font = cv2.FONT_HERSHEY_SIMPLEX 
new_frame_time = time.time() 
fps = 1/(new_frame_time-prev_frame_time) 
prev_frame_time = new_frame_time 
fps = str(int(fps))
#cv2.putText(img, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 






