import cv2
import threading
from Test_2_endoscope_YOLOV8 import *  # Make sure your holeDetection of corner casting function is correctly imported

class camThread(threading.Thread):
    def __init__(self, camID):
        threading.Thread.__init__(self)
        self.camID = camID
        self.frame = None
        self.running = True
        self.output = None

    def run(self):
        cam = cv2.VideoCapture(self.camID)
        while self.running:
            success, self.frame = cam.read()  # Read the frame
            if not success:
                break
            
            # Process the frame and get the output
            self.output = holeDetection (self.frame)  # Pass the frame to the detection function


        cam.release()  # Release the camera when done

    def stop(self):
        self.running = False

def main():
    cv2.namedWindow("Camera 1")
    cv2.namedWindow("Camera 2")

    thread1 = camThread(0)
    thread2 = camThread(2)
    thread1.start()
    thread2.start()

    while True:
        if thread1.output is not None:
            cv2.imshow("Camera 1", thread1.output)
        if thread2.output is not None:
            cv2.imshow("Camera 2", thread2.output)

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            thread1.stop()
            thread2.stop()
            break

    thread1.join()
    thread2.join()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

