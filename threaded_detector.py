from imutils.video import WebcamVideoStream
import cv2
from mtcnn import MTCNN
import recognizer

detector = MTCNN()
face_cascade = cv2.CascadeClassifier("L:/Anaconda3/envs/facedet/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

def detectFaceUsingCascadeClassifier(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_boxes = face_cascade.detectMultiScale(gray, 1.05, 6)
    return face_boxes
    
def detectFaceUsingMTCNN(image):
    face_boxes = []
    detections = detector.detect_faces(img)
    for detection in detections:
        if(detection['confidence'] >= 0.9):
            face_boxes.append(detection["box"])
            
    return face_boxes

#SOURCE = "D:/CodingAndProjects/AI Project/VID_20220215_005004057.mp4"
# SOURCE = 0
SOURCE = "http://192.168.29.180:8080/video"

# Starts a thread for capturing video frames from camera or video input
cap = WebcamVideoStream(src=SOURCE).start()

while True:
    # Read the frame
    img = cap.read()
    img = cv2.resize(img, (128,128))
    
    # Detect the faces
    face_boxes = detectFaceUsingCascadeClassifier(img)
        
    # Crop the region of interest i.e. face
    if(len(face_boxes) > 0):
        x, y, w, h = face_boxes[0]
        cropped_image = img[y:y+h, x:x+w]
        
        # Recognize the class label
        class_label = recognizer.recognizeClass(cropped_image,5)
        
        # Draw the rectangle around the face
        img = cv2.rectangle(img, (x, y), ((x+w), (y+h)), (255, 0, 0), 2)
        cv2.putText(img, class_label,(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12), 2)
    
    # Display
    cv2.imshow('Face Recognizer', img)
    
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
    
# Release video capture object
cv2.destroyAllWindows()
cap.stop()