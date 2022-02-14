import cv2
from mtcnn import MTCNN
import recognizer


detector = MTCNN()
#face_cascade = cv2.CascadeClassifier("L:/Anaconda3/envs/facedet/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()
    
    detections = detector.detect_faces(img)
    # Detect the faces
    face_boxes = []
    for detection in detections:
        if(detection['confidence'] >= 0.95):
            face_boxes.append(detection["box"])
        
    # Crop the region of interest i.e. face
    if(len(face_boxes) > 0):
        x, y, w, h = face_boxes[0]
        cropped_image = img[y:y+h, x:x+w]
        
        # Recognize the class label
        class_label = recognizer.recognizeClass(cropped_image,5)
    
        
        # Draw the rectangle around the face
        img = cv2.rectangle(img, (x, y), ((x+w), (y+h)), (255, 0, 0), 3)
        cv2.putText(img, class_label,(x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    
    # Display
    cv2.imshow('Video', img)
    
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
    
# Release video capture object
cap.release()