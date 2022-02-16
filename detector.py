import cv2
from mtcnn import MTCNN
import recognizer

# detector = MTCNN()  # Uncomment to use MTCNN face detector
face_cascade = cv2.CascadeClassifier("L:/Anaconda3/envs/facedet/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")

def detectFaceUsingCascadeClassifier(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_boxes = face_cascade.detectMultiScale(gray, 1.05, 6)
    return face_boxes
    
def detectFaceUsingMTCNN(image):
    face_boxes = []
    detections = detector.detect_faces(image)
    for detection in detections:
        if(detection['confidence'] >= 0.9):
            face_boxes.append(detection["box"])
            
    return face_boxes
    


SOURCE = "D:/CodingAndProjects/AI Project/test_videos/anshuman.mp4"
# SOURCE = "D:/CodingAndProjects/AI Project/test_videos/monalisa.mp4"
# SOURCE = "D:/CodingAndProjects/AI Project/test_videos/kirti.mp4"
# SOURCE = "D:/CodingAndProjects/AI Project/test_videos/sattwik4.mp4"

# Starts capturing video frames from camera or video input
cap = cv2.VideoCapture(SOURCE)

while True:
    # Read the frame
    _, img = cap.read()
    if img is None:
        break
    
    img = cv2.resize(img, (100,100))
    
    
    # Detect the faces
    face_boxes = detectFaceUsingCascadeClassifier(img)
        
    # Crop the region of interest i.e. face
    if(len(face_boxes) > 0):
        x, y, w, h = face_boxes[0]
        cropped_image = img[y:y+h, x:x+w]
        
        # Recognize the class label
        class_label = recognizer.recognizeClass(cropped_image,5)
        
        # Draw the rectangle around the face
        img = cv2.rectangle(img, (x, y), ((x+w), (y+h)), (255, 0, 0), 1)
        cv2.putText(img, class_label,(x-15, y+h+10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (36,255,12), 1)
    
    # Display
    img = cv2.resize(img, (256, 256))
    cv2.imshow('Face Recognizer', img)
    
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
    
# Release video capture object
cv2.destroyAllWindows()
cap.release()
