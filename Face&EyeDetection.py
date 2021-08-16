import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    grayscale_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(grayscale_frame, 1.4, 5)

    for (x ,y ,w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w ,y+h), (0,255,0), 5)
        
        faceArea_gray = grayscale_frame[y:y+h ,x:x+w]
        faceArea = frame[y:y+h ,x:x+w]
        
        eyes = eye_cascade.detectMultiScale(faceArea_gray, 1.4, 5)

        for (ex ,ey ,ew ,eh) in eyes:
            cv2.rectangle(faceArea ,(ex ,ey ),(ex + ew,ey + eh) ,(0,0,255) ,5)
        
    cv2.imshow('Window' , frame)

    if cv2.waitKey(1) == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()