import numpy as np
import cv2
cam=cv2.VideoCapture(0)
face= cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    r,vid=cam.read()
    height,width=vid.shape[:2]
    gray=cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces=face.detectMultiScale(gray, minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    for a,b,x,y in faces:
        cv2.rectangle(vid, (x,y), (a+x, b+y), (148, 69, 17), 6)
        cv2.imshow('face-found!!',vid)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
