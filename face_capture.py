#!/usr/bin/env python


import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cascPath = "./xml/haarcascade_frontalface_default.xml"

count = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier(cascPath)

    faces = faceCascade.detectMultiScale(
      gray,
      scaleFactor=1.2,
      minNeighbors=3,
      minSize=(100, 100)
    )
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    print(faces)


    # Display the resulting frame

    cv2.imshow('frame',frame)

    if len(faces) <> 1:
        key = cv2.waitKey(10)
        #print("faces count is not one (%d)" % len(faces))
        continue

    key = cv2.waitKey(50)
    (x, y, w, h) = faces[0]
    if key & 0xFF == ord('q'):
        break
    if key & 0xFF == ord('s'):
        cv2.imwrite("faces/faces-%04d.png" % count, gray[y-10:(y+h), x:(x+w)])
        count += 1
    if count >= 100:
        break
    cv2.imwrite("faces/faces-%04d.png" % count, gray[y-10:(y+h), x:(x+w)])
    print("file saved %04d" % (count))
    count += 1

    '''
    cv2.imwrite("faces/faces-%04d.png" % count, gray[y:(y+h), x:(x+w)])
    count += 1
    '''
		

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

