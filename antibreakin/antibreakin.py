import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
from datetime import datetime

cascPath = "./antibreakin/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='antibreakin.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
# anterior = 0
prev_i = -1
fps = 10
frame_delay = 1000 // fps
time_in_frame_ms = 0
api_call_delay_ms = 2500
twilio_delay_ms = 5000
img_path = "./antibreakin/captures"
img_height = 150
img_width = 150

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    cv2.waitKey(frame_delay)
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    curr_i = bool(len(faces))

    if not curr_i:
        time_in_frame_ms = 0

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        time_in_frame_ms += frame_delay
        if time_in_frame_ms == api_call_delay_ms:
            print("Bose API called")
        if time_in_frame_ms == twilio_delay_ms:
            # face = gray[y:y + h, x:x + w]
            # cv2.imwrite(f'{img_path}/capture_{str(datetime.now())}.png'.replace(' ','-'), cv2.resize(face, (img_height, img_width)))
            # print(f'{img_path}/capture_{str(datetime.now())}.png'.replace(' ','-'))
            print("Twilio API called")

    # if anterior != len(faces):
    #     anterior = len(faces)
    #     log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_i = curr_i
    
    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()