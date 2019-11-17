import cv2, sys, numpy, os 
cascPath = "./cascades/haarcascade_frontalface_default.xml"

# All the faces data will be 
#  present this folder 
# datasets = './userdetection/images'  
  
# These are sub data sets of folder,  
# for my faces I've used my name you can  
# change the label here 
# sub_data = '.user_images'     

path = os.path.join(os.getcwd(),'userdetection','images','user')
# path = './userdetection/images/user_images'
if not os.path.isdir(path):
    os.makedirs(path)
  
# defining the size of images  
width = 150
height = 120
  
#'0' is used for my webcam,  
# if you've any other camera 
#  attached use '1' like this 
face_cascade = cv2.CascadeClassifier(cascPath) 
webcam = cv2.VideoCapture(0)  
  
# The program loops until it has 30 images of the face. 
count = 1
while count < 1000:  
    (_, im) = webcam.read() 
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray, 1.3, 4) 
    for (x, y, w, h) in faces: 
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
        face = gray[y:y + h, x:x + w] 
        face_resize = cv2.resize(face, (width, height)) 
        cv2.imwrite('% s/% s.png' % (path, count), face_resize) 
    count += 1
      
    cv2.imshow('OpenCV', im) 
    key = cv2.waitKey(10) 
    if key == 27: 
        break