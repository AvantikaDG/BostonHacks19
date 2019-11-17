# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 03:16:00 2019

@author: AvantikaDG
"""

from flask import Flask, jsonify, abort
from flask import request
import requests
import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
from datetime import datetime
from twilio.rest import Client
import numpy
import os
from statistics import mode
from keras.models import load_model
import numpy as np
from mooddetection.src.utils.datasets import get_labels
from mooddetection.src.utils.inference import detect_faces
from mooddetection.src.utils.inference import draw_text
from mooddetection.src.utils.inference import draw_bounding_box
from mooddetection.src.utils.inference import apply_offsets
from mooddetection.src.utils.inference import load_detection_model
from mooddetection.src.utils.preprocessor import preprocess_input
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.oauth2 as oauth2
from bottle import route, run, request



app = Flask(__name__)

app_key='McGwemKazwF4sMtPGRLT9VjHF5ToeIzp'
ip_address='192.168.1.141'
mp3_url='http://www.accesscontrolsales.com/Ingram_Products/mp3/s3-siren.mp3'



@app.route("/play", methods=['GET'])
def play():
   data = "<?xml version=\"1.0\" ?> <key state=\"press\" sender=\"Gabbo\">PLAY</key>"
   response = requests.post('http://'+str(ip_address)+':8090/key', data=data)
   return str(response)

@app.route("/pause", methods=['GET'])
def pause():
   data = "<?xml version=\"1.0\" ?> <key state=\"press\" sender=\"Gabbo\">PAUSE</key>"
   response = requests.post('http://'+str(ip_address)+':8090/key', data=data)
   return str(response)

@app.route("/increase_volume", methods=['GET'])
def increase_volume():
   data = "<?xml version=\"1.0\" ?> <volume>30</volume>"
   response = requests.post('http://'+str(ip_address)+':8090/volume', data=data)
   return str(response)

@app.route("/decrease_volume", methods=['GET'])
def decrease_volume():
   data = "<?xml version=\"1.0\" ?> <volume>10</volume>"
   response = requests.post('http://'+str(ip_address)+':8090/volume', data=data)
   return str(response)

@app.route("/next_track", methods=['GET'])
def next_track():
   data = "<?xml version=\"1.0\" ?> <key state=\"press\" sender=\"Gabbo\">NEXT_TRACK</key>"
   response = requests.post('http://'+str(ip_address)+':8090/key', data=data)
   return str(response)

@app.route("/prev_track", methods=['GET'])
def prev_track():
   data = "<?xml version=\"1.0\" ?> <key state=\"press\" sender=\"Gabbo\">PREV_TRACK</key>"
   response = requests.post('http://'+str(ip_address)+':8090/key', data=data)
   return str(response)

@app.route("/shuffle_off", methods=['GET'])
def shuffle_off():
   data = "<?xml version=\"1.0\" ?> <key state=\"press\" sender=\"Gabbo\">SHUFFLE_OFF</key>"
   response = requests.post('http://'+str(ip_address)+':8090/key', data=data)
   return str(response)

@app.route("/shuffle_on", methods=['GET'])
def shuffle_on():
   data = "<?xml version=\"1.0\" ?> <key state=\"press\" sender=\"Gabbo\">SHUFFLE_ON</key>"
   response = requests.post('http://'+str(ip_address)+':8090/key', data=data)
   return str(response)

@app.route("/now_playing", methods=['GET'])
def now_playing():
   response = requests.get('http://'+str(ip_address)+':8090/now_playing')
   return str(response.text)

@app.route("/add_track", methods=['GET'])
def add_track():
    CLIENT_ID = "bdf58bd0c8954b8c97dd7772d9a0fb48"
    CLIENT_SECRET = "88fcfa2951394544baf2060caeb626fb"
    credentials = oauth2.SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret= CLIENT_SECRET)
    token = credentials.get_access_token()
    spotify = spotipy.Spotify(auth=token)
    #track = "coldplay yellow"
    #res = spotify.search(track, type="track", market=market, limit=1)
    #print(res)
    print("REQUEST:", request)
#    song_name = request.args.get('song')
#    artist_name = request.args.get('artist')
    
#    request.args.get('param')
    song_name = 'Numb'
    artist_name = 'Linkin Park'
    results = spotify.search(q='track:' + song_name  + ' artist:' +artist_name, type='track')
    #playlist='51GTWGAMzL5zt7oUPcxwne'
    items = results['tracks']['items']
    song_id = items[0]['id']
#    print(song_id)
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'Bearer BQCV5ifScmbadx5V0F7gjplNc3cG2dGRDZa9Gha_LUanAleNX7Y6BhdbDiGlhu97M0-BayINFdR0M0T9YK1G2HEEuN4CLieI5dKemjudlFK_7hGu0poLigfI0iYmE2XjFJkNMdym4UzJRB69wz5eg7jrhP_Pa8ZGCAYX8VPjH48TUu9i1gLVMfu4b7m70aF1eVrb1EgZ2FyYA74Dffwdv9fXtEh6wiCIOy5Hk22Jx0bJtKPWzXOx82TCBZu7Xi76wwthpEZ9F5UIwXi9ibkv6I6dJfGQCaBB1w',
    }
    params = (
        ('uris', 'spotify:track:'+str(song_id)),
    )
    response = requests.post('https://api.spotify.com/v1/playlists/6DJBPRMt1N72fSz7izSEuB/tracks', headers=headers, params=params)
    
    return str(response.text)

@app.route("/anti_break_in", methods=['GET'])
def anti_break_in():
    cascPath = "./cascades/haarcascade_frontalface_default.xml"
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
                data = '<play_info><app_key>'+str(app_key)+'</app_key><url>'+str(mp3_url)+'</url><service>service text</service><reason>reason text</reason><message>message text</message><volume>25</volume></play_info>'
                response = requests.post('http://'+str(ip_address)+':8090/speaker', data=data)                
                print("Bose API called")
            if time_in_frame_ms == twilio_delay_ms:
                # Your Account SID from twilio.com/console
                account_sid = "AC0d63a56e9fd20c654f3c6c3b050a84b1"
                # Your Auth Token from twilio.com/console
                auth_token  = "653bec204875d1159186cba9d8e6c1d6"
                
                client = Client(account_sid, auth_token)
                
                message = client.messages.create(
                    to="+16175045086", 
                    from_="+12055576032",
                    body="ALERT!!! There is an intruder in your house!")
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

    
    return str(message)

@app.route("/user_detection", methods=['GET'])
def user_detection():
    size = 4
    cascPath = "./cascades/haarcascade_frontalface_default.xml"
    datasets = os.path.join(os.getcwd(),'userdetection','images')
      
    # Part 1: Create fisherRecognizer 
    # print('Recognizing Face Please Be in sufficient Lights...') 
      
    # Create a list of images and a list of corresponding names 
    (images, lables, names, id) = ([], [], {}, 0) 
    for (subdirs, dirs, files) in os.walk(datasets): 
        for subdir in dirs: 
            names[id] = subdir 
            subjectpath = os.path.join(datasets, subdir) 
            for filename in os.listdir(subjectpath): 
                path = subjectpath + '/' + filename 
                lable = id
                images.append(cv2.imread(path, 0)) 
                lables.append(int(lable)) 
            id += 1
    width, height = 150, 120
      
    # Create a Numpy array from the two lists above 
    (images, lables) = [numpy.array(lis) for lis in [images, lables]] 
      
    # OpenCV trains a model from the images 
    # NOTE FOR OpenCV2: remove '.face' 
    model = cv2.face.LBPHFaceRecognizer_create() 
    model.train(images, lables) 
      
    # Part 2: Use fisherRecognizer on camera stream 
    face_cascade = cv2.CascadeClassifier(cascPath) 
    webcam = cv2.VideoCapture(0) 
    cnt = 0
    while cnt < 15: 
        (_, im) = webcam.read() 
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 
        faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
        if len(faces) == 0:
            cnt = 0
        for (x, y, w, h) in faces: 
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            face = gray[y:y + h, x:x + w] 
            face_resize = cv2.resize(face, (width, height)) 
            # Try to recognize the face 
            prediction = model.predict(face_resize) 
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3) 
      
            if prediction[1]<70:
               cv2.putText(im, names[prediction[0]], (x-10, y-10),  
    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
               cnt += 1
            else: 
              cv2.putText(im, 'not recognized',  
    (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) 
      
        cv2.imshow('OpenCV', im) 
          
        key = cv2.waitKey(20) 
        if key == 27: 
            break
    webcam.release()
    cv2.destroyAllWindows()
    
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'BQCV5ifScmbadx5V0F7gjplNc3cG2dGRDZa9Gha_LUanAleNX7Y6BhdbDiGlhu97M0-BayINFdR0M0T9YK1G2HEEuN4CLieI5dKemjudlFK_7hGu0poLigfI0iYmE2XjFJkNMdym4UzJRB69wz5eg7jrhP_Pa8ZGCAYX8VPjH48TUu9i1gLVMfu4b7m70aF1eVrb1EgZ2FyYA74Dffwdv9fXtEh6wiCIOy5Hk22Jx0bJtKPWzXOx82TCBZu7Xi76wwthpEZ9F5UIwXi9ibkv6I6dJfGQCaBB1w',
    }
    data = '{"device_ids":["10640d809a60bd540378d37abc213d91f52357bc"]}'
    response = requests.put('https://api.spotify.com/v1/me/player', headers=headers, data=data)
    print(response.content)
    params = (
        ('device_id', '10640d809a60bd540378d37abc213d91f52357bc'),
    )
    data = '{"context_uri":"spotify:playlist:6DJBPRMt1N72fSz7izSEuB","offset":{"position":2},"position_ms":0}'
    
    response = requests.put('https://api.spotify.com/v1/me/player/play', headers=headers, params=params, data=data)

    
    return str(response)

@app.route("/mood_detection", methods=['GET'])
def mood_detection():
   exec(open('mooddetection/src/mooddetection.py').read())
   return str('ok')
#    detection_model_path = './cascades/haarcascade_frontalface_default.xml'
#    # emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
#    emotion_model_path = './mooddetection/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
#    emotion_labels = get_labels('fer2013')
#    
#    # hyper-parameters for bounding boxes shape
#    frame_window = 10
#    emotion_offsets = (20, 40)
#    
#    # loading models
#    face_detection = load_detection_model(detection_model_path)
#    emotion_classifier = load_model(emotion_model_path, compile=False)
#    
#    # getting input model shapes for inference
#    emotion_target_size = emotion_classifier.input_shape[1:3]
#    
#    # starting lists for calculating modes
#    emotion_window = []
#    
#    # starting video streaming
#    cv2.namedWindow('window_frame')
#    video_capture = cv2.VideoCapture(0)
#    fps = 20
#    frame_delay = 1000 // fps
#    emotion_duration_ms = 1250
#    emotions = []
#    max_emotions_len = emotion_duration_ms // frame_delay
#    while len(emotions) < max_emotions_len:
#        cv2.waitKey(frame_delay)
#        bgr_image = video_capture.read()[1]
#        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
#        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
#        faces = detect_faces(face_detection, gray_image)
#    
#        for face_coordinates in faces:
#    
#            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
#            gray_face = gray_image[y1:y2, x1:x2]
#            try:
#                gray_face = cv2.resize(gray_face, (emotion_target_size))
#            except:
#                continue
#    
#            gray_face = preprocess_input(gray_face, True)
#            gray_face = np.expand_dims(gray_face, 0)
#            gray_face = np.expand_dims(gray_face, -1)
#            emotion_prediction = emotion_classifier.predict(gray_face)
#            emotion_probability = np.max(emotion_prediction)
#            emotion_label_arg = np.argmax(emotion_prediction)
#            emotion_text = emotion_labels[emotion_label_arg]
#            emotion_window.append(emotion_text)
#            emotions.append(emotion_text)
#    
#            if len(emotion_window) > frame_window:
#                emotion_window.pop(0)
#            try:
#                emotion_mode = mode(emotion_window)
#            except:
#                continue
#    
#            if emotion_text == 'angry':
#                color = emotion_probability * np.asarray((255, 0, 0))
#            elif emotion_text == 'sad':
#                color = emotion_probability * np.asarray((0, 0, 255))
#            elif emotion_text == 'happy':
#                color = emotion_probability * np.asarray((255, 255, 0))
#            elif emotion_text == 'surprise':
#                color = emotion_probability * np.asarray((0, 255, 255))
#            else:
#                color = emotion_probability * np.asarray((0, 255, 0))
#    
#            color = color.astype(int)
#            color = color.tolist()
#    
#            draw_bounding_box(face_coordinates, rgb_image, color)
#            draw_text(face_coordinates, rgb_image, emotion_mode,
#                      color, 0, -45, 1, 1)
#    
#        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
#        cv2.imshow('window_frame', bgr_image)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
#        headers = {
#            'Accept': 'application/json',
#            'Content-Type': 'application/json',
#            'Authorization': 'Bearer BQBFaEnaf6HDMRZq1DS1bHECNOrjoImHAiMV2y6UjA3PvVMbveaol9WucDo5-IE8BUy2r9zgxg8bv1eq70wI8nwhyKmV_xSEqY-pQLFIAS8WzCagZx-iKvByCj1A_pEQSactDwG6hNm2h0Q72HL06PwaR5EWIU4_8pmGU6xn6ZT9zFWLLqFDdo3qRJslOInZNe-Vu7J6XLCS1Hja6DJ9-eSdEMLkID5vAp0qAVODqIZN_7qKZ-KpbvZfIs6hacPlcWjct-C0ayM9wDm5rXhlGFCS44UZETlVCw',
#        }
#        
#        video_capture.release()
#        cv2.destroyAllWindows()
#        
#        data = '{"device_ids":["10640d809a60bd540378d37abc213d91f52357bc"]}'
#        response = requests.put('https://api.spotify.com/v1/me/player', headers=headers, data=data)
#        print(response.content)
#        params = (
#            ('device_id', '10640d809a60bd540378d37abc213d91f52357bc'),
#        )
#        final_emotion = mode(emotions)
#        if final_emotion == 'happy':
#            data = '{"context_uri":"spotify:playlist:1llkez7kiZtBeOw5UjFlJq","offset":{"position":1},"position_ms":0}'
#        elif final_emotion == 'sad':
#            data = '{"context_uri":"spotify:playlist:3m0JCCYnh27D4J13rWBfgs","offset":{"position":1},"position_ms":0}'
#        elif final_emotion == 'surprise':
#            data = '{"context_uri":"spotify:playlist:3ylCiverZxobHBTbMy1dO5","offset":{"position":1},"position_ms":0}'
#        elif final_emotion == 'angry':
#            data = '{"context_uri":"spotify:playlist:0KPEhXA3O9jHFtpd1Ix5OB","offset":{"position":1},"position_ms":0}'
#        else:
#            data = '{"context_uri":"spotify:playlist:0s609Gm5FgpZu6VfJA4I1H","offset":{"position":1},"position_ms":0}'
#
#        response = requests.put('https://api.spotify.com/v1/me/player/play', headers=headers, params=params, data=data)

#    print(f"Spotify API for emotion {mode(emotions)}")


if __name__ == "__main__":
    app.run(debug=True)