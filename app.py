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
   data = "<?xml version=\"1.0\" ?> <volume>50</volume>"
   response = requests.post('http://'+str(ip_address)+':8090/volume', data=data)
   return str(response)

@app.route("/decrease_volume", methods=['GET'])
def decrease_volume():
   data = "<?xml version=\"1.0\" ?> <volume>25</volume>"
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

@app.route("/add_track", methods=['GET','POST'])
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
#    print("REQUEST:", request)
#    song_name = str(request.args.get('song'))
#    artist_name = str(request.args.get('artist'))
#    print(song_name + " " + artist_name)
#    request.args.get('param')
    #print("REQ: ", request.args.get("a"))
    song_name = 'Paradise'
    artist_name = 'Coldplay'
    results = spotify.search(q='track:' + song_name  + ' artist:' +artist_name, type='track')
    #playlist='51GTWGAMzL5zt7oUPcxwne'
    items = results['tracks']['items']
    song_id = items[0]['id']
#    print(song_id)
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': 'Bearer BQCaR_vaYaQ687FxN-j5EfvKujeH-WYR9LEsotQsSliH-X8Tm3xgRnLRA9yTqxwhHRVzSx-IwS1mQAeMs6_qqJc_ORuMb9XExtAZs6bjEuJ7lfnYV1s8AfMPwynTSsIfgRFB35q6DlbyPB6voZUeVcX-54A_hDtGEaA11RFex9yDDMBPuFHddctY6kb0Tac-O2wW0V4tigAeqbn1IvveXhh-g1eSq1Klqx43kS9aWKvA62-TzKVNC8rsAruWBn4K9FeeeM0sjTBePcbd1AHZwGFaivVy1NK5aw',
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
                data = '<play_info><app_key>'+str(app_key)+'</app_key><url>'+str(mp3_url)+'</url><service>service text</service><reason>reason text</reason><message>message text</message><volume>50</volume></play_info>'
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
        'Authorization': 'Bearer BQCaR_vaYaQ687FxN-j5EfvKujeH-WYR9LEsotQsSliH-X8Tm3xgRnLRA9yTqxwhHRVzSx-IwS1mQAeMs6_qqJc_ORuMb9XExtAZs6bjEuJ7lfnYV1s8AfMPwynTSsIfgRFB35q6DlbyPB6voZUeVcX-54A_hDtGEaA11RFex9yDDMBPuFHddctY6kb0Tac-O2wW0V4tigAeqbn1IvveXhh-g1eSq1Klqx43kS9aWKvA62-TzKVNC8rsAruWBn4K9FeeeM0sjTBePcbd1AHZwGFaivVy1NK5aw',
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


if __name__ == "__main__":
    app.run(debug=True)