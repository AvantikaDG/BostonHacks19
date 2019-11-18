from statistics import mode
import requests
import cv2
from keras.models import load_model
import numpy as np

from mooddetection.src.utils.datasets import get_labels
from mooddetection.src.utils.inference import detect_faces
from mooddetection.src.utils.inference import draw_text
from mooddetection.src.utils.inference import draw_bounding_box
from mooddetection.src.utils.inference import apply_offsets
from mooddetection.src.utils.inference import load_detection_model
from mooddetection.src.utils.preprocessor import preprocess_input

# parameters for loading data and images
# detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
detection_model_path = './cascades/haarcascade_frontalface_default.xml'
# emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_model_path = './mooddetection/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
fps = 20
frame_delay = 1000 // fps
emotion_duration_ms = 1250
emotions = []
max_emotions_len = 2000 // frame_delay
while len(emotions) < max_emotions_len:
    cv2.waitKey(frame_delay)
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)
        emotions.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
final_emotion = mode(emotions)
print(f"Spotify API for emotion {final_emotion}")
video_capture.release()
cv2.destroyAllWindows()


headers = {
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Authorization': 'Bearer BQCaR_vaYaQ687FxN-j5EfvKujeH-WYR9LEsotQsSliH-X8Tm3xgRnLRA9yTqxwhHRVzSx-IwS1mQAeMs6_qqJc_ORuMb9XExtAZs6bjEuJ7lfnYV1s8AfMPwynTSsIfgRFB35q6DlbyPB6voZUeVcX-54A_hDtGEaA11RFex9yDDMBPuFHddctY6kb0Tac-O2wW0V4tigAeqbn1IvveXhh-g1eSq1Klqx43kS9aWKvA62-TzKVNC8rsAruWBn4K9FeeeM0sjTBePcbd1AHZwGFaivVy1NK5aw'
}


data = '{"device_ids":["10640d809a60bd540378d37abc213d91f52357bc"]}'
response = requests.put('https://api.spotify.com/v1/me/player', headers=headers, data=data)
print(response.content)
params = (
    ('device_id', '10640d809a60bd540378d37abc213d91f52357bc'),
)

if final_emotion == 'happy':
    data = '{"context_uri":"spotify:playlist:1llkez7kiZtBeOw5UjFlJq","offset":{"position":1},"position_ms":0}'
elif final_emotion == 'sad':
    data = '{"context_uri":"spotify:playlist:3m0JCCYnh27D4J13rWBfgs","offset":{"position":1},"position_ms":0}'
elif final_emotion == 'surprise':
    data = '{"context_uri":"spotify:playlist:3ylCiverZxobHBTbMy1dO5","offset":{"position":1},"position_ms":0}'
elif final_emotion == 'angry':
    data = '{"context_uri":"spotify:playlist:0KPEhXA3O9jHFtpd1Ix5OB","offset":{"position":1},"position_ms":0}'
else:
    data = '{"context_uri":"spotify:playlist:0s609Gm5FgpZu6VfJA4I1H","offset":{"position":1},"position_ms":0}'

response = requests.put('https://api.spotify.com/v1/me/player/play', headers=headers, params=params, data=data)
