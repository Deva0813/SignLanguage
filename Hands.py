import cv2
import mediapipe as mp
import streamlit as st
import tensorflow as tf
import numpy as np
import google_trans_new
from google_trans_new import google_translator
import os
import glob
import time
from gtts import gTTS


# from streamlit_webrtc import( webrtc_streamer, WebRtcMode)
# import av

# Importing our custom model
model = tf.keras.models.load_model('action.h5')

# words that are trained
actions = np.array(["hi", "thanks", "i love you", "sorry", "namaste", "man", "woman", "sign",
                    "language", "good", "india", "one", "two", "three", "four", "five"])

# Holistics Model
mp_holistics = mp.solutions.holistic

# To draw Keypoints
mp_drawing = mp.solutions.drawing_utils

#  function to capture the keypoints from a video


def media_pipe_detection(image, model):
    # converting color from default opencv's BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    # Making Predictions
    results = model.process(image)
    image.flags.writeable = True
    # converting back to RGB to opencv's Default BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# function to draw landmarks
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistics.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(80, 44, 250), thickness=2, circle_radius=2)
                              )

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistics.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(80, 66, 230), thickness=2, circle_radius=2)
                              )


# defining a function to extract all the keypoints( left hand, right hand) which is in 2d arrray to 1d array.
# In case if there is no key points then we will be returning array of zeros.
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)

    return np.concatenate([lh, rh])


# to display probability in the video using opencv (Development purpose only)
# colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (117, 245, 16)]
# def prob_viz(res, actions, input_frame, colors):
#     output_frame = input_frame.copy()
#     for num, prob in enumerate(res):
#         cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
#         cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
#                     cv2.LINE_AA)
#
#     return output_frame

# to get key for the language
def get_key(val, lang):
    for key, value in lang.items():
        if val == value:
            return key

    return ""

# to remove mp3 files that is displayed


def remove_files(n):
    mp3_files = glob.glob("temp/*mp3")
    if len(mp3_files) != 0:
        now = time.time()
        n_days = n * 86400
        for f in mp3_files:
            if os.stat(f).st_mtime < now - n_days:
                os.remove(f)
                print("Deleted ", f)

# function to convert text to speech


def text_to_speech(lang, text):
    tts = gTTS(text, lang=lang, tld="com", slow=False)
    try:
        my_file_name = text[0:20]
    except:
        my_file_name = "audio"
    tts.save(f"temp/{my_file_name}.mp3")
    return my_file_name


# creating a directory to store mp3 files generated by gtts
try:
    os.mkdir("temp")
except:
    pass

# Page Header
st.title("Indian Sign Language Translator")

st.markdown("<hr/>", unsafe_allow_html=True)

# asking user permission to turn webcam on
use_webcam = st.checkbox('Use Webcam')

# storing all the languages in a list
options = list(google_trans_new.LANGUAGES.values())

# if webcam is on, do the following
if use_webcam:
    # to store the keypoints
    sequence = []

    # to store correct the predicted words by the model
    sentence = []

    # to store all the predicted words by the model
    predictions = []

    # min prediction accuracy
    threshold = 0.5

    # displaying selectbox for user to select language to translate
    display3 = st.empty()
    option = display3.selectbox(
        'Translate to',
        (options), options.index("english"), key="optionSelect")
    translator = google_translator()

    # turning webcam on
    cap = cv2.VideoCapture(0)
    # caching frame
    stframe = st.empty()

    # caching both textfields
    display = st.empty()
    display2 = st.empty()
    display3 = st.empty()
    conv = st.button("convert")

    # starting to read keypoints
    with mp_holistics.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:

            # Read feed
            ret, frame = cap.read()

            if not ret:
                continue

            # Make detections
            image, results = media_pipe_detection(frame, holistic)

            # Draw landmarks
            draw_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if max(res) > 0.65:
                    predictions.append(np.argmax(res))

                    # 3. Viz logic
                    if np.unique(predictions[-7:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:

                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 4:
                        sentence = sentence[-4:]

                    # to display text in webpage
                    if len(sentence) > 0:

                        display.text("Your Text: " + sentence[-1])
                        language = get_key(option, google_trans_new.LANGUAGES)
                        trans_text = translator.translate(
                            sentence[-1], lang_tgt=language)
                        display2.text("Translated Text: " + trans_text)
                        if conv:
                            result = text_to_speech(language, trans_text)
                            audio_file = open(f"temp/{result}.mp3", "rb")
                            audio_bytes = audio_file.read()
                            # st.markdown(f"## Your audio:")
                            display3.audio(
                                audio_bytes, format="audio/mp3", start_time=0)
                        remove_files(7)

            # Show to screen
            frame = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)

            stframe.image(image, channels='BGR', use_column_width=True)

        cap.release()

# # using streamlit webrtc
#
# #to store correct the predicted words by the model
# sentence = []
#
# # to store all the predicted words by the model
# predictions = []
#
# # to store the keypoints
# sequence = []
#
# class VideoProcessor:
#     def recv(self, frame):
#
#         global sentence
#         global predictions
#         global sequence
#
#         # min prediction accuracy
#         threshold = 0.5
#
#         # displaying selectbox for user to select language to translate
#         display3 = st.empty()
#         option = display3.selectbox(
#         'Translate to',
#         (options), options.index("english"), key="optionSelect")
#         translator = google_translator()
#
#         # caching both textfields
#         display = st.empty()
#         display2 = st.empty()
#
#         with mp_holistics.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
#         # while True:
#             frame = frame.to_ndarray(format="bgr24")
#
#             # Make detections
#             image, results = media_pipe_detection(frame, holistic)
#
#             # Draw landmarks
#             draw_landmarks(image, results)
#
#             # 2. Prediction logic
#             keypoints = extract_keypoints(results)
#             sequence.append(keypoints)
#             sequence = sequence[-30:]
#
#             if len(sequence) == 30:
#                 res = model.predict(np.expand_dims(sequence, axis=0))[0]
#                 if max(res) > 0.65:
#                     predictions.append(np.argmax(res))
#
#                     # 3. Viz logic
#                     if np.unique(predictions[-7:])[0] == np.argmax(res):
#                         if res[np.argmax(res)] > threshold:
#
#                             if len(sentence) > 0:
#                                 if actions[np.argmax(res)] != sentence[-1]:
#                                     sentence.append(actions[np.argmax(res)])
#                             else:
#                                 sentence.append(actions[np.argmax(res)])
#
#                     if len(sentence) > 4:
#                         sentence = sentence[-4:]
#                         print("=====================================================================")
#
#                     # to display text in webpage
#                     # print(sentence)
#                     if len(sentence) > 0:
#
#                         display.text("Your Text: " + sentence[-1])
#                         language = get_key(option, google_trans_new.LANGUAGES)
#                         trans_text = translator.translate(sentence[-1], lang_tgt=language)
#                         display2.text("Translated Text: " + trans_text)
#
#                     # Viz probabilities(development purpose)
#                     # image = prob_viz(res, actions, image, colors)
#
#             # Show to screen
#             frame = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)
#
#             return av.VideoFrame.from_ndarray(frame, format='bgr24')
#
# webrtc_streamer(key="translator", video_processor_factory=VideoProcessor, media_stream_constraints={"video": True, "audio": False},   async_processing=True,  mode=WebRtcMode.SENDRECV,)
