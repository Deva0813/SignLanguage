import cv2
import mediapipe as mp
import streamlit as st
import tensorflow as tf
import numpy as np
# from translate import Translator

model = tf.keras.models.load_model('action.h5')

actions = np.array(['hello', 'thanks', 'iloveyou'])

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_styled_landmarks(image, results):

    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):

    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

# to get key for the language


# def translate(txt, lang):
#     tr = Translator(to_lang=lang)
#     txt = tr.translate(txt)
#     return txt


st.title("Sing Language Translator")

st.markdown("<hr/>", unsafe_allow_html=True)

use_webcam = st.checkbox('Use Webcam')

op = ["Tamil", "Hindi", "Malayalam", "Spanish",
      "French", "Arabic", "Bengali", "Russian"]

if use_webcam:

    sequence = []
    sentence = ''
    threshold = 0.6

    dis = st.empty()
    # option = dis.selectbox('Translate to : ', op)

    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    display = st.empty()
    display1 = st.empty()
    # conv = st.button("Convert")

    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            # print(results)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
    #         sequence.insert(0,keypoints)
    #         sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])

            # 3. Viz logic
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence = actions[np.argmax(res)]
                    else:
                        sentence = actions[np.argmax(res)]

                # if len(sentence) > 5:
                #     sentence = sentence[-5:]

                if len(sentence) > 0:
                    display.subheader(" Your Text : " + sentence)

                # if conv:
                #     display1.subheader(
                #         "The Translation : " + translate(sentence, option))
                # Viz probabilities
                # image = prob_viz(res, actions, image, colors)

            # cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            # cv2.putText(image, sentence, (270, 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
        #     cv2.imshow('OpenCV Feed', image)

        #     # Break gracefully
        #     if cv2.waitKey(10) & 0xFF == ord('q'):
        #         break
        # cap.release()
        # cv2.destroyAllWindows()
            frame = cv2.resize(image, (0, 0), fx=0.8, fy=0.8)
            stframe.image(image, channels="BGR", use_column_width=True)
