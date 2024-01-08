import cv2
import numpy as np
import os
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils

mp_drawing_styles = mp.solutions.drawing_styles

mp_hands = mp.solutions.hands

def mediapipe_detection(Picture, model):
    
    Picture = cv2.cvtColor(Picture, cv2.COLOR_BGR2RGB) 

    Picture.flags.writeable = False                 
    results = model.process(Picture)                 
    Picture.flags.writeable = True                   
    Picture = cv2.cvtColor(Picture, cv2.COLOR_RGB2BGR) 
    return Picture, results

def draw_styled_landmarks(Picture, results):
    
    if results.multi_hand_landmarks:

      for hand_landmarks in results.multi_hand_landmarks:

        mp_drawing.draw_landmarks(
           Picture,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


def extract_keypoints(results):
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
        return(np.concatenate([rh]))


DATA_PATH = os.path.join('Project_Data') 

actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])

no_sequences = 30

sequence_length = 30

