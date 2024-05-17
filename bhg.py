import cv2
import mediapipe as mp
import time
import tensorflow as tf
import random
from collections import deque

from utils import draw_landmarks, get_current_landmarks, predict_gesture, draw_point_history

HOLD_TIME_THRESHOLD = 0.3
model_save_path = "./gesture_detector_final2.hdf5"
mp_hands = mp.solutions.hands

model = tf.keras.models.load_model(model_save_path)
video = cv2.VideoCapture(0)

point_history = deque(maxlen=22)

hands = mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.8, min_tracking_confidence=0.5)

first_run = True
set_gesture = None

index = 0
while video.isOpened():
    success, image = video.read()
    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if not success:
        continue

    results = hands.process(image)
    image.flags.writeable = True

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

            landmarks = get_current_landmarks(image, hand_landmarks)

            current_gesture = predict_gesture(model, landmarks)
            draw_landmarks(image, handedness, landmarks, set_gesture)
            if first_run:
                set_gesture = current_gesture
                hold_start_time = time.time()
                prev_gesture = current_gesture
                first_run = False
                
            if current_gesture != prev_gesture:
                hold_start_time = time.time()
                prev_gesture = current_gesture
                
            if time.time() - hold_start_time > HOLD_TIME_THRESHOLD:
                set_gesture = current_gesture
            
            if set_gesture != current_gesture:
                print(f"Next Change: Hold {current_gesture} for {round(HOLD_TIME_THRESHOLD - (time.time() - hold_start_time), 2)}s")
            print(f"label: {set_gesture}\n")
            
            if set_gesture == "Up":
                point_history.append(landmarks["index_finger_tip"])
                draw_point_history(image, point_history)
                if len(point_history) >= 15:
                    set_gesture = "Turn"
                    point_history = deque(maxlen=22)
                        
            else:
                print("MAX LENGTH WAS", len(point_history))
                point_history = deque(maxlen=22)

            
    else:
        set_gesture = None
        print("Nothing Detected")
    
    cv2.imshow("Camera Feed", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if cv2.waitKey(10) & 0xFF == 27:
        break
    
video.release()
cv2.destroyAllWindows()