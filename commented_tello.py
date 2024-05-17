import cv2
import mediapipe as mp
import time
import tensorflow as tf
import random
from collections import deque
# from djitellopy import tello

from utils import draw_landmarks, get_current_landmarks, predict_gesture

HOLD_TIME_THRESHOLD = 0.3
model_save_path = "./gesture_detector_final2.hdf5"
mp_hands = mp.solutions.hands

# me = tello.Tello()

# me.connect()
# print("Battery Remaining:", me.get_battery())

# while True:
#     x = input("Enter t to take-off: ")
#     if x == 't':
#         me.takeoff()
#         break


model = tf.keras.models.load_model(model_save_path)
video = cv2.VideoCapture(0)

point_history = deque(maxlen=22)

hands = mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.5)

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
            
        #    if set_gesture == "Up":
        #        point_history.append(landmarks["index_finger_tip"])
        #        draw_point_history(image, point_history)
        #        if len(point_history) >= 15:
        #            set_gesture = "Turn"
        #            point_history = deque(maxlen=22)
                       
        #    else:
        #        print("MAX LENGTH WAS", len(point_history))
        #        point_history = deque(maxlen=22)

            
    else:
        set_gesture = None
        print("Nothing Detected")
        
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 20
    angle = 30
    
    if set_gesture == 'Forward':
        print("Going forward")
        fb = speed
    
    elif set_gesture == 'Backward':
        print("Going backward")
        fb = -speed
    
    elif set_gesture == 'Up':
        print("Going up")
        ud = speed
        
    elif set_gesture == 'Down':
        print("Going down")
        ud = -speed
    
    elif set_gesture == 'Left':
        print("Going left")
        lr = -speed
        
    elif set_gesture == 'Right':
        print("Going right")
        ud = speed
        
    elif set_gesture == 'Flip':
        print("Flipping")
        # Flip
        # me.flip(random.choice(['f', 'l', 'b', 'r']))
    
    if set_gesture == 'Land':
        print("Landing")
        # break
    
    #elif set_gesture == 'Turn':
    #    print("Turning Clockwise")
        # Turn
        # me.rotate_clockwise(angle)
    
    # me.send_rc_control(lr, fb, ud,  )
    
    cv2.imshow("Camera Feed", image)
    
    if cv2.waitKey(10) & 0xFF == 27:
        break

# me.land()
    
video.release()
cv2.destroyAllWindows()