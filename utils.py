import pandas as pd
import numpy as np
import cv2

c1 = (0, 0, 0)
c2 = (255, 255, 255)

landmark_list = {
    "wrist": 0,
    "thumb_cmc": 0,
    "thumb_mcp": 0,
    "thumb_ip": 0,
    "thumb_tip": 0,
    "index_finger_mcp": 0,
    "index_finger_pip": 0,
    "index_finger_dip": 0,
    "index_finger_tip": 0,
    "middle_finger_mcp": 0,
    "middle_finger_pip": 0,
    "middle_finger_dip": 0,
    "middle_finger_tip": 0,
    "ring_finger_mcp": 0,
    "ring_finger_pip": 0,
    "ring_finger_dip": 0,
    "ring_finger_tip": 0,
    "pinky_finger_mcp": 0,
    "pinky_finger_pip": 0,
    "pinky_finger_dip": 0,
    "pinky_finger_tip": 0
}


def get_current_landmarks(image, hand_landmarks):

    landmarks = {}
    for i, key in enumerate(landmark_list.keys()):
        landmarks[key] = (int(hand_landmarks.landmark[i].x * image.shape[1]),
                          int(hand_landmarks.landmark[i].y * image.shape[0]))
    
    return landmarks

def predict_gesture(model, landmarks):
    landmarks = list(landmarks.values())

    # Make them relative to each other so that it doesnt matter if hand is on left side or right side
    landmarks = list(map(lambda l: (l[0] - landmarks[0][0], l[1] - landmarks[0][1]), landmarks))

    # Normalize them so that distance from camera doesnt matter
    getx = lambda i: abs(i[0])
    gety = lambda i: abs(i[1])

    max_x = max(list(map(getx, landmarks)))
    max_y = max(list(map(gety, landmarks)))

    # final_landmarks = list(map(lambda l: (round(l[0]/max_x, 5), round(l[1]/max_y, 5)), landmarks))
    final_landmarks = list(map(lambda l: (l[0]/max_x, l[1]/max_y), landmarks))
    
    
    arr = []
    for i in range(21):
        arr.append(final_landmarks[i][0])
        arr.append(final_landmarks[i][1])
        
    predict_result = model.predict(np.array([arr]), verbose=False)
    # print(np.squeeze(predict_result))
    result = np.argmax(np.squeeze(predict_result))
    
    dic = {
        0: "Backward",
        1: "Down",
        2: "Flip",
        3: "Forward",
        4: "Land",
        5: "Left",
        6: "Right",
        7: "Up"
    }
    
    return dic[result]



def draw_landmarks(image, handedness, landmarks, prediction=None):
    handedness = handedness.classification[0].label[0]

    # Drawing the lines
    line(image, landmarks["wrist"], landmarks["thumb_cmc"])
    line(image, landmarks["wrist"], landmarks["pinky_finger_mcp"])
    line(image, landmarks["wrist"], landmarks["index_finger_mcp"])
    line(image, landmarks["wrist"], landmarks["middle_finger_mcp"])
    line(image, landmarks["wrist"], landmarks["ring_finger_mcp"])

    line(image, landmarks["thumb_mcp"], landmarks["index_finger_mcp"])
    line(image, landmarks["middle_finger_mcp"], landmarks["index_finger_mcp"])
    line(image, landmarks["middle_finger_mcp"], landmarks["ring_finger_mcp"])
    line(image, landmarks["pinky_finger_mcp"], landmarks["ring_finger_mcp"])

    line(image, landmarks["thumb_cmc"], landmarks["thumb_mcp"])
    line(image, landmarks["thumb_ip"], landmarks["thumb_mcp"])
    line(image, landmarks["thumb_ip"], landmarks["thumb_tip"])

    line(image, landmarks["index_finger_pip"], landmarks["index_finger_mcp"])
    line(image, landmarks["index_finger_pip"], landmarks["index_finger_dip"])
    line(image, landmarks["index_finger_tip"], landmarks["index_finger_dip"])

    line(image, landmarks["middle_finger_pip"], landmarks["middle_finger_mcp"])
    line(image, landmarks["middle_finger_pip"], landmarks["middle_finger_dip"])
    line(image, landmarks["middle_finger_tip"], landmarks["middle_finger_dip"])

    line(image, landmarks["ring_finger_pip"], landmarks["ring_finger_mcp"])
    line(image, landmarks["ring_finger_pip"], landmarks["ring_finger_dip"])
    line(image, landmarks["ring_finger_tip"], landmarks["ring_finger_dip"])

    line(image, landmarks["pinky_finger_pip"], landmarks["pinky_finger_mcp"])
    line(image, landmarks["pinky_finger_pip"], landmarks["pinky_finger_dip"])
    line(image, landmarks["pinky_finger_tip"], landmarks["pinky_finger_dip"])

    # Drawing the Points
    for i in list(landmarks.values()):
        cv2.circle(image, i, radius=3, color=c2, thickness=4)

    # Drawing the square
    x_min = min([i[0] for i in list(landmarks.values())])
    x_max = max([i[0] for i in list(landmarks.values())])
    y_min = min([i[1] for i in list(landmarks.values())])
    y_max = max([i[1] for i in list(landmarks.values())])
    
    if prediction:
        cv2.putText(image, f"{handedness}: {prediction}", (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, c1, 3)
        cv2.putText(image, f"{handedness}: {prediction}", (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, c2, 1)
    else:
        cv2.putText(image, handedness, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, c1, 3)
        cv2.putText(image, handedness, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, c2, 1)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), c2, thickness=1)
    
def line(image, p1, p2):
    cv2.line(image, p1, p2, c1, 3)
    
    
def create_dataset(landmarks, label, dataset):
    landmarks = list(landmarks.values())

    # Make them relative to each other so that it doesnt matter if hand is on left side or right side
    landmarks = list(map(lambda l: (l[0] - landmarks[0][0], l[1] - landmarks[0][1]), landmarks))

    # Normalize them so that distance from camera doesnt matter
    getx = lambda i: abs(i[0])
    gety = lambda i: abs(i[1])

    max_x = max(list(map(getx, landmarks)))
    max_y = max(list(map(gety, landmarks)))

    # final_landmarks = list(map(lambda l: (round(l[0]/max_x, 5), round(l[1]/max_y, 5)), landmarks))
    final_landmarks = list(map(lambda l: (l[0]/max_x, l[1]/max_y), landmarks))

    df = pd.read_csv(dataset, index_col=0)
    columns = list(landmark_list.keys())
    
    num = df.shape[0]
    
    for i, col in enumerate(columns):
        df.loc[num, col+'_x'] = final_landmarks[i][0]
        df.loc[num, col+'_y'] = final_landmarks[i][1]
        df.loc[num, 'label'] = label

    df.to_csv(dataset)
    print(num)


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv2.circle(image, (point[0], point[1]), 1 + int(index/4), (152, 251, 152), 3)
