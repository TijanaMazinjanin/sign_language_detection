import mediapipe as mp
import os
import cv2
import pickle
import numpy as np

IMAGES_DIR = "./images"

data = []
signes = []

def translate_hand(data, dx, dy):
    translated_data = data.copy()
    for i in range(0, len(translated_data), 2):
        if translated_data[i] != 0:
            translated_data[i] = min(max(translated_data[i] + dx, 0), 1)
            translated_data[i + 1] = min(max(translated_data[i + 1] + dy, 0), 1)
    return translated_data

def scale_hand(data, scale):
    scaled_data = data.copy()
    center_x = np.mean(scaled_data[0::2])
    center_y = np.mean(scaled_data[1::2])
    for i in range(0, len(scaled_data), 2):
        if scaled_data[i] != 0:
            scaled_data[i] = min(max(center_x + (scaled_data[i] - center_x) * scale, 0), 1)
            scaled_data[i + 1] = min(max(center_y + (scaled_data[i + 1] - center_y) * scale, 0), 1)
    return scaled_data


def rotate_hand(data, angle):
    radians = np.deg2rad(angle)
    cos_angle = np.cos(radians)
    sin_angle = np.sin(radians)
    rotated_data = data.copy()
    center_x = np.mean(rotated_data[0::2])
    center_y = np.mean(rotated_data[1::2])

    for i in range(0, len(rotated_data), 2):
        if rotated_data[i] != 0:
            x = rotated_data[i] - center_x
            y = rotated_data[i + 1] - center_y
            rotated_data[i] = min(max(center_x + x * cos_angle - y * sin_angle, 0), 1)
            rotated_data[i + 1] = min(max(center_y + x * sin_angle + y * cos_angle, 0), 1)

    return rotated_data


def create_dataset():
    mp_hands = mp.solutions.hands
    with mp_hands.Hands( static_image_mode = True, max_num_hands=2, min_detection_confidence=0.3) as hands:
        for sign_dir in os.listdir(IMAGES_DIR):
            print(sign_dir)
            for sign_img in os.listdir(os.path.join(IMAGES_DIR, sign_dir)):
                print(sign_img)
                img_path = os.path.join(IMAGES_DIR, sign_dir)
                img_path = os.path.join(img_path, sign_img)

                data_ = [0]*84
                index = 0
                
                image = cv2.imread(img_path)
                
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not results.multi_hand_landmarks:
                    print("NO HAND LANDMARKS")
                    continue

                image_height, image_width, _ = image.shape
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        x = landmark.x
                        y = landmark.y

                        data_[index] = x
                        data_[index + 1] = y
                        index += 2


                data.append(data_)
                signes.append(sign_dir)

                #expand dataset

                translations = [
                    (0.05, 0.05), (-0.05, -0.05), (0.05, -0.05), (-0.05, 0.05),
                    (0.1, 0.1), (-0.1, -0.1), (0.1, -0.1), (-0.1, 0.1),
                    (0.2, 0.2), (-0.2, -0.2), (0.2, -0.2), (-0.2, 0.2),
                    (0.3, 0.3), (-0.3, -0.3), (0.3, -0.3), (-0.3, 0.3),
                    (0.4, 0.4), (-0.4, -0.4), (0.4, -0.4), (-0.4, 0.4)
                ]
                scales = [0.9, 1.1]
                rotations = [5, 10, 15, -5, -10, -15]

                for dx, dy in translations:
                    translated_data = translate_hand(data_, dx, dy)
                    data.append(translated_data)
                    signes.append(sign_dir)

                for scale in scales:
                    scaled_data = scale_hand(data_, scale)
                    data.append(scaled_data)
                    signes.append(sign_dir)

                for angle in rotations:
                    rotated_data = rotate_hand(data_, angle)
                    data.append(rotated_data)
                    signes.append(sign_dir)
        
    
    file = open('data.pickle', 'wb')
    pickle.dump({'data':data, 'signes':signes}, file)
    file.close()

create_dataset()
                