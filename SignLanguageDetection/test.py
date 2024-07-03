import cv2
import mediapipe as mp
import pickle
import numpy as np

vid = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, max_num_hands=2, min_detection_confidence=0.3)

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['modelKNN']

while True:
    ret, frame = vid.read()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
        data_ = [0]*84
        index = 0
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                x = landmark.x
                y = landmark.y
                data_[index] = x
                data_[index + 1] = y
                index += 2
                x_.append(x)
                y_.append(y)

        prediction = model.predict([np.asarray(data_)])
        predicted_character = prediction[0]
        print(predicted_character)

        cv2.putText(frame, predicted_character, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()