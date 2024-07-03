import os
import cv2
import keyboard
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PATH = "./images"

if not os.path.exists(PATH):
    os.makedirs(PATH)

vid = cv2.VideoCapture(0)

letters_srb = ['A', 'B', 'V', 'G', 'D', 'Đ', 'E', 'Ž', 'Z', 'I', 'J', 'K', 'L', 'M', 'N', 'NJ', 'O', 'P', 'R', 'S', 'T', 'Ć', 'U', 'F', 'H', 'C', 'Č', 'DŽ', 'Š'] 
letters = ['A', 'B', 'V', 'G', 'D', 'DJ', 'E', 'ZZ', 'Z', 'I', 'J', 'K', 'L', 'M', 'N', 'NJ', 'O', 'P', 'R', 'S', 'T', 'TJ', 'U', 'F', 'H', 'C', 'CC', 'DZ', 'SS'] 
dataset_size = 100

font_path = 'font.ttf'
font = ImageFont.truetype(font_path, 32)

for i in range(len(letters)):
    letter_srb = letters_srb[i]
    letter = letters[i]

    if not os.path.exists(os.path.join(PATH, str(letter))):
        os.makedirs(os.path.join(PATH, str(letter)))

    while(True):
        ret, frame = vid.read()
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        draw.text((50, 50), "Collecting data for letter "+str(letter_srb), font=font, fill=(255, 255, 0, 255))
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.putText(frame,
                    'When you are ready, press Q!',
                    (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1,
                    (0, 255, 255), 2, cv2.LINE_4)
        
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):
            break

    for i in range(dataset_size):
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(PATH, str(letter), str(i) + ".jpg" ), frame)


vid.release()

cv2.destroyAllWindows()
