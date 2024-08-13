import cv2
import mediapipe as mp
import numpy as np
import os


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
frame_count = 0 
max_images = 1000  
output_dir = 'C:/Users/HP/Desktop/whiskey/pre final/dataimage/stop'



while cap.isOpened() and frame_count < max_images:
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

   
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
   
    blank_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                blank_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    
        cv2.imwrite(os.path.join(output_dir, f'hand_landmarks_{frame_count}.png'), blank_image)
        frame_count += 1

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
