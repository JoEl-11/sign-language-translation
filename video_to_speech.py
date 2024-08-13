import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os
from collections import Counter
from moviepy.video.io.VideoFileClip import VideoFileClip
from gtts import gTTS
from spellchecker import SpellChecker
from moviepy.editor import VideoFileClip

def correct_and_convert_text_to_speech(text, filename="output.mp3"):
    spell = SpellChecker()
    corrected_text = []
    
    words = text.split()
    misspelled = spell.unknown(words)
    
    for word in words:
        correction = spell.correction(word) if word in misspelled else word
        corrected_text.append(correction if correction else word)
    
    corrected_text = ' '.join(corrected_text)
    print(f"Corrected Text: {corrected_text}")
    
    tts = gTTS(text=corrected_text, lang='en')
    tts.save(filename)

def save_preprocessed_image(image, filename):
    folder = r'media'
    if not os.path.exists(folder):
        os.makedirs(folder)
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, image)


def split_video(video_path, clip_duration=5, output_dir='video'):
    video = VideoFileClip(video_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_duration = video.duration
    num_clips = int(total_duration // clip_duration)
    
    '''if total_duration % clip_duration != 0:
        num_clips += 1'''
    
    #splitting according to clip duration
    for i in range(num_clips):
        start_time = i * clip_duration
        end_time = min((i + 1) * clip_duration, total_duration)
        
        
        subclip = video.subclip(start_time, end_time)
        
        # saving each clip to video folder
        subclip_filename = os.path.join(output_dir, f"clip_{i+1}.mp4")
        subclip.write_videofile(subclip_filename, codec="libx264")

model=load_model(r"models/rizwinmodel3.keras")
frame_count=0
class_names=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','skip','space','t','u','v','w','x','y','z']  

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

split_video(r"pred.mp4")

# video folder
video_folder = r"video"
result_string = ""

def preprocess_image(image, target_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype("float") / 255.0
    return image


for video_file in os.listdir(video_folder):
    if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv','webm')):
        video_path = os.path.join(video_folder, video_file)
        print(f"Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        predictions = []
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success or frame is None:
                break

            # BGR image to RGB
            rgb_frame = cv2.cvtColor(cv2.flip(frame,1), cv2.COLOR_BGR2RGB)

            # detect hand
            rgb_frame.flags.writeable = False
            result = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # initialize the black_image ONCE per frame
            black_image = np.zeros((rgb_frame.shape[0], rgb_frame.shape[1], 3), dtype=np.uint8)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    frame_copy = frame.copy()
                    mp_drawing.draw_landmarks(
                        black_image, 
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)
                    )
            save_preprocessed_image(black_image, f"frame_{frame_count}.png")
            frame_count += 1
        
            processed_image = preprocess_image(black_image, target_size=(128, 128))
            prediction = model.predict(processed_image)
            class_id = np.argmax(prediction)
            predicted_class = class_names[class_id]
            predictions.append(predicted_class)

        cap.release()
        if predictions:
            most_common_class = Counter(predictions).most_common(1)[0][0]
            if most_common_class == 'space':
                result_string += "  "
            else:
                result_string += most_common_class

print("\nFinal results:\n", result_string)

correct_and_convert_text_to_speech(result_string)
