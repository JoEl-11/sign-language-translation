from flask import Flask, render_template, request, jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Load the pre-trained model
model = load_model(r'model128.keras')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Class names for the model predictions
class_names = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['zz']

def preprocess_image(image, target_size):
    # Convert image to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image
    image = cv2.resize(image, target_size)
    # Convert to array and normalize
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image.astype("float") / 255.0
    return image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    # Get the image data from the form
    data = request.form['image']
    image_data = data.split(',')[1]  # Remove the data URL scheme part
    image = Image.open(BytesIO(base64.b64decode(image_data)))

    # Convert PIL image to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create a black background image
    black_image = np.zeros_like(image)

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    result = hands.process(rgb_image)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the black background image
            mp_drawing.draw_landmarks(
                black_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2))

        # Preprocess the black image for model prediction
        processed_image = preprocess_image(black_image, target_size=(128, 128))

        # Predict the class of the hand gesture
        prediction = model.predict(processed_image)
        class_id = np.argmax(prediction)
        confidence = np.max(prediction)
        class_name = class_names[class_id]
    else:
        class_name = 'No hand detected'
        confidence = 0

    # Save the original image
    original_image_path = os.path.join(r'static\resources', 'uploaded_image.png')
    cv2.imwrite(original_image_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Save the black image with landmarks
    landmarks_image_path = os.path.join(r'static\resources', 'black_image_with_landmarks.png')
    cv2.imwrite(landmarks_image_path, cv2.cvtColor(black_image, cv2.COLOR_BGR2RGB))

    # Save the preprocessed image
    preprocessed_image = preprocess_image(black_image, target_size=(128, 128))
    preprocessed_image_path = os.path.join(r'static\resources', 'preprocessed_image.png')
    preprocessed_image = (preprocessed_image.squeeze() * 255).astype(np.uint8)  # Convert back to 0-255 range for saving
    cv2.imwrite(preprocessed_image_path, preprocessed_image)

    return jsonify({"class": class_name, "confidence": float(confidence)})

if __name__ == '__main__':
    app.run(debug=True)
