# For webcam input:
import cv2
import mediapipe as mp
import numpy as np
import time
from tensorflow.keras.models import load_model
import tensorflow as tf

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Load the trained model with the properly defined focal_loss
model = load_model('model.h5')
actions = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
sequence_length = 30
frames_buffer = []
results_list = []

cap = cv2.VideoCapture(0)
success, image = cap.read()
h, w, c = image.shape

def predict_gesture(image):
    # Preprocess the image
    img = cv2.resize(image, (224, 224))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]

    return predicted_class, confidence

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate bounding box
                x_max = 0
                y_max = 0
                x_min = w
                y_min = h
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_max = max(x, x_max)
                    y_max = max(y, y_max)
                    x_min = min(x, x_min)
                    y_min = min(y, y_min)

                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)

                # Draw bounding box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Process and predict
                cropped_img = image[y_min:y_max, x_min:x_max]
                if cropped_img.size > 0:
                    predicted_class, confidence = predict_gesture(cropped_img)
                    
                    if confidence > 0.7:  # Only show high confidence predictions
                        # Draw prediction above the bounding box
                        result_text = f"{actions[predicted_class]} ({confidence:.2f})"
                        # Calculate text position above bounding box
                        text_x = x_min
                        text_y = y_min - 10  # 10 pixels above the box
                        
                        # Make sure text doesn't go off screen
                        if text_y < 20:
                            text_y = y_min + 30  # Put text inside box if too close to top
                            
                        # Add background rectangle for text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        thickness = 2
                        (text_width, text_height), _ = cv2.getTextSize(result_text, font, font_scale, thickness)
                        cv2.rectangle(image, (text_x, text_y - text_height - 5),
                                    (text_x + text_width, text_y + 5), (0, 255, 0), -1)
                        
                        # Draw text
                        cv2.putText(image, result_text, (text_x, text_y),
                                  font, font_scale, (0, 0, 0), thickness)

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Show the frame (already flipped)
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()