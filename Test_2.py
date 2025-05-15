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

cap = cv2.VideoCapture(0)
success, image = cap.read()
h, w, c = image.shape


def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten()
    return np.zeros(63)


with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Tính toán bounding box
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
                x_min, y_min = max(0, x_min - 20), max(0, y_min - 20)  # Thêm padding
                x_max, y_max = min(w, x_max + 20), min(h, y_max + 20)  # Thêm padding

                # Cắt và resize ảnh bàn tay
                cropped_img = image[y_min:y_max, x_min:x_max]
                if cropped_img.size > 0:
                    cropped_img = cv2.resize(cropped_img, (128, 128))

                    # Predict
                    img = np.expand_dims(cropped_img, axis=0)
                    prediction = model.predict(img, verbose=0)
                    predicted_class = int(np.argmax(prediction[0]))
                    confidence = float(prediction[0][predicted_class])

                    # Vẽ bounding box
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Hiển thị kết quả trên bounding box
                    if confidence > 0.7:  # Threshold cho độ tin cậy
                        result_text = f"{actions[predicted_class]} ({confidence:.2f})"
                        cv2.putText(image, result_text, (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Vẽ landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()