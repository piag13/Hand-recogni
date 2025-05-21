from keras.models import load_model
import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math
import os
import tensorflow as tf

# Load model with explicit H5 format
model = tf.keras.models.load_model("model.h5", custom_objects=None, compile=False)

detector = HandDetector(maxHands=1)
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

IMG_SIZE = 128

cap = cv2.VideoCapture(0)

def predict_gesture(image):
    # Preprocess the image
    img = cv2.resize(image, (128, 128))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)

    # Make prediction
    prediction = model.predict(img, verbose=0)
    predicted_class = np.argmax(prediction[0])

    return predicted_class

while True:
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    hands, image = detector.findHands(image)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255

        x1, y1 = max(x - 20, 0), max(y - 20, 0)
        x2, y2 = x + w + 20, y + h + 20
        crop_image = image[y1:y2, x1:x2]

        if crop_image.size == 0:
            continue

        aspect_ratio = h / w
        if aspect_ratio > 1:
            scale = IMG_SIZE / h
            new_width = math.floor(scale * w)
            resized = cv2.resize(crop_image, (new_width, IMG_SIZE))
            w_gap = math.floor((IMG_SIZE - new_width) / 2)
            img_white[:, w_gap:new_width + w_gap] = resized
        else:
            scale = IMG_SIZE / w
            new_height = math.floor(scale * h)
            resized = cv2.resize(crop_image, (IMG_SIZE, new_height))
            h_gap = math.floor((IMG_SIZE - new_height) / 2)
            img_white[h_gap:new_height + h_gap, :] = resized

        # Predict and display
        prediction = predict_gesture(img_white)
        cv2.putText(image, f"Class: {classes[prediction]}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Cropped Hand", img_white)

    cv2.imshow("Webcam", image)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\n✅ Hoàn thành nhận diện cử chỉ!")
cap.release()
cv2.destroyAllWindows()