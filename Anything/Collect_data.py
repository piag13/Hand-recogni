import os
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

DATA_DIR = '../data/Dataset'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 10
dataset_size = 200

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"Collecting data for class {j}")
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Press "Q" to start collection for class {}.'.format(j), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    counter = 0
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while counter < dataset_size:
            ret, frame = cap.read()
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_max, y_max = 0, 0
                    x_min, y_min = w, h

                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_max = max(x, x_max)
                        y_max = max(y, y_max)
                        x_min = min(x, x_min)
                        y_min = min(y, y_min)

                    padding = 20
                    x_min, y_min = max(0, x_min - padding), max(0, y_min - padding)
                    x_max, y_max = min(w, x_max + padding), min(h, y_max + padding)

                    cropped_img = frame[y_min:y_max, x_min:x_max]
                    black_img = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

                    # Draw landmarks on black background
                    mp_drawing.draw_landmarks(
                        black_img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # Resize to 224x224
                    black_img = cv2.resize(black_img, (128, 128))

                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Save the image
                    file_path = os.path.join(class_dir, f'{counter}.jpg')
                    cv2.imwrite(file_path, cropped_img)

                    cv2.imshow('black_img', cropped_img)

                    print(f"Saved {file_path}")
                    counter += 1

            cv2.imshow('Hand Data Collection', frame)
            if cv2.waitKey(10) & 0xFF == ord('t'):
                break

cap.release()
cv2.destroyAllWindows()
