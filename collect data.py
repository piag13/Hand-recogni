# Collect data
import cv2
import os
import mediapipe as mp
import csv
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# create folder data
DATA_PATH = 'hand_landmark_data'
DRAWN_IMAGE_PATH = os.path.join(DATA_PATH, 'drawn_images')
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(DRAWN_IMAGE_PATH, exist_ok=True)

#create csv
csv_file = os.path.join(DATA_PATH, 'hand_landmarks.csv')
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    # Header gồm nhãn và 21 landmarks (mỗi landmark có x, y, z)
    header = ['label', 'image_name'] + [f'{i}_{axis}' for i in range(21) for axis in ['x', 'y', 'z']]
    writer.writerow(header)

#running
def collect_data(label):
    cap = cv2.VideoCapture(0)
    success, image = cap.read()
    h, w, c = image.shape
    last_time = 0
    crop_interval = 1
    image_count = 0
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Prepare black background
        black_img = np.zeros((h, w, 3), dtype=np.uint8)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        current_time = time.time()
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
              x_max = 0
              y_max = 0
              x_min = w
              y_min = h
              landmarks = []

              for lm in hand_landmarks.landmark:
                  x, y = int(lm.x * w), int(lm.y * h)
                  x_max = max(x, x_max)
                  y_max = max(y, y_max)
                  x_min = min(x, x_min)
                  y_min = min(y, y_min)
                  landmarks += [x, y]

              x_min, y_min = max(0, x_min), max(0, y_min)
              x_max, y_max = min(w, x_max), min(h, y_max)

              #draw landmarks
              mp_drawing.draw_landmarks(
                  black_img,
                  hand_landmarks,
                  mp_hands.HAND_CONNECTIONS,
                  mp_drawing_styles.get_default_hand_landmarks_style(),
                  mp_drawing_styles.get_default_hand_connections_style())

              #Crop bounding box every 1 seconds
              if (current_time - last_time) >= crop_interval:
                  cropped_img = black_img[y_min:y_max, x_min:x_max]
                  last_time = current_time

                  #save data
                  drawn_image_name = f'{label}_{image_count}.png'
                  drawn_image_path = os.path.join(DRAWN_IMAGE_PATH, drawn_image_name)
                  cv2.imwrite(drawn_image_path, cropped_img)

                  #save data to csv
                  with open(csv_file, mode='a', newline='') as f:
                      writer = csv.writer(f)
                      writer.writerow([label, drawn_image_name] + landmarks)

                  image_count += 1

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('Hand data collection', cv2.flip(black_img, 1))
        if cv2.waitKey(10) & 0xFF == ord('q'):
          break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    label = input('Nhập nhãn cho dữ liệu: ')
    collect_data(label)