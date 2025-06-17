import cv2
import cvzone.HandTrackingModule as htm
import numpy as np
import math
import os
import time

cap = cv2.VideoCapture(0)
detector = htm.HandDetector(maxHands=1)

IMG_SIZE = 240
DATA_DIR = 'data/Dataset'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 10
dataset_size = 600

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"\n=== Collecting data for class {j} ===")
    print("Press 's' to start saving images.")

    counter = 0
    start_collection = False

    while True:
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)  # Flip camera for natural view
        hands, image = detector.findHands(image)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            Img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255  # White background

            # Crop with margin
            x1, y1 = max(x - 20, 0), max(y - 20, 0)
            x2, y2 = x + w + 20, y + h + 20
            crop_image = image[y1:y2, x1:x2]

            if crop_image.size == 0:
                continue

            aspectRatio = h / w
            if aspectRatio > 1:
                k = IMG_SIZE / h
                wCal = math.floor(k * w)
                resized_image = cv2.resize(crop_image, (wCal, IMG_SIZE))
                wGap = math.floor((IMG_SIZE - wCal) / 2)
                Img_white[:, wGap:wCal + wGap] = resized_image
            else:
                k = IMG_SIZE / w
                hCal = math.floor(k * h)
                resized_image = cv2.resize(crop_image, (IMG_SIZE, hCal))
                hGap = math.floor((IMG_SIZE - hCal) / 2)
                Img_white[hGap:hCal + hGap, :] = resized_image

            if start_collection and counter < dataset_size:
                file_path = os.path.join(class_dir, f'{counter}.jpg')
                cv2.imwrite(file_path, Img_white)
                counter += 1
                print(f"Saved {file_path}")

            cv2.imshow("Cropped Hand", Img_white)

        cv2.putText(image, f'Class {j} - Images Saved: {counter}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if start_collection else (0, 0, 255), 2)

        cv2.imshow("Webcam", image)

        key = cv2.waitKey(1)
        if key == ord('s'):
            start_collection = True
        elif key == ord('q') or counter >= dataset_size:
            break

print("\nHoàn thành việc thu thập dữ liệu!")
cap.release()
cv2.destroyAllWindows()
