import cv2
import cvzone.HandTrackingModule as htm
import numpy as np
import math
import os
import yaml
from cvzone.SelfiSegmentationModule import SelfiSegmentation

with open("src/config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

cap = cv2.VideoCapture(0)
detector = htm.HandDetector(maxHands=1)
segmentor = SelfiSegmentation()  # Kept but not used; can remove if desired

IMG_SIZE = config["Training"]["image_size"]
DATA_DIR = config["Training"]["DATA_PATH"]

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 10
dataset_size = 200

# Define finger connection chains (MediaPipe style, from wrist 0)
finger_connections = [
    [0, 1, 2, 3, 4],    # Thumb
    [0, 5, 6, 7, 8],    # Index
    [0, 9, 10, 11, 12], # Middle
    [0, 13, 14, 15, 16],# Ring
    [0, 17, 18, 19, 20] # Pinky
]

# Define palm connections (optional, for gray lines)
palm_connections = [
    (5, 9), (9, 13), (13, 17)
]

# Colors for fingers (adjusted to match the colorful style)
finger_colors = [
    (255, 255, 0),  # Yellow for thumb
    (0, 255, 0),    # Green for index
    (0, 0, 255),    # Blue for middle
    (128, 0, 128),  # Purple for ring
    (0, 255, 255)   # Cyan for pinky
]

# Gray for palm
palm_color = (128, 128, 128)  # Gray

# Dot colors: can vary per landmark, but for simplicity, red for palm/wrist, colored for fingers
# But to match, perhaps gradient, but keep simple: color per finger for tips, red for bases

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
        hands, image = detector.findHands(image, draw=False)  # Set draw=False to not draw on original image

        Img_black = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)  # Black background

        if hands:
            hand = hands[0]
            landmarks = hand['lmList']
            x, y, w, h = hand['bbox']

            # Add margin to bbox
            x1, y1 = max(x - 20, 0), max(y - 20, 0)
            x2, y2 = x + w + 20, y + h + 20
            crop_width = x2 - x1
            crop_height = y2 - y1

            if crop_width <= 0 or crop_height <= 0:
                continue

            aspectRatio = crop_height / crop_width

            # Calculate scaling and gaps to center the skeleton
            if aspectRatio > 1:
                k = IMG_SIZE / crop_height
                scaled_width = math.ceil(k * crop_width)
                scaled_height = IMG_SIZE
                wGap = math.ceil((IMG_SIZE - scaled_width) / 2)
                hGap = 0
            else:
                k = IMG_SIZE / crop_width
                scaled_height = math.ceil(k * crop_height)
                scaled_width = IMG_SIZE
                hGap = math.ceil((IMG_SIZE - scaled_height) / 2)
                wGap = 0

            # Map landmarks to new positions
            mapped_landmarks = []
            for lm in landmarks:
                rel_x = (lm[0] - x1) / crop_width
                rel_y = (lm[1] - y1) / crop_height
                new_x = wGap + rel_x * scaled_width
                new_y = hGap + rel_y * scaled_height
                mapped_landmarks.append((int(new_x), int(new_y)))

            # Draw palm connections in gray
            for conn in palm_connections:
                pt1 = mapped_landmarks[conn[0]]
                pt2 = mapped_landmarks[conn[1]]
                cv2.line(Img_black, pt1, pt2, palm_color, 2)

            # Draw finger connections with colors
            for i, conn in enumerate(finger_connections):
                color = finger_colors[i % len(finger_colors)]
                for j in range(len(conn) - 1):
                    pt1 = mapped_landmarks[conn[j]]
                    pt2 = mapped_landmarks[conn[j + 1]]
                    cv2.line(Img_black, pt1, pt2, color, 2)

            # Draw dots (colored based on finger, wrist red)
            dot_colors = [(0, 0, 255)] * 21  # Default red
            for i, conn in enumerate(finger_connections):
                finger_color = finger_colors[i % len(finger_colors)]
                for idx in conn[1:]:  # Skip wrist (0), color finger landmarks
                    dot_colors[idx] = finger_color

            for idx, (cx, cy) in enumerate(mapped_landmarks):
                cv2.circle(Img_black, (cx, cy), 5, dot_colors[idx], cv2.FILLED)

            if start_collection and counter < dataset_size:
                file_path = os.path.join(class_dir, f'{counter}.jpg')
                cv2.imwrite(file_path, Img_black)
                counter += 1
                print(f"Saved {file_path}")

            cv2.imshow("Cropped Hand", Img_black)

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