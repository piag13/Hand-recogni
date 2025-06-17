import cv2
import torch
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
import torch.nn as nn
import time

# ==== Cấu hình ====
IMG_SIZE = 240
classes = [str(i) for i in range(10)]

# ==== Bắt buộc dùng GPU ====
if not torch.cuda.is_available():
    raise RuntimeError("❌ Không phát hiện GPU! Vui lòng bật GPU để chạy.")
DEVICE = torch.device('cuda')
print("✅ GPU đang dùng:", torch.cuda.get_device_name(0))

# ==== Định nghĩa mô hình CNN ====
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 30 * 30, 240),  # do ảnh đầu vào 240x240
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(240, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(classes))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==== Load mô hình đã huấn luyện ====
model = CNNModel().to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()

# ==== Hàm dự đoán ====
def predict_gesture(image_np):
    image = image_np.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image_tensor = torch.tensor(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).item()
    return pred

# ==== Hand detector & webcam ====
detector = HandDetector(maxHands=2)
cap = cv2.VideoCapture(0)

# FPS counter
prev_time = time.time()

while True:
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    hands, image = detector.findHands(image)

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            hand_type = hand["type"]  # 'Left' hoặc 'Right'

            img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8) * 255
            x1, y1 = max(x - 20, 0), max(y - 20, 0)
            x2, y2 = x + w + 20, y + h + 20
            crop_image = image[y1:y2, x1:x2]

            if crop_image.size == 0:
                continue

            # Resize giữ tỉ lệ
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

            # Dự đoán
            prediction = predict_gesture(img_white)
            class_label = f"{hand_type}: {classes[prediction]}"

            # Vẽ kết quả
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text_pos = (x1, y2 + 30)
            cv2.putText(image, class_label, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(image, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Webcam", image)
    # cv2.imshow("Cropped Hand", img_white)  # chỉ hiển thị tay cuối

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Hoàn thành nhận diện tay trái/phải riêng biệt!")

