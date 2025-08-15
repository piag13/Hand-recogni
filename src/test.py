import cv2
import torch
import numpy as np
import math
import time
import yaml
from cvzone.HandTrackingModule import HandDetector
from Backbone.CNN import CNNModel


class ConfigLoader:
    """Lớp load cấu hình từ file YAML."""
    @staticmethod
    def load(path="src/config/config.yaml"):
        with open(path, "r") as file:
            return yaml.safe_load(file)


class HandGestureRecognizer:
    """Lớp nhận diện cử chỉ tay với mô hình CNN."""
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["Training"]["Device"])

        # Kiểm tra GPU
        if "cuda" in config["Training"]["Device"] and not torch.cuda.is_available():
            raise RuntimeError("❌ Không phát hiện GPU! Vui lòng bật GPU để chạy.")
        print(f"✅ Sử dụng thiết bị: {self.device}")

        # Load model
        self.model = CNNModel().to(self.device)
        self.model.load_state_dict(torch.load(config["Model"]["Model_path"], map_location=self.device))
        self.model.eval()

        # Load thông tin khác
        self.classes = [str(i) for i in range(config["Training"]["Num_classes"])]
        self.img_size = config["Training"]["image_size"]

        self.detector = HandDetector(maxHands=2)

        # Define finger connection chains (MediaPipe style, from wrist 0)
        self.finger_connections = [
            [0, 1, 2, 3, 4],    # Thumb
            [0, 5, 6, 7, 8],    # Index
            [0, 9, 10, 11, 12], # Middle
            [0, 13, 14, 15, 16],# Ring
            [0, 17, 18, 19, 20] # Pinky
        ]

        # Define palm connections (optional, for gray lines)
        self.palm_connections = [
            (5, 9), (9, 13), (13, 17)
        ]

        # Colors for fingers (adjusted to match the colorful style)
        self.finger_colors = [
            (255, 255, 0),  # Yellow for thumb
            (0, 255, 0),    # Green for index
            (0, 0, 255),    # Blue for middle
            (128, 0, 128),  # Purple for ring
            (0, 255, 255)   # Cyan for pinky
        ]

        # Gray for palm
        self.palm_color = (128, 128, 128)  # Gray

    def _preprocess(self, image_np):
        """Tiền xử lý ảnh trước khi đưa vào model."""
        image = image_np.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image_tensor = torch.tensor(image).unsqueeze(0).to(self.device)
        return image_tensor

    def predict(self, image_np):
        """Dự đoán nhãn từ ảnh."""
        image_tensor = self._preprocess(image_np)
        with torch.no_grad():
            output = self.model(image_tensor)
            pred = torch.argmax(output, dim=1).item()
        return pred

    def _create_landmark_image(self, landmarks, bbox):
        """Tạo ảnh với các điểm landmark trên nền đen."""
        Img_black = np.zeros((self.img_size, self.img_size, 3), np.uint8)  # Black background

        x, y, w, h = bbox

        # Add margin to bbox
        x1, y1 = max(x - 20, 0), max(y - 20, 0)
        x2, y2 = x + w + 20, y + h + 20
        crop_width = x2 - x1
        crop_height = y2 - y1

        if crop_width <= 0 or crop_height <= 0:
            return None

        aspectRatio = crop_height / crop_width

        # Calculate scaling and gaps to center the skeleton
        if aspectRatio > 1:
            k = self.img_size / crop_height
            scaled_width = math.ceil(k * crop_width)
            scaled_height = self.img_size
            wGap = math.ceil((self.img_size - scaled_width) / 2)
            hGap = 0
        else:
            k = self.img_size / crop_width
            scaled_height = math.ceil(k * crop_height)
            scaled_width = self.img_size
            hGap = math.ceil((self.img_size - scaled_height) / 2)
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
        for conn in self.palm_connections:
            pt1 = mapped_landmarks[conn[0]]
            pt2 = mapped_landmarks[conn[1]]
            cv2.line(Img_black, pt1, pt2, self.palm_color, 2)

        # Draw finger connections with colors
        for i, conn in enumerate(self.finger_connections):
            color = self.finger_colors[i % len(self.finger_colors)]
            for j in range(len(conn) - 1):
                pt1 = mapped_landmarks[conn[j]]
                pt2 = mapped_landmarks[conn[j + 1]]
                cv2.line(Img_black, pt1, pt2, color, 2)

        # Draw dots (colored based on finger, wrist red)
        dot_colors = [(0, 0, 255)] * 21  # Default red
        for i, conn in enumerate(self.finger_connections):
            finger_color = self.finger_colors[i % len(self.finger_colors)]
            for idx in conn[1:]:  # Skip wrist (0), color finger landmarks
                dot_colors[idx] = finger_color

        for idx, (cx, cy) in enumerate(mapped_landmarks):
            cv2.circle(Img_black, (cx, cy), 5, dot_colors[idx], cv2.FILLED)

        return Img_black

    def process_frame(self, frame):
        """Xử lý 1 frame video, trả về frame đã gắn nhãn."""
        hands, image = self.detector.findHands(frame, draw=False)  # Set draw=False to not draw on original image
        if hands:
            for hand in hands:
                landmarks = hand['lmList']
                bbox = hand['bbox']
                hand_type = hand["type"]  # Left / Right

                landmark_image = self._create_landmark_image(landmarks, bbox)
                if landmark_image is None:
                    continue

                prediction = self.predict(landmark_image)
                class_label = f"{hand_type}: {self.classes[prediction]}"

                # Vẽ kết quả
                x1, y1 = bbox[0], bbox[1]
                x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(image, class_label, (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return image


class WebcamApp:
    """Lớp chạy ứng dụng webcam."""
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.cap = cv2.VideoCapture(0)
        self.prev_time = time.time()

    def run(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            frame = self.recognizer.process_frame(frame)

            # Tính FPS
            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time)
            self.prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Hand Gesture Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Hoàn thành nhận diện tay trái/phải!")