import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Khởi tạo MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Load model đã train
model = load_model('model.h5')
actions = ['0', '1', '2', '3', '4', '5']


def predict_gesture(image):
    # Tiền xử lý ảnh
    img = cv2.resize(image, (224, 224))
    img = img / 255.0  # Chuẩn hóa
    img = np.expand_dims(img, axis=0)

    # Dự đoán
    prediction = model.predict(img, verbose=0)
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]

    return predicted_class, confidence


# Khởi tạo camera
cap = cv2.VideoCapture(0)
_, frame = cap.read()
frame_height, frame_width, _ = frame.shape

# Xử lý hands tracking
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Không thể đọc frame từ camera.")
            continue

        # Lật ảnh để hiển thị dạng gương
        frame = cv2.flip(frame, 1)

        # Xử lý ảnh với MediaPipe
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Tính toán bounding box
                x_max = y_max = 0
                x_min = frame_width
                y_min = frame_height

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * frame_width), int(lm.y * frame_height)
                    x_max = max(x, x_max)
                    y_max = max(y, y_max)
                    x_min = min(x, x_min)
                    y_min = min(y, y_min)

                # Thêm padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(frame_width, x_max + padding)
                y_max = min(frame_height, y_max + padding)

                # Crop vùng bàn tay
                cropped_hand = frame[y_min:y_max, x_min:x_max]

                if cropped_hand.size > 0:  # Kiểm tra xem ảnh có hợp lệ không
                    # Dự đoán cử chỉ
                    predicted_class, confidence = predict_gesture(cropped_hand)

                    # Hiển thị kết quả
                    cv2.putText(frame,
                                f"Class: {actions[predicted_class]} ({confidence:.2f})",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                2)

                    # Vẽ bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Vẽ landmarks trên frame gốc
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Hiển thị ảnh đã crop (tùy chọn)
                if cropped_hand.size > 0:
                    cropped_hand_display = cv2.resize(cropped_hand, (224, 224))
                    cv2.imshow('Cropped Hand', cropped_hand_display)

        # Hiển thị frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()