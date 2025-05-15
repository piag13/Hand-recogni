import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# Load model đã train
model = load_model('model.h5')

def evaluate_model():
    # Load model và dữ liệu test
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')

    # Đánh giá mô hình
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Loss: {test_loss:.4f} - Accuracy: {test_acc:.4f}")

    # Dự đoán
    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    # Ma trận nhầm lẫn
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    actions = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=actions, yticklabels=actions)
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.title("Seaborn")
    plt.show()

    # Báo cáo phân loại
    print(classification_report(y_true_labels, y_pred_labels, target_names=actions))


if __name__ == "__main__":
    evaluate_model()


