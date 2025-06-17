import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import TensorDataset, DataLoader

# ==== Cấu hình ====
BATCH_SIZE = 32
DEVICE = "cpu"
sns.set_theme(style="whitegrid")

# ==== Mô hình ====
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout(0.5),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 30 * 30, 240), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(240, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==== Load model ====
model = CNNModel()
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==== Load dữ liệu ====
X_test = np.load("X_test.npy").astype(np.float32)
y_test = np.load("y_test.npy").astype(np.int64)

# Nếu là NHWC -> chuyển sang NCHW
if X_test.shape[1] == 240 and X_test.shape[-1] == 3:
    X_test = np.transpose(X_test, (0, 3, 1, 2))

# Tạo Tensor
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)

# Dataloader
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==== Đánh giá ====
all_preds, all_labels = [], []
loss_fn = nn.CrossEntropyLoss()
total_loss = 0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        total_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

# ==== Kết quả ====
y_pred_np = np.array(all_preds)
y_true_np = np.array(all_labels)
accuracy = accuracy_score(y_true_np, y_pred_np)
avg_loss = total_loss / len(test_loader.dataset)

print(f"\n✅ Accuracy: {accuracy:.4f}")
print(f"📉 Loss: {avg_loss:.4f}")
print("\n📑 Classification Report:")
print(classification_report(y_true_np, y_pred_np))

# ==== Confusion Matrix ====
cm = confusion_matrix(y_true_np, y_pred_np)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
