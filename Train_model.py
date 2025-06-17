# dùng gpu để train
# dùng pytorch

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

# Cấu hình
DATA_PATH = os.path.join('data', 'Dataset')
IMG_SIZE = 240
BATCH_SIZE = 32
EPOCHS = 100
if not torch.cuda.is_available():
    raise RuntimeError("GPU không khả dụng. Hãy bật GPU để tiếp tục training.")
DEVICE = torch.device('cuda')


# Lấy danh sách lớp
actions = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
num_classes = len(actions)
print(f"Số lớp: {num_classes} - {actions}")

# Dataset Class
class ActionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        for idx, action in enumerate(actions):
            action_path = os.path.join(root_dir, action)
            image_files = [f for f in os.listdir(action_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            print(f"Loading {len(image_files)} images for class '{action}'")
            for image_name in image_files:
                image_path = os.path.join(action_path, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                    self.images.append(image)
                    self.labels.append(idx)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Transform cho ảnh (có augmentation)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

# Load dataset
dataset = ActionDataset(DATA_PATH, transform=transform)

# Tách train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Kiểm tra số lượng ảnh từng class
def count_classes(loader):
    class_counts = [0] * num_classes
    for _, labels in loader:
        for label in labels:
            class_counts[label] += 1
    return class_counts

train_counts = count_classes(train_loader)
test_counts = count_classes(test_loader)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.bar(range(num_classes), train_counts, color='blue', alpha=0.7)
plt.xticks(range(num_classes), actions, rotation=45)
plt.title("Số lượng mẫu trong tập Train")

plt.subplot(1, 2, 2)
plt.bar(range(num_classes), test_counts, color='red', alpha=0.7)
plt.xticks(range(num_classes), actions, rotation=45)
plt.title("Số lượng mẫu trong tập Test")
plt.tight_layout()
plt.show()

# Mô hình CNN
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
            nn.Linear(128 * 30 * 30, 240),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(240, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNNModel().to(DEVICE)
print(model)

# Loss & Optimizer (có L2 regularization)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Early Stopping setup
best_val_acc = 0
early_stop_counter = 0
patience = 10

train_acc_list, val_acc_list = [], []
train_loss_list, val_loss_list = [], []

# Train loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct, total = 0, 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.long().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    # Validation
    model.eval()
    val_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.long().to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(test_loader)
    val_acc = correct / total

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    print(f"[{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'model.pth')
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break

# Vẽ biểu đồ training
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_acc_list, label='Train Accuracy')
plt.plot(val_acc_list, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Độ chính xác theo Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_list, label='Train Loss')
plt.plot(val_loss_list, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Độ lỗi theo Epoch')
plt.legend()
plt.tight_layout()
plt.show()

# Lưu dữ liệu test
np.save('X_test.npy', np.array([x[0].numpy() for x in test_dataset]))
np.save('y_test.npy', np.array([x[1] for x in test_dataset]))
np.save('y_train.npy', np.array([x[1] for x in train_dataset]))
