# dùng gpu để train
# dùng pytorch

import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from Backbone.CNN import CNNModel
from dataset.Dataset import ActionDataset

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


# Lấy danh sách lớp
actions = sorted([d for d in os.listdir(config["Training"]["DATA_PATH"]) if os.path.isdir(os.path.join(config["Training"]["DATA_PATH"], d))])
num_classes = len(actions)
print(f"Số lớp: {num_classes} - {actions}")

# Transform cho ảnh (có augmentation)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

# Load dataset
dataset = ActionDataset(config["Training"]["DATA_PATH"], transform=transform)

# Tách train/test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=config["Training"]["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config["Training"]["batch_size"])

# Kiểm tra số lượng ảnh từng class
def count_classes(loader):
    class_counts = [0] * num_classes
    for _, labels in loader:
        for label in labels:
            class_counts[label] += 1
    return class_counts

train_counts = count_classes(train_loader)
test_counts = count_classes(test_loader)

# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.bar(range(num_classes), train_counts, color='blue', alpha=0.7)
# plt.xticks(range(num_classes), actions, rotation=45)
# plt.title("Số lượng mẫu trong tập Train")

# plt.subplot(1, 2, 2)
# plt.bar(range(num_classes), test_counts, color='red', alpha=0.7)
# plt.xticks(range(num_classes), actions, rotation=45)
# plt.title("Số lượng mẫu trong tập Test")
# plt.tight_layout()
# plt.show()

model = CNNModel().to(config["Training"]["Device"])
print(model)

# Loss & Optimizer (có L2 regularization)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["Training"]["lr"], weight_decay=1e-4)

# Early Stopping setup
best_val_acc = 0
early_stop_counter = 0


train_acc_list, val_acc_list = [], []
train_loss_list, val_loss_list = [], []

# Train loop
for epoch in range(config["Training"]["epochs"]):
    model.train()
    running_loss = 0
    correct, total = 0, 0

    for images, labels in train_loader:
        images = images.to(config["Training"]["Device"])
        labels = labels.long().to(config["Training"]["Device"])

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
            images = images.to(config["Training"]["Device"])
            labels = labels.long().to(config["Training"]["Device"])
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

    print(f"[{epoch+1}/{config["Training"]["epochs"]}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'model.pth')
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= config["Training"]["patience"]:
            print("Early stopping triggered.")
            break

# Vẽ biểu đồ training
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(train_acc_list, label='Train Accuracy')
# plt.plot(val_acc_list, label='Val Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Độ chính xác theo Epoch')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(train_loss_list, label='Train Loss')
# plt.plot(val_loss_list, label='Val Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Độ lỗi theo Epoch')
# plt.legend()
# plt.tight_layout()
# plt.show()

# Lưu dữ liệu test
np.save('X_test.npy', np.array([x[0].numpy() for x in test_dataset]))
np.save('y_test.npy', np.array([x[1] for x in test_dataset]))
np.save('y_train.npy', np.array([x[1] for x in train_dataset]))
