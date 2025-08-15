import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from Backbone.CNN import CNNModel
from Dataset import ActionDataset


class TrainModel:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config["Training"]["Device"])

        if "cuda" in self.config["Training"]["Device"] and not torch.cuda.is_available():
            raise RuntimeError("❌ Không phát hiện GPU! Vui lòng bật GPU để train.")

        # Dataset
        self.actions = sorted([
            d for d in os.listdir(self.config["Training"]["DATA_PATH"])
            if os.path.isdir(os.path.join(self.config["Training"]["DATA_PATH"], d))
        ])
        self.num_classes = len(self.actions)
        print(f"Số lớp: {self.num_classes} - {self.actions}")

        transform = self._init_transform()
        dataset = ActionDataset(self.config["Training"]["DATA_PATH"], transform=transform)

        # Train/test split
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.config["Training"]["batch_size"], shuffle=True)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.config["Training"]["batch_size"])

        # Model
        self.model = CNNModel().to(self.device)
        print(self.model)

        # Loss & optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.config["Training"]["lr"],
                                    weight_decay=1e-4)

        # Early stopping
        self.best_val_acc = 0
        self.early_stop_counter = 0

        # Logs
        self.train_acc_list, self.val_acc_list = [], []
        self.train_loss_list, self.val_loss_list = [], []

    def _load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for key in ["Training"]:
            if key not in cfg:
                raise KeyError(f"Missing key in config: {key}")
        return cfg

    def _init_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor()
        ])

    def count_classes(self, loader):
        """Đếm số lượng ảnh từng class."""
        class_counts = [0] * self.num_classes
        for _, labels in loader:
            for label in labels:
                class_counts[label] += 1
        return class_counts

    def train(self):
        epochs = self.config["Training"]["epochs"]
        patience = self.config["Training"]["patience"]

        for epoch in range(epochs):
            # Training
            self.model.train()
            running_loss, correct, total = 0, 0, 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.long().to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / len(self.train_loader)
            train_acc = correct / total

            # Validation
            val_loss, val_acc = self.validate()

            # Log
            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)
            self.val_loss_list.append(val_loss)
            self.val_acc_list.append(val_acc)

            print(f"[{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'model.pth')
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= patience:
                    print("Early stopping triggered.")
                    break

        # Lưu dataset test/train
        self.save_dataset()

    def validate(self):
        self.model.eval()
        val_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.long().to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(self.test_loader)
        val_acc = correct / total
        return val_loss, val_acc

    def save_dataset(self):
        np.save('X_test.npy', np.array([x[0].numpy() for x in self.test_dataset]))
        np.save('y_test.npy', np.array([x[1] for x in self.test_dataset]))
        np.save('y_train.npy', np.array([x[1] for x in self.train_dataset]))
        print("✅ Đã lưu dataset test/train.")


if __name__ == "__main__":
    trainer = TrainModel("src/config/config.yaml")
    trainer.train()
