import yaml
import numpy as np
import cv2
from torch.utils.data import Dataset
import os

with open("src/config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

actions = sorted([d for d in os.listdir(config["Training"]["DATA_PATH"]) if os.path.isdir(os.path.join(config["Training"]["DATA_PATH"], d))])


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
                    image = cv2.resize(image, (config["Training"]["image_size"], config["Training"]["image_size"]))
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