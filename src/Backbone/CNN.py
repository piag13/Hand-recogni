import torch.nn as nn
import yaml

with open("src/config/config.yaml", "r") as file:
    config = yaml.safe_load(file) 


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
            nn.Linear(64, config["Training"]["Num_classes"])
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
