import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCNNClassifier(nn.Module):
    # define network architecture
    def __init__(self):
        super().__init__()
        self.init_conv = nn.Conv2d(3, 16, 3, padding=1)
        self.conv1 = nn.ModuleList([nn.Conv2d(16, 16, 3, padding=1) for _ in range(3)])
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(16) for _ in range(3)])
        self.pool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.ModuleList([nn.Linear(16*16*16, 128), nn.Linear(128, 32)])
        self.output_fc = nn.Linear(32, 10)

    # forward calculation
    def forward(self, x):
        x = F.relu(self.init_conv(x))
        for l, bn in zip(self.conv1, self.bn1):
            x = F.relu(bn(l(x)))
        x = self.pool(x)
        x = x.view(-1, 16*16*16)  # flatten
        for l in self.fc1:
            x = F.relu(l(x))
        x = self.output_fc(x)
        return x
