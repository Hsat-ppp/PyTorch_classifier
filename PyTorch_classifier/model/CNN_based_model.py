import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, in_channel, out_channel, num_of_flat_conv):
        """
        init function which defines network architecture
        """
        super().__init__()
        self.conv1 = nn.ModuleList([nn.Conv2d(in_channel, in_channel, 3, padding=1) for _ in range(num_of_flat_conv)])
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(in_channel) for _ in range(num_of_flat_conv)])
        self.conv_output = nn.Conv2d(in_channel, out_channel, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        for l, bn in zip(self.conv1, self.bn1):
            x = F.relu(bn(l(x)))
        x = F.relu(self.bn2(self.conv_output(x)))
        return x


class CNNBlockSkipConnection(nn.Module):
    def __init__(self, in_channel, out_channel, num_of_flat_conv):
        """
        init function which defines network architecture
        """
        super().__init__()
        self.conv1 = nn.ModuleList([nn.Conv2d(in_channel, in_channel, 3, padding=1) for _ in range(num_of_flat_conv)])
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(in_channel) for _ in range(num_of_flat_conv)])
        self.conv_output = nn.Conv2d(in_channel, out_channel, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x_input = x.clone()
        for l, bn in zip(self.conv1, self.bn1):
            x = F.relu(bn(l(x)))
        x = F.relu(x + x_input)
        x = F.relu(self.bn2(self.conv_output(x)))
        return x


class BasicCNNClassifier(nn.Module):
    """
    basic CNN classifier model.
    """
    def __init__(self):
        """
        init function which defines network architecture
        """
        super().__init__()
        self.init_conv = nn.Conv2d(3, 32, 3, padding=1)
        # self.blocks = nn.ModuleList([CNNBlock(32, 64, 2), CNNBlock(64, 128, 2), CNNBlock(128, 256, 2)])
        self.blocks = nn.ModuleList([CNNBlockSkipConnection(32, 64, 2), CNNBlockSkipConnection(64, 128, 2), CNNBlockSkipConnection(128, 256, 2)])
        self.fc1 = nn.ModuleList([nn.Linear(256*4*4, 1024), nn.Linear(1024, 128)])
        self.do1 = nn.ModuleList([nn.Dropout(p=0.2) for _ in range(len(self.fc1))])
        self.output_fc = nn.Linear(128, 10)

    def forward(self, x):
        """
        forward calculation
        :param x: input img data
        :return: classification
        """
        x = F.relu(self.init_conv(x))
        for bl in self.blocks:
            x = bl(x)
        x = x.view(-1, 256*4*4)  # flatten
        for l, d in zip(self.fc1, self.do1):
            x = d(F.relu(l(x)))
        x = self.output_fc(x)
        return x
