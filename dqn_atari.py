import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.fc1(x)
        return self.fc2(x)
