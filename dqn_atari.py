import random
import gymnasium as gym
import numpy as np
import torch
import ale_py
import logging

from matplotlib import pyplot as plt
from torch import nn, tensor
from collections import deque
import torch.optim as optim

gym.register_envs(ale_py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(7)


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=4, out_channels=16, kernel_size=8, stride=4, padding=0
            ),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(nn.Linear(2592, 256), nn.ReLU())

        self.fc2 = nn.Linear(256, 6)

    def forward(self, x):
        if x.ndimension() == 3:
            x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return self.fc2(x)
