import random
import gymnasium as gym
import numpy as np
import torch

from torch import nn
from collections import deque
from matplotlib import pyplot as plt

import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
class DQNAgent():
    def __init__(self, env, epsilon, memory_size):
        self.env = env
        self.q_net = QNetwork()
        self.epsilon = epsilon
        self.memory = deque(maxlen=memory_size)

    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.q_net(torch.from_numpy(obs).float())
            action = torch.argmax(q_values).item()
            return action
        

    def train(self, episodes, batch_size, lr, gamma):
        criterion = nn.MSELoss()
        for e in range(episodes):
            print(f"training episode {e}")
            episode_over = False
            state, _ = self.env.reset()
            optimizer = optim.SGD(self.q_net.parameters(), lr=lr)
            while not episode_over:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.memory.append([state, action, reward, next_state, done])

                if len(self.memory) >= batch_size:
                    minibatch = random.sample(self.memory, batch_size)
                    for state, action, reward, next_state, done in minibatch:
                        state_t = torch.from_numpy(state).float()
                        next_state_t = torch.from_numpy(next_state).float()

                        q_values = self.q_net(state_t)
                        output = q_values[action]
                        
                        with torch.no_grad():
                            next_q_values = self.q_net(next_state_t)
                            max_next_q = torch.max(next_q_values).item()
                        target = reward if done else reward + gamma*max_next_q

                        target_t = torch.tensor([target], dtype=torch.float32)
                        output_t = output.unsqueeze(0)

                        optimizer.zero_grad()
                        loss = criterion(output, target_t)
                        loss.backward()
                        optimizer.step()

                state = next_state

                if terminated or truncated:
                    episode_over = True


env = gym.make('CartPole-v1')
agent = DQNAgent(env, 0.05, 10000)
agent.train(10000, 100, 0.01, 0.1)
print("done")