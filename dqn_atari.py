import random
import time
import gymnasium as gym
import numpy as np
import torch
import ale_py

from matplotlib import pyplot as plt
from torch import nn, tensor
from collections import deque
import torch.optim as optim

gym.register_envs(ale_py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                        out_channels=16,
                        kernel_size=8,
                        stride=4,
                        padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                        out_channels=32,
                        kernel_size=4,
                        stride=2,
                        padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)))
        
        self.fc1 = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU())

        self.fc2 = nn.Linear(256, 6)


    def forward(self, x):
        if x.ndimension() == 3:
            x = x.unsqueeze(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return self.fc2(x)


    
class DQNAgent:
    def __init__(self, env, epsilon=1.0, epsilon_min = 0.05, epsilon_decay=0.95,
                 memory_size=10000, lr=0.001, gamma=0.99, batch_size=128):
        self.env = env
        self.q_net = DQN().to(device)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size

    def get_action(self, history):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.q_net(history)
            action = torch.argmax(q_values).item()
            return action
        
    def preprocess(self, sequence):
        t = torch.empty(3, 210, 160)
        for i in range(len(sequence[-3:])):
            t[i] = torch.FloatTensor(sequence[i])
        return t
        
        
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        
        histories, actions, rewards, next_histories, dones = zip(*minibatch)

        histories_t = torch.stack(histories).to(device)
        actions_t = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        next_histories_t = torch.stack(next_histories).to(device)
        dones_t = torch.FloatTensor(dones).to(device)

        q_values = self.q_net(histories_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            next_q_values = self.q_net(next_histories_t)
            max_next_q_values = next_q_values.max(dim=1)[0]
            targets = rewards_t + self.gamma * max_next_q_values * (1 - dones_t)

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, targets)
        loss.backward()
        self.optimizer.step()


    def train(self, episodes):
        scores = []
        scores_avg = []

        plt.ion()
        fig, ax = plt.subplots()

        for episode in range(episodes):
            sequence = []
            state, _ = self.env.reset()

            sequence.append(state)
            preprocessed = self.preprocess(sequence)

            done = False
            score = 0

            for t in range(100):
                print(t)
                action = self.get_action(preprocessed)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = False
                score += reward

                sequence.append(next_state)
                next_preprocessed = self.preprocess(sequence)

                self.memory.append([preprocessed, action, reward, next_preprocessed, done])
                if len(self.memory) >= self.batch_size:
                    self.replay()

                state = next_state
                self.decay_epsilon()

            print(f"episode: {episode}/{episodes}, score: {score}, e: {self.epsilon:.2}")
            scores.append(score)
            scores_avg.append(np.mean(scores[-10:]))

            ax.cla()
            ax.plot(scores)
            ax.plot(scores_avg)
            ax.set_xlabel("Training Episode")
            ax.set_ylabel("Score")
            fig.canvas.flush_events()

        torch.save(self.q_net.state_dict(), f"model_params/{self.env.spec.name}.params")


    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

    def load(self, model):
        self.q_net.load_state_dict(torch.load(model, weights_only=True))
        self.q_net.eval()

def play(agent):
    # agent.epsilon = 0
    # agent.epsilon_min = 0
    while True:
        obs, _ = env.reset()

        episode_over = False
        while not episode_over:
            action = agent.get_action(obs)
            obs, _, terminated, truncated, _ = agent.env.step(action)

            episode_over = terminated or truncated
        
        time.sleep(1)

def test(agent, episodes=200):
    agent.epsilon = 0
    agent.epsilon_min = 0
    
    scores = []
    scores_avg = []

    plt.ion()
    fig, ax = plt.subplots()
    

    for episode in range(episodes):
        state, _ = agent.env.reset()
        
        done = False
        score = 0
        for t in range(500):
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = agent.env.step(action)
            score += reward
            
        print(f"episode: {episode}/{episodes}, score: {score}")
        scores.append(score)
        scores_avg.append(np.mean(scores[-10:]))

        ax.cla()
        ax.plot(scores)
        ax.plot(scores_avg)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score")
        fig.canvas.flush_events()

    print(f"average: {np.mean(scores)}")

if __name__ == "__main__":
    env = gym.make("Pong-v4", obs_type="grayscale", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env=env, video_folder="videos",
                                    name_prefix=f"{env.spec.name}",
                                    episode_trigger=lambda x: x%50 == 0)
    agent = DQNAgent(env)
    # # agent.load("model_params/MountainCar.params")
    # # agent.epsilon = 0.5
    # # agent.epsilon_min = 0.01
    
    agent.train(episodes=500)
    

    # env = gym.make("CartPole-v1", render_mode="rgb_array")
    # env = gym.wrappers.RecordVideo(env=env, video_folder="videos",
    #                                 episode_trigger=lambda x: x%50 == 0)
    # agent.env = env
    # agent = DQNAgent(env, 8, 4)
    # agent.load("model_params/LunarLander.params")
    # test(agent)

    # play(agent)