import random
import time
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    filename="dqn.log",
    encoding="utf-8",
    filemode="a"
)

class DQN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=4,
                        out_channels=16,
                        kernel_size=8,
                        stride=4,
                        padding=0),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                        out_channels=32,
                        kernel_size=4,
                        stride=2,
                        padding=0),
            nn.ReLU())
        
        self.fc1 = nn.Sequential(
            nn.Linear(2592, 256),
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

class EpsilonDecaySchedule:
        def __init__(self, start, end, steps):
            self.step = 0
            self.schedule = np.linspace(start, end, steps)

        def next_epsilon(self):
            self.step += 1
            return self.schedule[min(self.step, len(self.schedule)-1)]
    
class DQNAgent:
    def __init__(self, env, epsilon=1.0, epsilon_min = 0.05, epsilon_decay=0.95,
                 memory_size=100000, lr=0.001, gamma=0.99, batch_size=32):
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
        self.k = 4

    def get_action(self, history):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.q_net(history)
            action = torch.argmax(q_values).item()
            return action
        
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
        epsilon_schedule = EpsilonDecaySchedule(self.epsilon, self.epsilon_min, 1000)
        scores = []
        scores_avg = []

        # plt.ion()
        # fig, ax = plt.subplots()

        for episode in range(1, episodes+1):
            sequence = []
            state, _ = self.env.reset()

            sequence.append(state)

            done = False
            score = 0

            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                score += reward

                sequence.append(next_state)

                self.memory.append([state, action, reward, next_state, (reward != 0)])
                if len(self.memory) >= self.batch_size:
                    self.replay()

                state = next_state

            self.epsilon = epsilon_schedule.next_epsilon()

            logging.info(f"episode: {episode}/{episodes}, score: {score}, e: {self.epsilon:.2}")
            scores.append(score)
            scores_avg.append(np.mean(scores[-10:]))

            # ax.cla()
            # ax.plot(scores)
            # ax.plot(scores_avg)
            # ax.set_xlabel("Training Episode")
            # ax.set_ylabel("Score")
            # fig.canvas.flush_events()
            if (episode+1)%25 == 0:
                torch.save(self.q_net.state_dict(), f"model_params/{self.env.spec.name}2.params.save")
                logging.info("Model parameters saved.")

        torch.save(self.q_net.state_dict(), f"model_params/{self.env.spec.name}2.params")
        logging.info("Training completed. Model parameters saved.")


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
        print(f"episode {episode}")
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

class DownsampleFrameWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        obs, info = self.env.reset()
        obs_t = torch.FloatTensor(obs).unsqueeze(0).unsqueeze(0)
        downsampled = nn.functional.interpolate(
            obs_t, (110, 84), mode="bilinear"
        ).squeeze(0).squeeze(0)
        downsampled = downsampled[20:104]
        return downsampled, info

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0).unsqueeze(0)
        downsampled = nn.functional.interpolate(
            next_state_t, (110, 84), mode="bilinear"
        ).squeeze(0).squeeze(0)
        downsampled = downsampled[20:104]
        return downsampled, reward, terminated, truncated, info

class StackFrameWrapper(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.buffer = deque(maxlen=k)
        self.k = k

    def reset(self):
        obs, info = self.env.reset()
        
        for _ in range(self.k):
            self.buffer.append(obs)
        return torch.stack(tuple(self.buffer)), info
    
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.buffer.append(next_state)
        
        return torch.stack(tuple(self.buffer)), reward, terminated, truncated, info

class MagnifyRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        if reward == 1:
            reward = 2
        return next_state, reward, terminated, truncated, info

if __name__ == "__main__":
    env = gym.make("Pong-v4", obs_type="grayscale", render_mode="rgb_array")
    # env = gym.make("Pong-v4", obs_type="grayscale", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env=env, video_folder="videos",
                                    name_prefix=f"{env.spec.name}",
                                    episode_trigger=lambda x: x%25 == 0)
    env = DownsampleFrameWrapper(env)
    env = StackFrameWrapper(env, 4)
    env = MagnifyRewardWrapper(env)

    agent = DQNAgent(env)
    # agent.load("model_params/Pong.params.save")
    # agent.epsilon = 0.1
    # agent.epsilon_min = 0.1
    
    logging.info("Begin training")
    agent.train(episodes=10000)
