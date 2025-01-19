import random
import time
import gymnasium as gym
import numpy as np
import torch

from matplotlib import pyplot as plt
from torch import nn
from collections import deque
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_dim, 120),
            nn.ReLU(),
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.Linear(120, out_dim),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

    
class DQNAgent:
    def __init__(self, env, in_dim, out_dim, epsilon=1.0, epsilon_min = 0.05, epsilon_decay=0.95,
                 memory_size=10000, lr=0.001, gamma=0.99, batch_size=128):
        self.env = env
        self.q_net = QNetwork(in_dim, out_dim).to(device)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size

    def get_action(self, obs):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            q_values = self.q_net(obs_t)
            action = torch.argmax(q_values).item()
            return action
        
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array()
        actions = np.array()
        rewards = np.array()
        next_states = np.array()
        dones = np.array()


        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
            
        states_t = torch.FloatTensor(states).to(device)
        actions_t = torch.LongTensor(actions).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        next_states_t = torch.FloatTensor(next_states).to(device)
        dones_t = torch.FloatTensor(dones).to(device)

        q_values = self.q_net(states_t)
        outputs = q_values.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
        
        with torch.no_grad():
            next_q_values = self.q_net(next_states_t)
            max_next_q_values = next_q_values.max(dim=1)[0]
            targets = rewards_t + self.gamma * max_next_q_values * (1-dones_t)

        self.optimizer.zero_grad()
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()


    def train(self, episodes):
        scores = []
        scores_avg = []

        plt.ion()
        fig, ax = plt.subplots()

        for episode in range(episodes):
            state, _ = self.env.reset()
            done = False
            score = 0

            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                score += reward

                self.memory.append([state, action, reward, next_state, done])
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
        while not done:
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated
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
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env=env, video_folder="videos",
    #                                 name_prefix=f"{env.spec.name}",
    #                                 episode_trigger=lambda x: x%50 == 0)
    # agent = DQNAgent(env, 8, 4)
    # # agent.load("model_params/MountainCar.params")
    # # agent.epsilon = 0.5
    # # agent.epsilon_min = 0.01
    
    # agent.train(episodes=500)
    

    # env = gym.make("CartPole-v1", render_mode="rgb_array")
    # env = gym.wrappers.RecordVideo(env=env, video_folder="videos",
    #                                 episode_trigger=lambda x: x%50 == 0)
    # agent.env = env
    agent = DQNAgent(env, 8, 4)
    agent.load("model_params/LunarLander.params")
    test(agent)

    # play(agent)