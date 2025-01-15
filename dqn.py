import random
import gymnasium as gym
import numpy as np
import torch

from torch import nn
from collections import deque
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 2),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

    
class DQNAgent:
    def __init__(self, env, epsilon=1.0, epsilon_min = 0.01, epsilon_decay=0.999,
                 memory_size=2000, lr=0.001, gamma=0.95, batch_size=32):
        self.env = env
        self.q_net = QNetwork().to(device)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size

    def get_action(self, obs):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            q_values = self.q_net(obs_t)
            action = torch.argmax(q_values).item()
            return action
        
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []


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
        for episode in range(episodes):
            print(f"Episode {episode}")
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
        
        torch.save(self.q_net.state_dict(), "./model")


    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load(self, model):
        self.q_net.load_state_dict(torch.load(model))
        self.q_net.eval()


if __name__ == "__main__":
    #env = gym.make("CartPole-v1", render_mode="rgb_array")
    #env = gym.wrappers.RecordVideo(env=env, video_folder="videos/",
    #                                episode_trigger=lambda x: x % 50 == 0)
    #agent = DQNAgent(env)

    #agent.train(episodes=1000)
    #print("Training complete.")

    env = gym.make("CartPole-v1", render_mode="human")
    agent = DQNAgent(env)
    agent.load("model")

    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

    env.close()