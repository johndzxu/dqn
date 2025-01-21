import random
import torch.nn as nn
import torch.nn.functional as F
from utils.memories import ReplayBuffer, ReservoirBuffer
import numpy as np
import torch


class NFSPAgent:
    def __init__(
        self, env, agent, eta, epsilon, reservoir_size, replay_size, batch_size
    ):
        self.env = env
        self.agent = agent
        self.rl_memory = ReplayBuffer(replay_size)
        self.sl_memory = ReservoirBuffer(reservoir_size)
        self.rl_criterion = nn.MSELoss()
        self.sl_criterion = nn.NLLLoss()
        self.batch_size = batch_size
        self.Pi = DQN
        self.Q = DQN
        self.target_Q = DQN
        self.get_policy_action = self.get_best_action

        self.target_Q.load()

        self.eta = eta
        self.epsilon = epsilon

    def get_best_action(self, obs):
        return self.Q(obs).max()

    def get_average_action(self, obs):
        return self.Pi(obs)

    def reset_policy(self):
        if random.random() < self.eta:
            self.get_policy_action = self.get_best_action
        else:
            self.get_policy_action = self.get_average_action

    def play(self):
        obs, reward, termination, truncation, info = self.env.last()

        if termination or truncation:
            action = None
        else:
            action = self.get_policy_action(obs)
            self.env.step(action)
            next_obs, reward, termination, truncation, info = self.env.last()
            self.rl_memory.store([obs, action, reward, next_obs])
            if self.get_policy_action == self.get_best_action:
                self.sl_memory.store([obs, action])

    def SGD(self):
        # Update best response strategy
        obs_batch, act_batch, reward_batch, next_obs_batch, done_batch = (
            self.memory.sample(self.batch_size)
        )

        obs_batch = torch.from_numpy(obs_batch).type(torch.float32)
        next_obs_batch = torch.from_numpy(next_obs_batch).type(torch.float32)
        act_batch = torch.from_numpy(act_batch)
        reward_batch = torch.from_numpy(reward_batch)
        done_batch = torch.from_numpy(done_batch.astype(np.uint8))

        Q_values = self.Q(obs_batch)
        Q_values = Q_values.gather(1, act_batch.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_Q_values = self.target_Q(next_obs_batch)
            next_Q_values = next_Q_values.max(dim=1)[0]
            target_Q_values = reward_batch + self.gamma * next_Q_values * (
                1 - done_batch
            )

        self.optimizer.zero_grad()
        rl_loss = self.rl_criterion(Q_values, target_Q_values)
        rl_loss.backward()
        self.optimizer.step()

        # Update average strategy
        obs_batch, act_batch = self.sl_memory.sample(self.batch_size)
        obs_batch = torch.from_numpy(obs_batch).type(torch.float32)
        act_batch = torch.from_numpy(act_batch)

        with torch.no_grad():
            target_act_batch = self.Pi(obs_batch)
            target_act_batch = target_act_batch.max(dim=1)[0]

        self.optimizer.zero_grad()
        sl_loss = self.sl_criterion(act_batch, target_act_batch)
        sl_loss.backward()
        self.optimizer.step()

    def update_target_Q(self):
        self.target_Q.load_state_dict(self.Q.state_dict())


class Arena:
    def __init__(self, env):
        self.env = env
        self.players = [players]

    def train(self, episodes):
        for episode in range(1, episodes + 1):
            self.env.reset()
            for p in self.players:
                p.reset_policy()

            for agent in self.env.iter_agents():
                self.players[agent].play()
                self.players[agent].SGD()


class DQN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return F.relu(x)
