import random
import torch.nn as nn
from utils.memories import ReplayBuffer, Reservoir


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fl1 = nn.Linear(64)
        self.fl2 = nn.Linear(64, out_dim)

    def forward(self):
        pass


class NFSPAgent:
    def __init__(self, env, agent, eta, epsilon, memory_size):
        self.env = env
        self.agent = agent
        self.rl_memory = ReplayBuffer(memory_size)
        self.sl_memory = Reservoir()
        self.Pi = DQN
        self.Q = DQN
        self.target_Q = DQN
        self.get_policy_action = self.get_best_action

        target_Q.load()

        self.eta = eta
        self.epsilon = epsilon

    def get_best_action(self, obs):
        return self.Q(obs).max()

    def get_average_action(self, obs):
        return self.Pi(obs)

    def reset_policy(self):
        if random.random() < eta:
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
        rl_loss = nn.MSELoss()
        sl_loss = nn.NLLLoss()

        backward()

    def update_target_Q():
        pass


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
