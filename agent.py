from collections import deque
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dqn_atari import DQN


class EpsilonDecaySchedule:
    def __init__(self, start, end, steps):
        self.step = 0
        self.schedule = np.linspace(start, end, steps)

    def next_epsilon(self):
        self.step += 1
        return self.schedule[min(self.step, len(self.schedule) - 1)]


class DQNAgent:
    def __init__(
        self,
        env,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay_steps=1000000,
        memory_size=1000000,
        learning_rate=0.00025,
        gamma=0.99,
        batch_size=32,
        k=4,
        num_actions=6,
    ):
        self.env = env
        self.Q = DQN(k, num_actions)
        self.target_Q = DQN(k, num_actions)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.batch_size = batch_size

    def get_action(self, obs: np.ndarray):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.Q(torch.Tensor(obs))
            action = torch.argmax(q_values).item()
            return action

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        obs_batch, act_batch, reward_batch, next_obs_batch, done_batch = zip(*minibatch)

        obs_batch = torch.as_tensor(np.array(obs_batch), dtype=torch.float32)
        next_obs_batch = torch.as_tensor(np.array(next_obs_batch), dtype=torch.float32)
        act_batch = torch.as_tensor(act_batch, dtype=torch.int64)
        reward_batch = torch.as_tensor(reward_batch)
        done_batch = torch.as_tensor(done_batch)

        q_values = self.Q(obs_batch)
        q_values = q_values.gather(1, act_batch.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.Q(next_obs_batch)
            max_next_q_values = next_q_values.max(dim=1)[0]
            targets = (
                reward_batch + self.gamma * max_next_q_values * (~done_batch).float()
            )

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, targets)
        loss.backward()
        self.optimizer.step()

    def learn(self, episodes):
        epsilon_schedule = EpsilonDecaySchedule(
            self.epsilon, self.epsilon_min, self.epsilon_decay_steps
        )
        episode_rewards = []
        avg_episode_rewards = []

        for episode in range(1, episodes + 1):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.get_action(obs[np.newaxis])
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                self.memory.append([obs, action, reward, next_obs, done])
                if len(self.memory) >= 50000:
                    self.replay()

                obs = next_obs
                self.epsilon = epsilon_schedule.next_epsilon()

            logging.info(
                f"Episode: {episode}/{episodes}, Reward: {episode_reward}, Epsilon: {self.epsilon:.2}"
            )
            logging.info(
                f"Episode Frame: {info.episode_frame_number}, Total Frame: {info.frame_number}"
            )
            episode_rewards.append(episode_reward)
            if episode >= 10:
                avg_episode_rewards.append(np.mean(episode_rewards[-10:]))
            else:
                avg_episode_rewards.append(0)

            if episode % 25 == 0:
                self.save(f"model_params/{self.env.spec.name}.params.tmp")
                logging.info(f"Last 10 Average: {avg_episode_rewards[-1]}")
                logging.info("Model parameters saved.")

        self.save(f"model_params/{self.env.spec.name}.params")
        logging.info("Learning completed. Model parameters saved.")

    def save(self, path):
        torch.save(self.Q.state_dict(), path)

    def load(self, path):
        self.Q.load_state_dict(torch.load(path, weights_only=True))
