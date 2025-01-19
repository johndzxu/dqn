import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from gymnasium.wrappers import RecordEpisodeStatistics

from utils.memories import ReplayBuffer
from utils.schedule import EpsilonDecaySchedule
from dqn import DQN

SAVE_EVERY_N_STEPS = 100000


class DQNAgent:
    def __init__(
        self,
        env,
        frame_stack=4,
        num_actions=6,
        memory_size=1000000,
        learning_rate=0.00025,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay_steps=1000000,
        batch_size=32,
        gamma=0.99,
        learning_freq=4,
        target_update_freq=10000,
    ):
        self.env = RecordEpisodeStatistics(env)
        self.Q = DQN(frame_stack, num_actions)
        self.target_Q = DQN(frame_stack, num_actions)
        self.memory = ReplayBuffer(memory_size)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.RMSprop(
            self.Q.parameters(),
            lr=learning_rate,
            alpha=0.95,
            eps=0.01,
        )

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_freq = learning_freq
        self.target_update_freq = target_update_freq

        self.steps = 0
        self.training_steps = 0
        self.update_target_Q()

    def get_action(self, obs):
        if np.random.random() > self.epsilon:
            obs = torch.Tensor(obs).unsqueeze(0)
            return torch.argmax(self.Q(obs), dim=1).item()
        else:
            return self.env.action_space.sample()

    def update_target_Q(self):
        self.target_Q.load_state_dict(self.Q.state_dict())

    def replay(self):
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
        loss = self.criterion(Q_values, target_Q_values)
        loss.backward()
        self.optimizer.step()
        self.training_steps += 1

        if self.training_steps == 1:
            logging.info("Replay started")

    def learn(self, episodes):
        epsilon_schedule = EpsilonDecaySchedule(
            self.epsilon, self.epsilon_min, self.epsilon_decay_steps
        )

        for episode in range(1, episodes + 1):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.memory.store([obs, action, reward, next_obs, done])

                if self.memory.size >= 50000 and self.steps % self.learning_freq == 0:
                    self.replay()

                if self.training_steps % self.target_update_freq == 0:
                    self.update_target_Q()

                self.epsilon = epsilon_schedule.next_epsilon()
                self.steps += 1
                episode_reward += reward
                done = terminated or truncated
                obs = next_obs

                # Saving
                if self.steps % SAVE_EVERY_N_STEPS == 0:
                    logging.info("Saving model parameters...")
                    self.save(f"model_params/{self.env.spec.name}.params.tmp")
                    logging.info("Model parameters saved.")

                
            
            # Logging       
            logging.info(f"[Episode {episode}] "
                         f"reward: {self.env.episode_returns}, "
                         f"length: {self.env.episode_lengths}, "
                         f"epsilon: {self.epsilon:.2}")
            logging.info(f"step: {self.steps}, "
                         f"training step: {self.training_steps}")
            if episode % 10 == 0:
                logging.info(f"Mean reward (last 100): {np.mean(self.env.return_queue)}")
                logging.info(f"Mean length (last 100): {np.mean(self.env.length_queue)}")


        # Saving
        logging.info("Saving model parameters...")
        self.save(f"model_params/{self.env.spec.name}.params")
        logging.info("Learning completed. Model parameters saved.")

    def save(self, path):
        torch.save(self.Q.state_dict(), path)

    def load(self, path):
        self.Q.load_state_dict(torch.load(path, weights_only=True))
