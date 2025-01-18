from collections import deque
import gymnasium as gym
import cv2
import numpy as np


class PreprocessFrameWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def preprocess(self, obs):
        cropped_frame = obs[34:194]
        downsampled_frame = cv2.resize(cropped_frame, dsize=(84, 84))
        return downsampled_frame

    def reset(self):
        obs, info = self.env.reset()
        return self.preprocess(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.preprocess(obs), reward, terminated, truncated, info


class StackFramesWrapper(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.buffer = deque(maxlen=k)
        self.k = k

    def reset(self):
        obs, info = self.env.reset()

        for _ in range(self.k):
            self.buffer.append(obs)
        return np.stack(self.buffer), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.buffer.append(obs)

        return np.stack(self.buffer), reward, terminated, truncated, info


class ModifyRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward == 1:
            reward = 2
        return obs, reward, terminated, truncated, info
