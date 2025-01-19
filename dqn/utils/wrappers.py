from collections import deque
import gymnasium as gym
import cv2
import numpy as np


class PreprocessFrameWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # cropped_frame = obs[34:194]
        # downsampled_frame = cv2.resize(cropped_frame, dsize=(84, 84))
        # downsampled_frame = cv2.resize(obs, dsize=(84, 84))
        return cv2.resize(obs, dsize=(84, 84))

class StackFramesWrapper(gym.Wrapper):
    def __init__(self, env, framestack):
        super().__init__(env)
        self.buffer = deque(maxlen=framestack)
        self.framestack = framestack

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        for _ in range(self.framestack):
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
