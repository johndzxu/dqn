import numpy as np


class ReplayBuffer:
    def __init__(self, memory_size):
        self.obs_buf = np.zeros([memory_size, 4, 84, 84], dtype=np.uint8)
        self.next_obs_buf = np.zeros([memory_size, 4, 84, 84], dtype=np.uint8)
        self.rew_buf = np.zeros([memory_size], dtype=np.float32)
        self.act_buf = np.zeros([memory_size], dtype=np.int64)
        self.done_buf = np.zeros([memory_size], dtype=np.bool)

        self.memory_size = memory_size
        self.idx = 0
        self.size = 0

    def store(self, transition):
        (
            self.obs_buf[self.idx],
            self.act_buf[self.idx],
            self.rew_buf[self.idx],
            self.next_obs_buf[self.idx],
            self.done_buf[self.idx],
        ) = transition

        self.idx = (self.idx + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, batch_size)
        return (
            self.obs_buf[idxs],
            self.act_buf[idxs],
            self.rew_buf[idxs],
            self.next_obs_buf[idxs],
            self.done_buf[idxs],
        )


class ReservoirBuffer:
    def __init__(self, capacity):
        self.obs_buf = np.ndarray(capacity)
        self.act_buf = np.ndarray(capacity)
        self.capacity = capacity
        self.W = np.exp(np.log(np.random.random()) / np.log(1 - self.W)) + 1
        self.idx = 0

    def store(self, transition):
        obs, act = transition
        if self.idx < self.capactiy:
            self.obs_buf[self.idx], self.act_buf[j] = obs, act
        else:
            j = np.random.randint(0, self.idx + 1)
            if j < self.capacity:
                self.obs_buf[j], self.act_buf[j] = obs, act
        self.idx += 1

    def sample(self, num_samples):
        if self.idx < num_samples:
            raise ValueError()
        else:
            idxs = np.random.randint(0, self.idx, num_samples)
            return self.obs_buf[idxs], self.act_buf[idxs]


class PriorityReplayBuffer:
    def __init__(self):
        pass
