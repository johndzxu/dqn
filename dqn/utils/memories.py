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


class Reservoir:
    pass
