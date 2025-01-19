class EpsilonDecaySchedule:
    def __init__(self, start, end, steps):
        self.step = 0
        self.schedule = np.linspace(start, end, steps)

    def next_epsilon(self):
        self.step += 1
        return self.schedule[min(self.step, len(self.schedule) - 1)]
