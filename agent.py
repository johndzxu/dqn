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
        epsilon_min=0.05,
        epsilon_decay=0.95,
        memory_size=100000,
        lr=0.001,
        gamma=0.99,
        batch_size=32,
    ):
        self.env = env
        self.q_net = DQN().to(device)
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.k = 4

    def get_action(self, history):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.q_net(history)
            action = torch.argmax(q_values).item()
            return action

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)

        histories, actions, rewards, next_histories, dones = zip(*minibatch)

        histories_t = torch.stack(histories).to(device)
        actions_t = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards_t = torch.FloatTensor(rewards).to(device)
        next_histories_t = torch.stack(next_histories).to(device)
        dones_t = torch.FloatTensor(dones).to(device)

        q_values = self.q_net(histories_t)
        q_values = q_values.gather(1, actions_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.q_net(next_histories_t)
            max_next_q_values = next_q_values.max(dim=1)[0]
            targets = rewards_t + self.gamma * max_next_q_values * (1 - dones_t)

        self.optimizer.zero_grad()
        loss = self.criterion(q_values, targets)
        loss.backward()
        self.optimizer.step()

    def train(self, episodes):
        epsilon_schedule = EpsilonDecaySchedule(self.epsilon, self.epsilon_min, 1000)
        scores = []
        scores_avg = []

        # plt.ion()
        # fig, ax = plt.subplots()

        for episode in range(1, episodes + 1):
            sequence = []
            state, _ = self.env.reset()

            sequence.append(state)

            done = False
            score = 0

            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                score += reward

                sequence.append(next_state)

                self.memory.append([state, action, reward, next_state, (reward != 0)])
                if len(self.memory) >= self.batch_size:
                    self.replay()

                state = next_state

            self.epsilon = epsilon_schedule.next_epsilon()

            logging.info(
                f"episode: {episode}/{episodes}, score: {score}, e: {self.epsilon:.2}"
            )
            scores.append(score)
            scores_avg.append(np.mean(scores[-10:]))

            # ax.cla()
            # ax.plot(scores)
            # ax.plot(scores_avg)
            # ax.set_xlabel("Training Episode")
            # ax.set_ylabel("Score")
            # fig.canvas.flush_events()
            if (episode + 1) % 25 == 0:
                torch.save(
                    self.q_net.state_dict(),
                    f"model_params/{self.env.spec.name}2.params.save",
                )
                logging.info("Model parameters saved.")

        torch.save(
            self.q_net.state_dict(), f"model_params/{self.env.spec.name}2.params"
        )
        logging.info("Training completed. Model parameters saved.")

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def load(self, model):
        self.q_net.load_state_dict(torch.load(model, weights_only=True))
        self.q_net.eval()
