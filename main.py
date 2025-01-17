import logging
from wrappers import *
from dqn_atari import DQNAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="dqn.log",
    encoding="utf-8",
    filemode="a",
)


def play(agent):
    # agent.epsilon = 0
    # agent.epsilon_min = 0
    while True:
        obs, _ = env.reset()

        episode_over = False
        while not episode_over:
            action = agent.get_action(obs)
            obs, _, terminated, truncated, _ = agent.env.step(action)

            episode_over = terminated or truncated

        time.sleep(1)


def test(agent, episodes=200):
    agent.epsilon = 0
    agent.epsilon_min = 0

    scores = []
    scores_avg = []

    plt.ion()
    fig, ax = plt.subplots()

    for episode in range(episodes):
        print(f"episode {episode}")
        state, _ = agent.env.reset()

        done = False
        score = 0
        for t in range(500):
            action = agent.get_action(state)
            state, reward, terminated, truncated, _ = agent.env.step(action)
            score += reward

        print(f"episode: {episode}/{episodes}, score: {score}")
        scores.append(score)
        scores_avg.append(np.mean(scores[-10:]))

        ax.cla()
        ax.plot(scores)
        ax.plot(scores_avg)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score")
        fig.canvas.flush_events()

    print(f"average: {np.mean(scores)}")


def main():
    pass


if __name__ == "__main__":
    env = gym.make("Pong-v4", obs_type="grayscale", render_mode="rgb_array")
    # env = gym.make("Pong-v4", obs_type="grayscale", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env=env,
        video_folder="videos",
        name_prefix=f"{env.spec.name}",
        episode_trigger=lambda x: x % 25 == 0,
    )
    env = DownsampleFrameWrapper(env)
    env = StackFrameWrapper(env, 4)
    env = MagnifyRewardWrapper(env)

    agent = DQNAgent(env)
    # agent.load("model_params/Pong.params.save")
    # agent.epsilon = 0.1
    # agent.epsilon_min = 0.1

    logging.info("Begin training")
    agent.train(episodes=10000)
