import numpy as np
import sys
sys.path.insert(0, '.')
from src.environment import TradingEnvironment
from src.agent import QLearningAgent
from src.config import NUM_EPISODES, RANDOM_SEED

np.random.seed(RANDOM_SEED)


def train():
    env = TradingEnvironment()
    agent = QLearningAgent()

    episode_rewards = []
    episode_epsilons = []

    print("Starting training...")
    print(f"Episodes: {NUM_EPISODES}")
    print("-" * 40)

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action_idx = agent.choose_action(state)
            action_value = agent.get_action_value(action_idx)
            next_state, reward, done = env.step(action_value)
            agent.remember(state, action_idx, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_epsilons.append(agent.get_epsilon())

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}/{NUM_EPISODES} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Epsilon: {agent.get_epsilon():.4f}")

    print("-" * 40)
    print("Training complete.")
    return agent, episode_rewards, episode_epsilons


if __name__ == "__main__":
    agent, rewards, epsilons = train()