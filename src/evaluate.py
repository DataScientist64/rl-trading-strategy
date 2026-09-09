import numpy as np
import sys
sys.path.insert(0, '.')
from src.environment import TradingEnvironment
from src.agent import QLearningAgent
from src.config import PERIODS_PER_YEAR


def evaluate(agent, num_episodes=100):
    env = TradingEnvironment()
    all_returns = []
    all_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        portfolio_values = [1.0]

        while not done:
            action_idx = np.argmax(agent._get_q_values(state))
            action_value = agent.get_action_value(action_idx)
            next_state, reward, done = env.step(action_value)
            total_reward += reward
            portfolio_values.append(1.0 + total_reward)
            state = next_state

        all_rewards.append(total_reward)
        portfolio_array = np.array(portfolio_values)
        daily_returns = np.diff(portfolio_array) / portfolio_array[:-1]
        all_returns.append(daily_returns)

    all_returns = np.concatenate(all_returns)

    metrics = {
        'total_return': np.mean(all_rewards),
        'sharpe_ratio': _sharpe_ratio(all_returns),
        'max_drawdown': _max_drawdown(all_returns),
        'volatility': np.std(all_returns) * np.sqrt(PERIODS_PER_YEAR)
    }

    return metrics


def _sharpe_ratio(returns):
    if np.std(returns) == 0:
        return 0
    return np.sqrt(PERIODS_PER_YEAR) * np.mean(returns) / np.std(returns)


def _max_drawdown(returns):
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return np.min(drawdown)


def print_metrics(metrics):
    print("\n--- Evaluation Metrics ---")
    print(f"Total Return:  {metrics['total_return']:.4f}")
    print(f"Sharpe Ratio:  {metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown:  {metrics['max_drawdown']*100:.2f}%")
    print(f"Volatility:    {metrics['volatility']*100:.2f}%")