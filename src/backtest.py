import numpy as np
import sys
sys.path.insert(0, '.')
from src.environment import TradingEnvironment
from src.agent import QLearningAgent
from src.config import RANDOM_SEED

np.random.seed(RANDOM_SEED)


def run_backtest(agent):
    env = TradingEnvironment()
    state = env.reset()
    done = False

    rl_portfolio = [1.0]
    bh_portfolio = [1.0]
    ma_portfolio = [1.0]

    prices = []
    rl_rewards = []

    while not done:
        current_price = np.mean(env.stock_prices[env.current_step])
        prices.append(current_price)

        # RL strategy
        action_idx = np.argmax(agent._get_q_values(state))
        action_value = agent.get_action_value(action_idx)
        next_state, reward, done = env.step(action_value)
        rl_rewards.append(reward)

        rl_portfolio.append(rl_portfolio[-1] + reward)
        state = next_state

    # Buy and hold strategy
    price_array = np.array(prices)
    bh_returns = np.diff(price_array) / price_array[:-1]
    bh_cum = np.cumprod(1 + bh_returns)
    bh_portfolio = np.concatenate([[1.0], bh_cum])

    # Moving average crossover strategy
    short_ma = _moving_average(price_array, 10)
    long_ma = _moving_average(price_array, 30)
    ma_signals = np.where(short_ma > long_ma, 1, -1)
    ma_returns = ma_signals[:-1] * bh_returns
    ma_cum = np.cumprod(1 + ma_returns)
    ma_portfolio = np.concatenate([[1.0], ma_cum])

    return {
        'rl': np.array(rl_portfolio),
        'buy_and_hold': bh_portfolio,
        'ma_crossover': ma_portfolio,
        'prices': price_array
    }


def _moving_average(prices, window):
    ma = np.zeros(len(prices))
    for i in range(len(prices)):
        if i < window:
            ma[i] = np.mean(prices[:i+1])
        else:
            ma[i] = np.mean(prices[i-window:i])
    return ma


def print_backtest_results(results):
    rl = results['rl']
    bh = results['buy_and_hold']
    ma = results['ma_crossover']

    min_len = min(len(rl), len(bh), len(ma))

    print("\n--- Backtest Results ---")
    print(f"{'Strategy':<20} {'Total Return':>14} {'Final Value':>12}")
    print("-" * 50)
    print(f"{'RL Strategy':<20} {(rl[min_len-1]-1)*100:>13.2f}% {rl[min_len-1]:>12.4f}")
    print(f"{'Buy and Hold':<20} {(bh[min_len-1]-1)*100:>13.2f}% {bh[min_len-1]:>12.4f}")
    print(f"{'MA Crossover':<20} {(ma[min_len-1]-1)*100:>13.2f}% {ma[min_len-1]:>12.4f}")