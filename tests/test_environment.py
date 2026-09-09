import sys
sys.path.insert(0, '.')
import numpy as np
from src.environment import TradingEnvironment
from src.config import S0, STRIKE, NUM_STEPS, NUM_PATHS


def test_environment_initializes():
    env = TradingEnvironment()
    assert env.s0 == S0
    assert env.strike == STRIKE
    print("PASSED: test_environment_initializes")


def test_reset_returns_state():
    env = TradingEnvironment()
    state = env.reset()
    assert len(state) == 3
    print("PASSED: test_reset_returns_state")


def test_stock_prices_shape():
    env = TradingEnvironment()
    env.reset()
    assert env.stock_prices.shape == (NUM_STEPS + 1, NUM_PATHS)
    print("PASSED: test_stock_prices_shape")


def test_stock_prices_positive():
    env = TradingEnvironment()
    env.reset()
    assert np.all(env.stock_prices > 0)
    print("PASSED: test_stock_prices_positive")


def test_step_returns_correct_format():
    env = TradingEnvironment()
    env.reset()
    next_state, reward, done = env.step(0.5)
    assert len(next_state) == 3
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    print("PASSED: test_step_returns_correct_format")


def test_episode_ends_at_num_steps():
    env = TradingEnvironment()
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done = env.step(0.0)
        steps += 1
    assert steps == NUM_STEPS
    print("PASSED: test_episode_ends_at_num_steps")


def test_option_payoff_positive():
    env = TradingEnvironment()
    env.reset()
    payoff = env.get_option_payoff()
    assert payoff >= 0
    print("PASSED: test_option_payoff_positive")


if __name__ == "__main__":
    test_environment_initializes()
    test_reset_returns_state()
    test_stock_prices_shape()
    test_stock_prices_positive()
    test_step_returns_correct_format()
    test_episode_ends_at_num_steps()
    test_option_payoff_positive()
    print("\nAll environment tests passed.")