import sys
sys.path.insert(0, '.')
import numpy as np
from src.evaluate import evaluate, _sharpe_ratio, _max_drawdown
from src.agent import QLearningAgent


def test_sharpe_ratio_positive():
    returns = np.array([0.01, 0.02, 0.01, 0.03, 0.01])
    sr = _sharpe_ratio(returns)
    assert sr > 0
    print("PASSED: test_sharpe_ratio_positive")


def test_sharpe_ratio_zero_std():
    returns = np.zeros(10)
    sr = _sharpe_ratio(returns)
    assert sr == 0
    print("PASSED: test_sharpe_ratio_zero_std")


def test_max_drawdown_negative():
    returns = np.array([0.01, -0.05, 0.02, -0.03, 0.01])
    md = _max_drawdown(returns)
    assert md < 0
    print("PASSED: test_max_drawdown_negative")


def test_evaluate_returns_metrics():
    agent = QLearningAgent()
    metrics = evaluate(agent, num_episodes=5)
    assert 'total_return' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'volatility' in metrics
    print("PASSED: test_evaluate_returns_metrics")


def test_volatility_positive():
    agent = QLearningAgent()
    metrics = evaluate(agent, num_episodes=5)
    assert metrics['volatility'] >= 0
    print("PASSED: test_volatility_positive")


if __name__ == "__main__":
    test_sharpe_ratio_positive()
    test_sharpe_ratio_zero_std()
    test_max_drawdown_negative()
    test_evaluate_returns_metrics()
    test_volatility_positive()
    print("\nAll evaluation tests passed.")