import sys
sys.path.insert(0, '.')
import numpy as np
from src.agent import QLearningAgent


def test_agent_initializes():
    agent = QLearningAgent()
    assert agent.state_size == 3
    assert agent.action_size == 11
    print("PASSED: test_agent_initializes")


def test_actions_range():
    agent = QLearningAgent()
    assert agent.actions[0] == -1.0
    assert agent.actions[-1] == 1.0
    print("PASSED: test_actions_range")


def test_choose_action_valid():
    agent = QLearningAgent()
    state = [1.0, 0.5, 1.05]
    action = agent.choose_action(state)
    assert 0 <= action < agent.action_size
    print("PASSED: test_choose_action_valid")


def test_remember_stores_experience():
    agent = QLearningAgent()
    state = [1.0, 0.5, 1.05]
    next_state = [1.01, 0.49, 1.06]
    agent.remember(state, 0, 0.5, next_state, False)
    assert len(agent.memory) == 1
    print("PASSED: test_remember_stores_experience")


def test_epsilon_decays():
    agent = QLearningAgent()
    initial_epsilon = agent.get_epsilon()
    state = [1.0, 0.5, 1.05]
    next_state = [1.01, 0.49, 1.06]
    for _ in range(100):
        agent.remember(state, 0, 0.1, next_state, False)
    agent.replay()
    assert agent.get_epsilon() <= initial_epsilon
    print("PASSED: test_epsilon_decays")


def test_q_table_updates():
    agent = QLearningAgent()
    state = [1.0, 0.5, 1.05]
    next_state = [1.01, 0.49, 1.06]
    for _ in range(100):
        agent.remember(state, 5, 0.1, next_state, False)
    agent.replay()
    assert len(agent.q_table) > 0
    print("PASSED: test_q_table_updates")


if __name__ == "__main__":
    test_agent_initializes()
    test_actions_range()
    test_choose_action_valid()
    test_remember_stores_experience()
    test_epsilon_decays()
    test_q_table_updates()
    print("\nAll agent tests passed.")