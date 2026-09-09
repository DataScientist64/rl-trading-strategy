import numpy as np
import random
from collections import deque
from src.config import (
    GAMMA, LEARNING_RATE, EPSILON, EPSILON_MIN,
    EPSILON_DECAY, BATCH_SIZE, MEMORY_SIZE, RANDOM_SEED
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class QLearningAgent:

    def __init__(self, state_size=3, action_size=11):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.lr = LEARNING_RATE

        # Actions: hedge ratios from -1.0 to 1.0
        self.actions = np.linspace(-1.0, 1.0, action_size)

        # Q-table: simplified as a dictionary
        self.q_table = {}

    def _get_state_key(self, state):
        return tuple(round(s, 2) for s in state)

    def _get_q_values(self, state):
        key = self._get_state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)
        return self.q_table[key]

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self._get_q_values(state))

    def get_action_value(self, action_idx):
        return self.actions[action_idx]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = random.sample(self.memory, BATCH_SIZE)

        for state, action, reward, next_state, done in batch:
            q_values = self._get_q_values(state)
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self._get_q_values(next_state))
            q_values[action] = q_values[action] + self.lr * (target - q_values[action])
            self.q_table[self._get_state_key(state)] = q_values

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_epsilon(self):
        return self.epsilon