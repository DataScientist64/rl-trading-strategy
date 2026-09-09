import numpy as np
from src.config import S0, STRIKE, VOLATILITY, T, RISK_FREE_RATE, MU, NUM_STEPS, NUM_PATHS, RANDOM_SEED


class TradingEnvironment:

    def __init__(self):
        self.s0 = S0
        self.strike = STRIKE
        self.vol = VOLATILITY
        self.T = T
        self.r = RISK_FREE_RATE
        self.mu = MU
        self.num_steps = NUM_STEPS
        self.num_paths = NUM_PATHS
        self.dt = self.T / self.num_steps
        self.gamma = np.exp(-self.r * self.dt)
        self.reset()

    def reset(self):
        np.random.seed(RANDOM_SEED)
        self.current_step = 0
        self.stock_prices = self._generate_paths()
        self.current_price = self.stock_prices[self.current_step]
        return self._get_state()

    def _generate_paths(self):
        prices = np.zeros((self.num_steps + 1, self.num_paths))
        prices[0] = self.s0
        for t in range(1, self.num_steps + 1):
            Z = np.random.standard_normal(self.num_paths)
            prices[t] = prices[t - 1] * np.exp(
                (self.mu - 0.5 * self.vol ** 2) * self.dt +
                self.vol * np.sqrt(self.dt) * Z
            )
        return prices

    def _get_state(self):
        price = np.mean(self.stock_prices[self.current_step])
        time_remaining = 1 - self.current_step / self.num_steps
        moneyness = price / self.strike
        return [price / self.s0, time_remaining, moneyness]

    def step(self, action):
        current_prices = self.stock_prices[self.current_step]
        next_prices = self.stock_prices[min(self.current_step + 1, self.num_steps)]
        delta_S = next_prices - np.exp(self.r * self.dt) * current_prices
        pnl = action * np.mean(delta_S)
        if self.current_step == self.num_steps - 1:
            payoff = np.mean(np.maximum(self.strike - next_prices, 0))
            reward = pnl - payoff
        else:
            reward = pnl
        self.current_step += 1
        done = self.current_step >= self.num_steps
        return self._get_state(), reward, done

    def get_option_payoff(self):
        return np.mean(np.maximum(self.strike - self.stock_prices[-1], 0))