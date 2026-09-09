import sys
sys.path.insert(0, '.')
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import os

st.set_page_config(page_title="RL Trading Strategy", layout="wide")

st.title("Reinforcement Learning Trading Strategy")
st.markdown("**Q-Learning agent trained on Black-Scholes option pricing environment**")

st.sidebar.header("Options")
mode = st.sidebar.radio("Mode", ["View Pre-trained Results", "Train Demo (Fast)"])

if mode == "Train Demo (Fast)":
    st.info("Demo mode uses 50 episodes and 100 Monte Carlo paths. Takes 2 to 3 minutes.")
    if st.sidebar.button("Start Demo Training"):

        import src.config as cfg
        cfg.NUM_EPISODES = 50
        cfg.NUM_PATHS = 100
        cfg.NUM_STEPS = 252
        cfg.BATCH_SIZE = 32
        cfg.MEMORY_SIZE = 1000

        from src.environment import TradingEnvironment
        from src.agent import QLearningAgent
        from src.train import train
        from src.evaluate import evaluate
        from src.backtest import run_backtest

        with st.spinner("Training demo agent... please wait 2 to 3 minutes"):
            agent, rewards, epsilons = train()
            metrics = evaluate(agent, num_episodes=20)
            results = run_backtest(agent)

        min_len = min(len(results['rl']),
                      len(results['buy_and_hold']),
                      len(results['ma_crossover']))

        st.success("Demo training complete.")

        col1, col2, col3 = st.columns(3)
        col1.metric("RL Strategy Return", f"{(results['rl'][min_len-1]-1)*100:.2f}%")
        col2.metric("Buy and Hold Return", f"{(results['buy_and_hold'][min_len-1]-1)*100:.2f}%")
        col3.metric("MA Crossover Return", f"{(results['ma_crossover'][min_len-1]-1)*100:.2f}%")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.4f}")
        col2.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
        col3.metric("Volatility", f"{metrics['volatility']*100:.2f}%")
        col4.metric("Total Return", f"{metrics['total_return']:.4f}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].plot(results['rl'][:min_len], label='RL Strategy', color='blue', linewidth=2)
        axes[0].plot(results['buy_and_hold'][:min_len], label='Buy and Hold', color='red', linewidth=2)
        axes[0].plot(results['ma_crossover'][:min_len], label='MA Crossover', color='green', linewidth=2)
        axes[0].set_title('Demo Backtest — Cumulative Portfolio Value')
        axes[0].set_xlabel('Trading Days')
        axes[0].set_ylabel('Portfolio Value')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(rewards, alpha=0.5, color='blue', label='Episode Reward')
        window = 10
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axes[1].plot(range(window-1, len(rewards)), moving_avg,
                        color='red', linewidth=2, label=f'{window}-episode MA')
        axes[1].set_title('Training Rewards')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Total Reward')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        st.pyplot(fig)

    st.stop()

# --- Pre-trained results ---
st.markdown("""
### About This Project
This system uses **Reinforcement Learning (Q-Learning)** to learn optimal trading
and hedging strategies in a **Black-Scholes** simulated financial market.

The agent was trained on **1,000 Monte Carlo paths** over **500 episodes** —
each episode representing one full trading year of 252 days.
""")

if not os.path.exists('results.json'):
    st.error("Results file not found.")
    st.stop()

with open('results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

rewards = data['rewards']
epsilons = data['epsilons']
metrics = data['metrics']
backtest = data['backtest']

rl = np.array(backtest['rl'])
bh = np.array(backtest['buy_and_hold'])
ma = np.array(backtest['ma_crossover'])
prices = np.array(backtest['prices'])

st.subheader("Strategy Performance")
col1, col2, col3 = st.columns(3)
col1.metric("RL Strategy Return", f"{(rl[-1]-1)*100:.2f}%")
col2.metric("Buy and Hold Return", f"{(bh[-1]-1)*100:.2f}%")
col3.metric("MA Crossover Return", f"{(ma[-1]-1)*100:.2f}%")

st.subheader("Risk Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.4f}")
col2.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
col3.metric("Volatility", f"{metrics['volatility']*100:.2f}%")
col4.metric("Total Return", f"{metrics['total_return']:.4f}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Cumulative Portfolio Value")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rl, label='RL Strategy', color='blue', linewidth=2)
    ax.plot(bh, label='Buy and Hold', color='red', linewidth=2)
    ax.plot(ma, label='MA Crossover', color='green', linewidth=2)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

with col2:
    st.subheader("Simulated Stock Price Path")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(prices, color='orange', linewidth=1.5)
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Stock Price")
    ax.grid(True)
    st.pyplot(fig)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Training Rewards per Episode")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.5, color='blue', label='Episode Reward')
    window = 20
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), moving_avg,
                color='red', linewidth=2, label=f'{window}-episode MA')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

with col2:
    st.subheader("Epsilon Decay")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epsilons, color='green')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.grid(True)
    st.pyplot(fig)

st.subheader("How It Works")
st.markdown("""
| Component | Description |
|---|---|
| **Environment** | Black-Scholes Monte Carlo simulation — 1,000 paths, 252 steps |
| **Agent** | Q-Learning with epsilon-greedy exploration |
| **State** | Normalised price, time remaining, moneyness (S/K) |
| **Actions** | 11 hedge ratios from -1.0 to +1.0 |
| **Reward** | P&L from hedge position minus option payoff at maturity |
| **Benchmarks** | Buy and Hold, Moving Average Crossover |
""")

st.subheader("Tech Stack")
st.markdown("""
- **Python** — modular src/ architecture across 6 files
- **NumPy** — Monte Carlo simulation and Q-learning
- **Streamlit** — production deployment
- **pytest** — 18 automated tests
- **GitHub** — version control
""")