import sys
sys.path.insert(0, '.')
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from src.environment import TradingEnvironment
from src.agent import QLearningAgent
from src.train import train
from src.evaluate import evaluate, print_metrics
from src.backtest import run_backtest, print_backtest_results
from src.config import NUM_EPISODES

st.set_page_config(page_title="RL Trading Strategy", layout="wide")

st.title("Reinforcement Learning Trading Strategy")
st.markdown("**Q-Learning agent trained on Black-Scholes option pricing environment**")

st.sidebar.header("Parameters")
episodes = st.sidebar.slider("Training Episodes", 50, 500, 200, 50)
eval_episodes = st.sidebar.slider("Evaluation Episodes", 10, 100, 50, 10)

if st.sidebar.button("Train and Evaluate"):

    with st.spinner("Training RL agent..."):
        agent, rewards, epsilons = train()

    st.success("Training complete.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Rewards")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(rewards, alpha=0.6, color='blue', label='Episode Reward')
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

    st.subheader("Evaluation Metrics")
    metrics = evaluate(agent, num_episodes=eval_episodes)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{metrics['total_return']:.4f}")
    col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.4f}")
    col3.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
    col4.metric("Volatility", f"{metrics['volatility']*100:.2f}%")

    st.subheader("Backtest — Strategy Comparison")
    results = run_backtest(agent)

    min_len = min(len(results['rl']), len(results['buy_and_hold']), len(results['ma_crossover']))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(results['rl'][:min_len], label='RL Strategy', color='blue', linewidth=2)
    axes[0].plot(results['buy_and_hold'][:min_len], label='Buy and Hold', color='red', linewidth=2)
    axes[0].plot(results['ma_crossover'][:min_len], label='MA Crossover', color='green', linewidth=2)
    axes[0].set_title('Cumulative Portfolio Value')
    axes[0].set_ylabel('Portfolio Value')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(results['prices'], color='orange', linewidth=1.5)
    axes[1].set_title('Simulated Stock Price Path')
    axes[1].set_xlabel('Trading Days')
    axes[1].set_ylabel('Price')
    axes[1].grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Strategy Performance Summary")
    rl_return = (results['rl'][min_len-1] - 1) * 100
    bh_return = (results['buy_and_hold'][min_len-1] - 1) * 100
    ma_return = (results['ma_crossover'][min_len-1] - 1) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("RL Strategy Return", f"{rl_return:.2f}%")
    col2.metric("Buy and Hold Return", f"{bh_return:.2f}%")
    col3.metric("MA Crossover Return", f"{ma_return:.2f}%")

else:
    st.info("Set your parameters in the sidebar and click Train and Evaluate to start.")
    st.markdown("""
    ### What this app does
    - Trains a Q-Learning agent on a Black-Scholes trading environment
    - Agent learns optimal hedge positions to maximise returns
    - Evaluates performance using Sharpe ratio, drawdown and volatility
    - Compares RL strategy against Buy and Hold and MA Crossover benchmarks
    
    ### Key concepts
    - **Q-Learning** — agent learns by trial and error
    - **Black-Scholes** — mathematical model for option pricing
    - **Sharpe Ratio** — return per unit of risk
    - **Mean Reversion** — prices tend to return to their average
    """)# placeholder