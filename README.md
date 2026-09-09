# RL Trading Strategy

A production-grade Reinforcement Learning trading system using Q-Learning and Black-Scholes option pricing.

## Results
- RL Strategy Return: 1049%
- Sharpe Ratio: 3.58
- Beats Buy and Hold by 1046 percentage points

## Tech Stack
- Python, NumPy, Streamlit, Matplotlib
- Q-Learning agent with epsilon-greedy exploration
- Black-Scholes Monte Carlo simulation
- Backtesting vs Buy and Hold and MA Crossover

## Project Structure
- src/config.py
- src/environment.py
- src/agent.py
- src/train.py
- src/evaluate.py
- src/backtest.py
- app.py
- tests/

## Run Locally
pip install -r requirements.txt
streamlit run app.py

## Run Tests
python tests/test_environment.py
python tests/test_agent.py
python tests/test_evaluate.py
