import sys
sys.path.insert(0, '.')
import numpy as np
import json
from src.train import train
from src.evaluate import evaluate
from src.backtest import run_backtest

print("Training with full settings...")
print("This will take 10-20 minutes. Please wait.")
print("-" * 50)

# Train agent
agent, rewards, epsilons = train()

# Evaluate
print("\nEvaluating...")
metrics = evaluate(agent, num_episodes=100)

# Backtest
print("Running backtest...")
results = run_backtest(agent)

min_len = min(len(results['rl']), 
              len(results['buy_and_hold']), 
              len(results['ma_crossover']))

# Save all results
save_data = {
    'rewards': rewards,
    'epsilons': epsilons,
    'metrics': {
        'total_return': float(metrics['total_return']),
        'sharpe_ratio': float(metrics['sharpe_ratio']),
        'max_drawdown': float(metrics['max_drawdown']),
        'volatility': float(metrics['volatility'])
    },
    'backtest': {
        'rl': results['rl'][:min_len].tolist(),
        'buy_and_hold': results['buy_and_hold'][:min_len].tolist(),
        'ma_crossover': results['ma_crossover'][:min_len].tolist(),
        'prices': results['prices'].tolist()
    }
}

with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(save_data, f)

print("\nResults saved to results.json")
print("\n--- Final Results ---")
print(f"Sharpe Ratio:     {metrics['sharpe_ratio']:.4f}")
print(f"Total Return:     {metrics['total_return']:.4f}")
print(f"Max Drawdown:     {metrics['max_drawdown']*100:.2f}%")
print(f"RL Return:        {(results['rl'][min_len-1]-1)*100:.2f}%")
print(f"Buy and Hold:     {(results['buy_and_hold'][min_len-1]-1)*100:.2f}%")
print(f"MA Crossover:     {(results['ma_crossover'][min_len-1]-1)*100:.2f}%")# placeholder