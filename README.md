# DRL Portfolio Allocation Reproduction

This repository is a practical reproduction scaffold for the paper:

`Deep Reinforcement Learning for Optimal Portfolio Allocation: A Comparative Study with Mean-Variance Optimization`

For the current end-to-end reproduction workflow, see [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md).

The project is designed with the following priority:

1. Run public author code if it becomes available.
2. Fall back to a faithful reimplementation of the paper's core modules.
3. Reproduce the paper's main DRL-vs-MVO comparison as closely as possible.

## Current status

No official public repository for the paper was clearly identifiable during the initial search, so this repo starts from a clean reimplementation based on the paper PDF.

Implemented:

- Yahoo Finance data download
- Feature engineering for asset log returns and volatility regime features
- Portfolio Gymnasium environment with long-only weights and cash
- Differential Sharpe Ratio reward
- PPO training through Stable-Baselines3
- Mean-Variance Optimization baseline with Ledoit-Wolf shrinkage covariance
- Unified backtest metrics
- Smoke-test command for quick validation

## Paper details currently encoded

- Daily data
- 60-day lookback
- Long-only portfolio with cash
- PPO as the DRL optimizer
- Differential Sharpe Ratio reward
- Sliding-window experiments with:
  - 5 years train
  - 1 year validation
  - 1 year test

## Important assumptions

Some implementation details are explicit in the paper, while others are not fully specified.
This repo currently uses the following assumptions:

- Market data source: Yahoo Finance
- Default asset universe: 11 liquid US sector ETF proxies with long history, plus cash
- Market regime features: `vol20`, `vol20 / vol60`, and `VIX`
- Whole-share rebalancing is approximated with continuous weights during training and exact whole-share execution during backtests
- The PPO network defaults to a compact MLP for reproducibility and reasonable runtime on a local machine

The paper states "S&P 500 sector indices"; if we later identify the exact ticker set used by the authors, we should replace the current proxy universe.

## Quick start

1. Create a virtual environment if you want isolation.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run a smoke test:

```bash
python -m drl_repro.cli smoke
```

4. Download the full dataset:

```bash
python -m drl_repro.cli download
```

5. Run a small rolling experiment:

```bash
python -m drl_repro.cli experiment --max-windows 1 --train-steps 50000
```

## Repo layout

- `drl_repro/config.py`: experiment configuration
- `drl_repro/data.py`: data download and feature preparation
- `drl_repro/env.py`: Gymnasium portfolio environment
- `drl_repro/mvo.py`: mean-variance baseline
- `drl_repro/metrics.py`: performance metrics
- `drl_repro/ppo_agent.py`: PPO training and action inference helpers
- `drl_repro/experiment.py`: rolling-window experiment runner
- `drl_repro/cli.py`: command-line entry point

## Next steps

- Confirm exact asset symbols used in the paper
- Match PPO hyperparameters more tightly to the paper
- Add multi-seed evaluation over all 10 windows
- Generate publication-style figures and tables
