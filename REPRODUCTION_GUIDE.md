## Reproduction Guide

This guide describes how to reproduce the paper:

`Deep Reinforcement Learning for Optimal Portfolio Allocation: A Comparative Study with Mean-Variance Optimization`

It covers:

1. What this repo currently matches from the paper
2. The recommended commands to run
3. What outputs are produced
4. The main remaining gaps versus the paper

## Current Scope

This repo now implements the paper's core experiment structure closely enough to support a serious reproduction attempt:

- Daily data from Yahoo Finance
- S&P 500 sector index style Yahoo symbols as the default asset universe
- `^GSPC` as the market index and `^VIX` as the volatility index
- 60-day lookback observations
- Long-only portfolio weights with cash
- Differential Sharpe Ratio reward
- PPO with Stable-Baselines3
- 10 rolling windows with:
  - 5 years train
  - 1 year validation / burn year
  - 1 year out-of-sample test
- 5 training seeds per window
- Validation-based model selection
- Warm-starting each new window from the previous window's best policy
- Whole-share execution with leftover cash
- MVO baseline with Ledoit-Wolf covariance shrinkage
- Summary tables, turnover statistics, and paper-style figures

## Experiment Protocol

The repo is configured around the paper's rolling evaluation design:

- First training window: `2006-01-01` to `2011-01-01`
- First validation / burn year: `2011`
- First test year: `2012`
- Final test year: `2021`

For each test year:

1. Train PPO on the 5-year training block
2. Evaluate candidate agents on the validation year
3. Select the best seed by validation reward
4. Backtest the selected DRL policy on the held-out test year
5. Backtest MVO on the same test year using rolling 60-day estimates

## Recommended Environment

The commands below assume your conda environment is `DRL`.

If you want to activate it first:

```bash
conda activate DRL
```

Or run commands without activating:

```bash
conda run --no-capture-output -n DRL <command>
```

## Recommended Workflow

### 1. Quick smoke test

Use this to verify the full pipeline is wired correctly before starting a long run:

```bash
conda run --no-capture-output -n DRL python -m drl_repro.cli smoke
```

This uses a short training budget and only one rolling window.

### 2. Full data download

```bash
conda run --no-capture-output -n DRL python -m drl_repro.cli download
```

This will create a cached market data file under `artifacts/data/`.

### 3. Small experimental run

Use this first if you want to test the full output pipeline without committing to the full paper-scale training budget:

```bash
conda run --no-capture-output -n DRL python -m drl_repro.cli experiment --max-windows 1 --train-steps 50000
```

### 4. Full rolling experiment

The default config is paper-oriented and uses `7.5M` timesteps per training round:

```bash
conda run --no-capture-output -n DRL python -m drl_repro.cli experiment
```

This is the main reproduction command if you want the full rolling study.

### 5. Build summary tables

```bash
conda run --no-capture-output -n DRL python -m drl_repro.cli summarize
```

This generates:

- `artifacts/results/summary/table2_summary.csv`
- `artifacts/results/summary/monthly_return_matrix.csv`
- `artifacts/results/summary/annual_return_matrix.csv`
- `artifacts/results/summary/turnover_summary.csv`

### 6. Generate figures

```bash
conda run --no-capture-output -n DRL python -m drl_repro.cli plot
```

This generates:

- `artifacts/results/figures/figure2_comparison.png`
- `artifacts/results/figures/figure3_drl_returns.png`
- `artifacts/results/figures/figure4_mvo_returns.png`

### 7. One-command pipeline

If you want the full end-to-end flow in one command:

```bash
conda run --no-capture-output -n DRL python -m drl_repro.cli reproduce
```

Useful variants:

```bash
conda run --no-capture-output -n DRL python -m drl_repro.cli reproduce --max-windows 1 --train-steps 50000
conda run --no-capture-output -n DRL python -m drl_repro.cli reproduce --refresh-data
```

## Output Layout

Key output directories:

- `artifacts/data/`
- `artifacts/results/rolling/<year>/drl/`
- `artifacts/results/rolling/<year>/mvo/`
- `artifacts/results/summary/`
- `artifacts/results/figures/`

For each yearly backtest, the repo stores:

- `nav.csv`
- `weights.csv`
- `metrics.csv`
- `turnover.csv`
- `monthly_returns.csv`
- `annual_returns.csv`

## What To Compare Against The Paper

The most useful checkpoints when comparing your run to the paper are:

- Year-by-year Sharpe ratio
- Year-by-year annual return
- Year-by-year max drawdown
- Average performance across all backtests
- Turnover behavior
- Monthly and annual return profiles

The generated files most relevant to this are:

- `artifacts/results/summary/table2_summary.csv`
- `artifacts/results/summary/monthly_return_matrix.csv`
- `artifacts/results/summary/annual_return_matrix.csv`
- `artifacts/results/summary/turnover_summary.csv`

## Important Remaining Gaps

This repo is much closer to the paper than the original scaffold, but it is still not guaranteed to be an exact author reproduction.

### 1. Yahoo ticker mapping may still differ from the authors' exact source

The paper says "S&P 500 sector indices," but it does not publish a code repository with a definitive ticker list.  
This repo uses Yahoo Finance symbols of the form:

- `^SP500-10`
- `^SP500-15`
- `^SP500-20`
- `^SP500-25`
- `^SP500-30`
- `^SP500-35`
- `^SP500-40`
- `^SP500-45`
- `^SP500-50`
- `^SP500-55`
- `^SP500-60`

If the authors used a different vendor feed, a different sector index variant, or a different Yahoo alias, results may shift.

### 2. Validation objective may still differ from the unpublished training code

The paper states that the best model is selected by highest mean episode validation reward.  
This repo follows that description using SB3 evaluation callbacks, but without author code we cannot guarantee every evaluation detail is identical.

### 3. MVO implementation is aligned in spirit, not guaranteed identical in solver details

This repo uses:

- 60-day rolling mean estimates
- Ledoit-Wolf covariance shrinkage
- PSD repair
- long-only fully invested optimization
- whole-share execution with cash residuals

The paper mentions PyPortfolioOpt. This repo uses a custom solver path around the same objective and constraints.

### 4. Market data availability can change over time

Yahoo Finance is not a fixed research archive.  
Adjusted close histories, missing values, and ticker support can change.

### 5. Existing historical artifacts in the repo may reflect older runs

If you want a clean paper-style result set, do not rely on previously committed `artifacts/results/rolling/` files.  
Run the experiment again with the current code and then regenerate summaries and figures.

## Practical Advice

- Start with `smoke`
- Then run `experiment --max-windows 1 --train-steps 50000`
- Then run full `experiment`
- Then run `summarize`
- Then run `plot`

If you are short on time, use `reproduce --max-windows 1 --train-steps 50000` first.

## Code Map

- `drl_repro/config.py`: paper-aligned config defaults
- `drl_repro/data.py`: download, caching, features, slicing
- `drl_repro/env.py`: portfolio environment and whole-share execution
- `drl_repro/ppo_agent.py`: PPO training, evaluation, seed selection helpers
- `drl_repro/mvo.py`: MVO baseline
- `drl_repro/experiment.py`: rolling-window experiment logic
- `drl_repro/metrics.py`: risk and return metrics plus turnover statistics
- `drl_repro/reporting.py`: summary table exports
- `drl_repro/plotting.py`: Figure 2/3/4 style plot generation
- `drl_repro/cli.py`: command-line entry points

## Minimal Command Set

If you only want the shortest useful sequence:

```bash
conda run --no-capture-output -n DRL python -m drl_repro.cli download
conda run --no-capture-output -n DRL python -m drl_repro.cli experiment
conda run --no-capture-output -n DRL python -m drl_repro.cli summarize
conda run --no-capture-output -n DRL python -m drl_repro.cli plot
```
