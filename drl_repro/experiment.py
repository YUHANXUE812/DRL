from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig
from .data import MarketData, build_market_dataset, save_backtest_outputs, slice_by_dates
from .env import PortfolioEnv, run_policy_backtest
from .metrics import compute_performance_metrics
from .mvo import run_mvo_backtest
from .ppo_agent import model_policy_fn, train_ppo


def build_rolling_windows(market_data: MarketData, config: ExperimentConfig) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    dates = market_data.prices.index
    start_year = dates.min().year
    end_year = dates.max().year

    windows = []
    train = config.train_years
    val = config.val_years
    test = config.test_years
    step = config.rolling_step_years

    year = start_year
    while year + train + val + test <= end_year:
        train_start = pd.Timestamp(f"{year}-01-01")
        train_end = pd.Timestamp(f"{year + train}-01-01")
        test_start = pd.Timestamp(f"{year + train + val}-01-01")
        test_end = pd.Timestamp(f"{year + train + val + test}-01-01")
        windows.append((train_start, train_end, test_start, test_end))
        year += step
    return windows


def run_single_window(
    market_data: MarketData,
    config: ExperimentConfig,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    output_root: Path,
) -> dict[str, dict[str, float]]:
    train_data = slice_by_dates(market_data, train_start, train_end)
    test_data = slice_by_dates(market_data, test_start - pd.Timedelta(days=120), test_end)

    model = train_ppo(train_data, config)

    drl_env = PortfolioEnv(
        market_data=test_data,
        lookback=config.lookback,
        initial_cash=config.initial_cash,
        reward_eta=config.reward_eta,
    )
    drl_nav, drl_weights = run_policy_backtest(drl_env, model_policy_fn(model))
    drl_metrics = compute_performance_metrics(drl_nav)

    mvo_nav, mvo_weights = run_mvo_backtest(
        market_data=test_data,
        lookback=config.lookback,
        initial_cash=config.initial_cash,
    )
    mvo_metrics = compute_performance_metrics(mvo_nav)

    tag = f"{test_start.year}"
    save_backtest_outputs(output_root / tag / "drl", drl_nav, drl_weights, drl_metrics)
    save_backtest_outputs(output_root / tag / "mvo", mvo_nav, mvo_weights, mvo_metrics)
    return {"drl": drl_metrics, "mvo": mvo_metrics}


def run_experiment(config: ExperimentConfig, max_windows: int | None = None, refresh_data: bool = False) -> pd.DataFrame:
    market_data = build_market_dataset(config, refresh=refresh_data)
    windows = build_rolling_windows(market_data, config)
    if max_windows is not None:
        windows = windows[:max_windows]

    rows = []
    output_root = config.results_dir / "rolling"
    output_root.mkdir(parents=True, exist_ok=True)

    for train_start, train_end, test_start, test_end in windows:
        result = run_single_window(
            market_data=market_data,
            config=config,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            output_root=output_root,
        )
        for method, metrics in result.items():
            row = {
                "method": method,
                "train_start": train_start.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_start": test_start.date().isoformat(),
                "test_end": test_end.date().isoformat(),
            }
            row.update(metrics)
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(config.results_dir / "experiment_summary.csv", index=False)
    pd.Series(asdict(config), name="value").astype(str).to_csv(config.results_dir / "config_snapshot.csv")
    return summary
