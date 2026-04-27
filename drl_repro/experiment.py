from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from stable_baselines3 import PPO

from .config import ExperimentConfig
from .data import (
    MarketData,
    build_market_dataset,
    save_backtest_outputs,
    slice_by_dates,
    slice_with_lookback,
)
from .env import PortfolioEnv, run_policy_backtest
from .metrics import (
    compute_performance_metrics,
    compute_periodic_returns,
    compute_turnover_metrics,
    compute_turnover_series,
)
from .mvo import run_mvo_backtest
from .ppo_agent import model_policy_fn, train_ppo


@dataclass(frozen=True)
class RollingWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_rolling_windows(market_data: MarketData, config: ExperimentConfig) -> list[RollingWindow]:
    first_date = market_data.prices.index.min()
    last_date = market_data.prices.index.max()
    windows: list[RollingWindow] = []

    for test_year in range(config.first_test_year, config.last_test_year + 1, config.rolling_step_years):
        train_start = pd.Timestamp(f"{test_year - config.val_years - config.train_years}-01-01")
        train_end = pd.Timestamp(f"{test_year - config.val_years}-01-01")
        validation_start = train_end
        validation_end = pd.Timestamp(f"{test_year}-01-01")
        test_start = validation_end
        test_end = pd.Timestamp(f"{test_year + config.test_years}-01-01")

        if train_start.year < config.first_window_start_year:
            continue
        if train_start < first_date or test_end > (last_date + pd.Timedelta(days=1)):
            continue
        windows.append(
            RollingWindow(
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    return windows


def train_best_drl_agent(
    train_data: MarketData,
    val_data: MarketData,
    config: ExperimentConfig,
    output_dir: Path,
    warm_start_policy_state: dict[str, torch.Tensor] | None,
) -> tuple[PPO, dict[str, torch.Tensor], int, float]:
    best_model: PPO | None = None
    best_policy_state: dict[str, torch.Tensor] | None = None
    best_seed = -1
    best_reward = float("-inf")

    for seed in config.train_seeds[: config.n_train_seeds]:
        seed_dir = output_dir / f"seed_{seed}"
        model, validation_reward, policy_state = train_ppo(
            train_data=train_data,
            val_data=val_data,
            config=config,
            seed=seed,
            output_dir=seed_dir,
            warm_start_policy_state=warm_start_policy_state,
        )
        if validation_reward > best_reward:
            if best_model is not None and best_model.get_env() is not None:
                best_model.get_env().close()
            best_model = model
            best_policy_state = policy_state
            best_seed = seed
            best_reward = validation_reward
        else:
            if model.get_env() is not None:
                model.get_env().close()

    if best_model is None or best_policy_state is None:
        raise RuntimeError("Failed to train and select a PPO agent.")

    return best_model, best_policy_state, best_seed, best_reward


def run_single_window(
    market_data: MarketData,
    config: ExperimentConfig,
    window: RollingWindow,
    output_root: Path,
    warm_start_policy_state: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, torch.Tensor]]:
    train_data = slice_by_dates(market_data, window.train_start, window.train_end)
    val_data = slice_by_dates(market_data, window.validation_start, window.validation_end)
    test_data = slice_with_lookback(market_data, window.test_start, window.test_end, config.lookback)

    drl_output_dir = output_root / f"{window.test_start.year}" / "drl_training"
    best_model, best_policy_state, best_seed, best_reward = train_best_drl_agent(
        train_data=train_data,
        val_data=val_data,
        config=config,
        output_dir=drl_output_dir,
        warm_start_policy_state=warm_start_policy_state,
    )

    drl_env = PortfolioEnv(
        market_data=test_data,
        lookback=config.lookback,
        initial_cash=config.initial_cash,
        reward_eta=config.reward_eta,
    )
    drl_nav, drl_weights = run_policy_backtest(drl_env, model_policy_fn(best_model))
    drl_metrics = compute_performance_metrics(drl_nav)
    drl_turnover = compute_turnover_series(drl_weights)
    drl_monthly_returns = compute_periodic_returns(drl_nav, "ME")
    drl_annual_returns = compute_periodic_returns(drl_nav, "YE")
    drl_metrics.update(compute_turnover_metrics(drl_weights))
    drl_metrics["selected_seed"] = float(best_seed)
    drl_metrics["validation_reward"] = best_reward
    if best_model.get_env() is not None:
        best_model.get_env().close()

    mvo_nav, mvo_weights = run_mvo_backtest(
        market_data=test_data,
        lookback=config.lookback,
        initial_cash=config.initial_cash,
    )
    mvo_metrics = compute_performance_metrics(mvo_nav)
    mvo_turnover = compute_turnover_series(mvo_weights)
    mvo_monthly_returns = compute_periodic_returns(mvo_nav, "ME")
    mvo_annual_returns = compute_periodic_returns(mvo_nav, "YE")
    mvo_metrics.update(compute_turnover_metrics(mvo_weights))

    tag = f"{window.test_start.year}"
    save_backtest_outputs(
        output_root / tag / "drl",
        drl_nav,
        drl_weights,
        drl_metrics,
        turnover=drl_turnover,
        monthly_returns=drl_monthly_returns,
        annual_returns=drl_annual_returns,
    )
    save_backtest_outputs(
        output_root / tag / "mvo",
        mvo_nav,
        mvo_weights,
        mvo_metrics,
        turnover=mvo_turnover,
        monthly_returns=mvo_monthly_returns,
        annual_returns=mvo_annual_returns,
    )
    return {"drl": drl_metrics, "mvo": mvo_metrics}, best_policy_state


def run_experiment(config: ExperimentConfig, max_windows: int | None = None, refresh_data: bool = False) -> pd.DataFrame:
    market_data = build_market_dataset(config, refresh=refresh_data)
    windows = build_rolling_windows(market_data, config)
    if max_windows is not None:
        windows = windows[:max_windows]

    rows = []
    output_root = config.results_dir / "rolling"
    output_root.mkdir(parents=True, exist_ok=True)

    warm_start_policy_state: dict[str, torch.Tensor] | None = None
    for window in windows:
        result, warm_start_policy_state = run_single_window(
            market_data=market_data,
            config=config,
            window=window,
            output_root=output_root,
            warm_start_policy_state=warm_start_policy_state,
        )
        for method, metrics in result.items():
            row = {
                "method": method,
                "train_start": window.train_start.date().isoformat(),
                "train_end": window.train_end.date().isoformat(),
                "validation_start": window.validation_start.date().isoformat(),
                "validation_end": window.validation_end.date().isoformat(),
                "test_start": window.test_start.date().isoformat(),
                "test_end": window.test_end.date().isoformat(),
            }
            row.update(metrics)
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(config.results_dir / "experiment_summary.csv", index=False)
    pd.Series(asdict(config), name="value").astype(str).to_csv(config.results_dir / "config_snapshot.csv")
    return summary
