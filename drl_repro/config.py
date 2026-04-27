from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    data_dir: Path = Path("artifacts/data")
    results_dir: Path = Path("artifacts/results")

    # Paper-aligned Yahoo Finance sector index symbols.
    asset_tickers: list[str] = field(
        default_factory=lambda: [
            "^SP500-15",  # materials
            "^SP500-10",  # energy
            "^SP500-40",  # financials
            "^SP500-20",  # industrials
            "^SP500-45",  # information technology
            "^SP500-30",  # consumer staples
            "^SP500-55",  # utilities
            "^SP500-35",  # health care
            "^SP500-25",  # consumer discretionary
            "^SP500-60",  # real estate
            "^SP500-50",  # communication services
        ]
    )
    market_ticker: str = "^GSPC"
    vix_ticker: str = "^VIX"

    start_date: str = "2006-01-01"
    end_date: str = "2022-01-01"
    lookback: int = 60
    initial_cash: float = 100_000.0
    reward_eta: float = 1.0 / 252.0

    train_years: int = 5
    val_years: int = 1
    test_years: int = 1
    rolling_step_years: int = 1
    first_window_start_year: int = 2006
    first_test_year: int = 2012
    last_test_year: int = 2021

    # Paper-inspired PPO defaults.
    train_steps: int = 7_500_000
    n_train_seeds: int = 5
    train_seeds: tuple[int, ...] = (7, 11, 19, 23, 31)
    ppo_n_envs: int = 10
    ppo_n_steps: int = 252 * 3
    ppo_batch_size: int = 252 * 5
    ppo_n_epochs: int = 16
    ppo_gamma: float = 0.9
    ppo_gae_lambda: float = 0.9
    ppo_clip_range: float = 0.25
    ppo_learning_rate: float = 3e-4
    ppo_learning_rate_final: float = 1e-5
    ppo_log_std_init: float = -1.0
    policy_hidden_sizes: tuple[int, int] = (64, 64)
    eval_frequency: int = 25_200
    use_subproc_vecenv: bool = True

    seed: int = 7

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
