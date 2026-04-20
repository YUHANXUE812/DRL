from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentConfig:
    data_dir: Path = Path("artifacts/data")
    results_dir: Path = Path("artifacts/results")

    # Proxy universe with long-enough history for 2006+ experiments.
    asset_tickers: list[str] = field(
        default_factory=lambda: [
            "VAW",  # materials
            "VDE",  # energy
            "VFH",  # financials
            "VIS",  # industrials
            "VGT",  # information technology
            "VDC",  # consumer staples
            "VPU",  # utilities
            "VHT",  # health care
            "VCR",  # consumer discretionary
            "VNQ",  # real estate proxy
            "VOX",  # communication services proxy
        ]
    )
    market_ticker: str = "SPY"
    vix_ticker: str = "^VIX"

    start_date: str = "2005-01-01"
    end_date: str = "2022-01-01"
    lookback: int = 60
    initial_cash: float = 100_000.0
    reward_eta: float = 1.0 / 252.0

    train_years: int = 5
    val_years: int = 1
    test_years: int = 1
    rolling_step_years: int = 1

    # Paper-inspired PPO defaults.
    train_steps: int = 200_000
    ppo_n_envs: int = 10
    ppo_n_steps: int = 252 * 3
    ppo_batch_size: int = 252 * 5
    ppo_n_epochs: int = 16
    ppo_gamma: float = 0.9
    ppo_gae_lambda: float = 0.9
    ppo_clip_range: float = 0.25
    ppo_learning_rate: float = 3e-4
    ppo_log_std_init: float = -1.0
    policy_hidden_sizes: tuple[int, int] = (64, 64)

    seed: int = 7

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
