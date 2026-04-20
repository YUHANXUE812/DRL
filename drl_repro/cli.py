from __future__ import annotations

import argparse

from .config import ExperimentConfig
from .data import build_market_dataset
from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="DRL portfolio allocation paper reproduction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download and cache market data")

    smoke = subparsers.add_parser("smoke", help="Run a quick end-to-end smoke test")
    smoke.add_argument("--refresh-data", action="store_true")

    experiment = subparsers.add_parser("experiment", help="Run rolling-window experiment")
    experiment.add_argument("--max-windows", type=int, default=None)
    experiment.add_argument("--train-steps", type=int, default=None)
    experiment.add_argument("--refresh-data", action="store_true")

    args = parser.parse_args()
    config = ExperimentConfig()

    if args.command == "download":
        data = build_market_dataset(config, refresh=True)
        print(f"Downloaded {len(data.prices)} rows for {len(data.prices.columns)} assets.")
        return

    if args.command == "smoke":
        config.train_steps = 2_048
        summary = run_experiment(config, max_windows=1, refresh_data=args.refresh_data)
        print(summary.to_string(index=False))
        return

    if args.command == "experiment":
        if args.train_steps is not None:
            config.train_steps = args.train_steps
        summary = run_experiment(config, max_windows=args.max_windows, refresh_data=args.refresh_data)
        print(summary.to_string(index=False))
        return


if __name__ == "__main__":
    main()
