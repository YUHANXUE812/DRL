from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .data import build_market_dataset
from .experiment import run_experiment
from .plotting import export_summary_figures
from .reporting import export_summary_reports


def print_output_paths(paths: dict[str, Path]) -> None:
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")


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

    summarize = subparsers.add_parser("summarize", help="Aggregate backtest artifacts into paper-style summary tables")
    summarize.add_argument("--results-dir", type=Path, default=None)

    plot = subparsers.add_parser("plot", help="Generate Figure 2/3/4 style plots from backtest artifacts")
    plot.add_argument("--results-dir", type=Path, default=None)

    reproduce = subparsers.add_parser("reproduce", help="Run download, experiment, summarize, and plotting in one command")
    reproduce.add_argument("--max-windows", type=int, default=None)
    reproduce.add_argument("--train-steps", type=int, default=None)
    reproduce.add_argument("--refresh-data", action="store_true")

    args = parser.parse_args()
    config = ExperimentConfig()

    if args.command == "download":
        data = build_market_dataset(config, refresh=True)
        print(f"Downloaded {len(data.prices)} rows for {len(data.prices.columns)} assets.")
        return

    if args.command == "smoke":
        config.train_steps = 2_048
        config.n_train_seeds = 2
        config.use_subproc_vecenv = False
        config.eval_frequency = 512
        summary = run_experiment(config, max_windows=1, refresh_data=args.refresh_data)
        print(summary.to_string(index=False))
        return

    if args.command == "experiment":
        if args.train_steps is not None:
            config.train_steps = args.train_steps
        summary = run_experiment(config, max_windows=args.max_windows, refresh_data=args.refresh_data)
        print(summary.to_string(index=False))
        return

    if args.command == "summarize":
        results_dir = args.results_dir or config.results_dir
        output_paths = export_summary_reports(results_dir)
        print_output_paths(output_paths)
        return

    if args.command == "plot":
        results_dir = args.results_dir or config.results_dir
        output_paths = export_summary_figures(results_dir)
        print_output_paths(output_paths)
        return

    if args.command == "reproduce":
        if args.train_steps is not None:
            config.train_steps = args.train_steps

        data = build_market_dataset(config, refresh=args.refresh_data)
        print(f"downloaded_rows: {len(data.prices)}")

        summary = run_experiment(config, max_windows=args.max_windows, refresh_data=False)
        print(summary.to_string(index=False))

        summary_outputs = export_summary_reports(config.results_dir)
        figure_outputs = export_summary_figures(config.results_dir)
        print_output_paths({**summary_outputs, **figure_outputs})
        return


if __name__ == "__main__":
    main()
