from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .reporting import build_return_matrix, build_turnover_summary, collect_backtest_rows


METHOD_COLORS = {
    "drl": "#1b6ca8",
    "mvo": "#d97706",
}


def load_summary_matrix(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    return frame


def _style_axis(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_figure2(yearly_metrics: pd.DataFrame, turnover_summary: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = [
        ("sharpe_ratio", "Sharpe Ratio"),
        ("annual_return", "Annual Return"),
        ("max_drawdown", "Max Drawdown"),
        ("average_turnover", "Average Turnover"),
    ]

    merged = yearly_metrics.copy()
    if not turnover_summary.empty:
        merged = merged.merge(
            turnover_summary[["year", "method", "average_turnover"]],
            on=["year", "method"],
            how="left",
        )

    years = sorted(merged["year"].astype(str).unique())
    x = np.arange(len(years))
    width = 0.38

    for ax, (metric, title) in zip(axes.flat, panels):
        drl_vals = []
        mvo_vals = []
        for year in years:
            drl_match = merged[(merged["year"].astype(str) == year) & (merged["method"] == "drl")]
            mvo_match = merged[(merged["year"].astype(str) == year) & (merged["method"] == "mvo")]
            drl_vals.append(float(drl_match[metric].iloc[0]) if not drl_match.empty and metric in drl_match else np.nan)
            mvo_vals.append(float(mvo_match[metric].iloc[0]) if not mvo_match.empty and metric in mvo_match else np.nan)

        ax.bar(x - width / 2, drl_vals, width=width, color=METHOD_COLORS["drl"], label="DRL")
        ax.bar(x + width / 2, mvo_vals, width=width, color=METHOD_COLORS["mvo"], label="MVO")
        ax.set_xticks(x)
        ax.set_xticklabels(years, rotation=45)
        _style_axis(ax, title)
        if metric == "max_drawdown":
            ax.axhline(0.0, color="#666666", linewidth=0.8)

    axes[0, 0].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Figure 2 Style Comparison Across Backtests", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_method_return_figure(
    monthly_matrix: pd.DataFrame,
    annual_matrix: pd.DataFrame,
    method: str,
    output_path: Path,
) -> None:
    method_cols_monthly = [col for col in monthly_matrix.columns if col.startswith(f"{method}_")]
    method_cols_annual = [col for col in annual_matrix.columns if col.startswith(f"{method}_")]

    if not method_cols_monthly or not method_cols_annual:
        return

    monthly_long = monthly_matrix[method_cols_monthly].stack().dropna()
    monthly_long.index = monthly_long.index.set_names(["date", "series"])
    monthly_frame = monthly_long.reset_index(name="return")
    monthly_frame["window"] = monthly_frame["series"].str.split("_").str[-1]

    annual_series = annual_matrix[method_cols_annual].iloc[0].dropna()
    annual_years = [col.split("_")[-1] for col in annual_series.index]
    annual_values = annual_series.to_numpy(dtype=float)
    annual_mean = float(np.nanmean(annual_values)) if len(annual_values) else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    color = METHOD_COLORS[method]

    for window, sub in monthly_frame.groupby("window"):
        axes[0].plot(sub["date"], sub["return"], color=color, alpha=0.35, linewidth=1.0)
    monthly_avg = monthly_frame.groupby("date")["return"].mean()
    axes[0].plot(monthly_avg.index, monthly_avg.values, color=color, linewidth=2.0)
    _style_axis(axes[0], f"{method.upper()} Monthly Returns")
    axes[0].axhline(0.0, color="#666666", linewidth=0.8)

    x = np.arange(len(annual_years))
    axes[1].bar(x, annual_values, color=color, alpha=0.9)
    axes[1].axhline(annual_mean, color="#222222", linestyle="--", linewidth=1.2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(annual_years, rotation=45)
    _style_axis(axes[1], f"{method.upper()} Annual Returns")
    axes[1].axhline(0.0, color="#666666", linewidth=0.8)

    axes[2].hist(monthly_long.values, bins=18, color=color, alpha=0.85, edgecolor="white")
    axes[2].axvline(float(monthly_long.mean()), color="#222222", linestyle="--", linewidth=1.2)
    _style_axis(axes[2], f"{method.upper()} Monthly Return Distribution")
    axes[2].axvline(0.0, color="#666666", linewidth=0.8)

    title = "Figure 3 Style DRL Return Diagnostics" if method == "drl" else "Figure 4 Style MVO Return Diagnostics"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_summary_figures(results_dir: Path) -> dict[str, Path]:
    summary_dir = results_dir / "summary"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    yearly_metrics = collect_backtest_rows(results_dir)
    turnover_summary = build_turnover_summary(results_dir)
    monthly_matrix = build_return_matrix(results_dir, "monthly_returns.csv")
    annual_matrix = build_return_matrix(results_dir, "annual_returns.csv")

    output_paths = {
        "figure2": figures_dir / "figure2_comparison.png",
        "figure3": figures_dir / "figure3_drl_returns.png",
        "figure4": figures_dir / "figure4_mvo_returns.png",
    }

    if not yearly_metrics.empty:
        plot_figure2(yearly_metrics, turnover_summary, output_paths["figure2"])
    if not monthly_matrix.empty and not annual_matrix.empty:
        plot_method_return_figure(monthly_matrix, annual_matrix, "drl", output_paths["figure3"])
        plot_method_return_figure(monthly_matrix, annual_matrix, "mvo", output_paths["figure4"])

    return output_paths
