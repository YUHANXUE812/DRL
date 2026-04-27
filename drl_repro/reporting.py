from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import compute_performance_metrics, compute_periodic_returns, compute_turnover_metrics, compute_turnover_series


TABLE2_METRIC_ORDER = [
    "annual_return",
    "cumulative_return",
    "annual_volatility",
    "sharpe_ratio",
    "calmar_ratio",
    "stability",
    "max_drawdown",
    "omega_ratio",
    "sortino_ratio",
    "skew",
    "kurtosis",
    "tail_ratio",
    "daily_value_at_risk",
]


def load_metric_series(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    series = frame.iloc[:, 0]
    series.index = series.index.astype(str)
    return pd.to_numeric(series, errors="coerce")


def load_return_series(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col=0)
    series = frame.iloc[:, 0]
    series.index = pd.to_datetime(series.index)
    series.name = path.parent.parent.name
    return pd.to_numeric(series, errors="coerce")


def collect_backtest_rows(results_dir: Path) -> pd.DataFrame:
    rolling_dir = results_dir / "rolling"
    rows = []

    for method_dir in sorted(rolling_dir.glob("*/*")):
        metrics_path = method_dir / "metrics.csv"
        nav_path = method_dir / "nav.csv"
        weights_path = method_dir / "weights.csv"
        if not nav_path.exists():
            continue

        year = method_dir.parent.name
        method = method_dir.name
        metrics = load_metric_series(metrics_path) if metrics_path.exists() else pd.Series(dtype=float)
        nav = load_return_series(nav_path)
        recomputed_metrics = pd.Series(compute_performance_metrics(nav))
        metrics = recomputed_metrics.combine_first(metrics)
        if weights_path.exists():
            weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
            turnover_metrics = pd.Series(compute_turnover_metrics(weights))
            metrics = turnover_metrics.combine_first(metrics)

        row = {"year": year, "method": method}
        row.update(metrics.to_dict())
        rows.append(row)

    return pd.DataFrame(rows)


def build_table2_summary(results_dir: Path) -> pd.DataFrame:
    frame = collect_backtest_rows(results_dir)
    if frame.empty:
        return pd.DataFrame(columns=["metric", "drl", "mvo"])
    summary_rows = []

    for metric in TABLE2_METRIC_ORDER:
        metric_row = {"metric": metric}
        for method in sorted(frame["method"].unique()):
            if metric not in frame.columns:
                metric_row[method] = float("nan")
                continue
            method_values = frame.loc[frame["method"] == method, metric].dropna()
            if method_values.empty:
                metric_row[method] = float("nan")
            elif metric == "max_drawdown":
                metric_row[method] = float(method_values.min())
            else:
                metric_row[method] = float(method_values.mean())
        summary_rows.append(metric_row)

    return pd.DataFrame(summary_rows)


def build_return_matrix(results_dir: Path, filename: str) -> pd.DataFrame:
    rolling_dir = results_dir / "rolling"
    matrices: list[pd.DataFrame] = []

    for method_dir in sorted(rolling_dir.glob("*/*")):
        path = method_dir / filename
        year = method_dir.parent.name
        method = method_dir.name
        if path.exists():
            series = load_return_series(path)
        else:
            nav_path = method_dir / "nav.csv"
            if not nav_path.exists():
                continue
            nav = load_return_series(nav_path)
            freq = "ME" if filename == "monthly_returns.csv" else "YE"
            series = compute_periodic_returns(nav, freq)
        frame = series.to_frame(name=f"{method}_{year}")
        matrices.append(frame)

    if not matrices:
        return pd.DataFrame()

    matrix = pd.concat(matrices, axis=1).sort_index()
    matrix.index.name = "date"
    return matrix


def build_turnover_summary(results_dir: Path) -> pd.DataFrame:
    rolling_dir = results_dir / "rolling"
    rows = []

    for method_dir in sorted(rolling_dir.glob("*/*")):
        turnover_path = method_dir / "turnover.csv"
        if turnover_path.exists():
            turnover = load_return_series(turnover_path)
        else:
            weights_path = method_dir / "weights.csv"
            if not weights_path.exists():
                continue
            weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
            turnover = compute_turnover_series(weights)
        rows.append(
            {
                "year": method_dir.parent.name,
                "method": method_dir.name,
                "average_turnover": float(turnover.mean()),
                "median_turnover": float(turnover.median()),
                "max_turnover": float(turnover.max()),
                "turnover_std": float(turnover.std(ddof=1)) if len(turnover) > 1 else 0.0,
            }
        )

    return pd.DataFrame(rows)


def export_summary_reports(results_dir: Path) -> dict[str, Path]:
    summary_dir = results_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    table2_summary = build_table2_summary(results_dir)
    monthly_matrix = build_return_matrix(results_dir, "monthly_returns.csv")
    annual_matrix = build_return_matrix(results_dir, "annual_returns.csv")
    turnover_summary = build_turnover_summary(results_dir)

    output_paths = {
        "table2": summary_dir / "table2_summary.csv",
        "monthly": summary_dir / "monthly_return_matrix.csv",
        "annual": summary_dir / "annual_return_matrix.csv",
        "turnover": summary_dir / "turnover_summary.csv",
    }

    table2_summary.to_csv(output_paths["table2"], index=False)
    monthly_matrix.to_csv(output_paths["monthly"])
    annual_matrix.to_csv(output_paths["annual"])
    turnover_summary.to_csv(output_paths["turnover"], index=False)
    return output_paths
