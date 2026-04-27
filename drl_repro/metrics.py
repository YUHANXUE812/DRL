from __future__ import annotations

import numpy as np
import pandas as pd


def compute_performance_metrics(nav: pd.Series) -> dict[str, float]:
    daily_returns = nav.pct_change().dropna()
    if daily_returns.empty:
        return {}

    ann_factor = 252.0
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    ann_return = float((1.0 + total_return) ** (ann_factor / max(len(daily_returns), 1)) - 1.0)
    ann_vol = float(daily_returns.std(ddof=1) * np.sqrt(ann_factor))
    sharpe = float((daily_returns.mean() / max(daily_returns.std(ddof=1), 1e-12)) * np.sqrt(ann_factor))

    downside = daily_returns[daily_returns < 0.0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = float((daily_returns.mean() / max(downside_std, 1e-12)) * np.sqrt(ann_factor))

    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(ann_return / max(abs(max_drawdown), 1e-12))

    omega = omega_ratio(daily_returns)
    stability = stability_of_timeseries(daily_returns)
    tail = tail_ratio(daily_returns)
    value_at_risk = daily_value_at_risk(daily_returns)

    return {
        "cumulative_return": total_return,
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "stability": stability,
        "max_drawdown": max_drawdown,
        "omega_ratio": omega,
        "sortino_ratio": sortino,
        "skew": float(daily_returns.skew()),
        "kurtosis": float(daily_returns.kurt()),
        "tail_ratio": tail,
        "daily_value_at_risk": value_at_risk,
    }


def stability_of_timeseries(returns: pd.Series) -> float:
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0

    cumulative_log_returns = np.log1p(clean).cumsum().values
    x = np.arange(len(cumulative_log_returns), dtype=np.float64)
    slope, intercept = np.polyfit(x, cumulative_log_returns, deg=1)
    fitted = slope * x + intercept
    residual = cumulative_log_returns - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((cumulative_log_returns - cumulative_log_returns.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0

    excess = clean - threshold
    gains = float(excess[excess > 0.0].sum())
    losses = float(-excess[excess < 0.0].sum())
    if losses <= 1e-12:
        return float("inf") if gains > 0.0 else 0.0
    return float(gains / losses)


def tail_ratio(returns: pd.Series) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0

    upper = float(clean.quantile(0.95))
    lower = float(clean.quantile(0.05))
    if abs(lower) <= 1e-12:
        return 0.0
    return float(upper / abs(lower))


def daily_value_at_risk(returns: pd.Series, cutoff: float = 0.05) -> float:
    clean = returns.dropna()
    if clean.empty:
        return 0.0
    return float(clean.quantile(cutoff))


def compute_turnover_series(weights: pd.DataFrame) -> pd.Series:
    if weights.empty:
        return pd.Series(dtype=np.float64, name="turnover")

    asset_columns = [col for col in weights.columns if col != "CASH"]
    previous = np.zeros(len(asset_columns), dtype=np.float64)
    turnover_values = []

    for _, row in weights[asset_columns].iterrows():
        current = row.to_numpy(dtype=np.float64)
        turnover_values.append(float(np.abs(current - previous).sum()))
        previous = current

    return pd.Series(turnover_values, index=weights.index, name="turnover")


def compute_turnover_metrics(weights: pd.DataFrame) -> dict[str, float]:
    turnover = compute_turnover_series(weights)
    if turnover.empty:
        return {}

    return {
        "average_turnover": float(turnover.mean()),
        "median_turnover": float(turnover.median()),
        "max_turnover": float(turnover.max()),
        "turnover_std": float(turnover.std(ddof=1)) if len(turnover) > 1 else 0.0,
    }


def compute_periodic_returns(nav: pd.Series, frequency: str) -> pd.Series:
    clean_nav = nav.dropna()
    if clean_nav.empty:
        return pd.Series(dtype=np.float64)

    if not isinstance(clean_nav.index, pd.DatetimeIndex):
        clean_nav.index = pd.to_datetime(clean_nav.index)

    sampled = clean_nav.resample(frequency).last()
    returns = sampled.pct_change().dropna()
    returns.name = "return"
    return returns
