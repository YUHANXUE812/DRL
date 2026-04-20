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
    ann_vol = float(daily_returns.std() * np.sqrt(ann_factor))
    sharpe = float((daily_returns.mean() / max(daily_returns.std(), 1e-12)) * np.sqrt(ann_factor))
    downside_std = daily_returns[daily_returns < 0].std()
    sortino = float((daily_returns.mean() / max(downside_std, 1e-12)) * np.sqrt(ann_factor))

    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    max_drawdown = float(drawdown.min())
    calmar = float(ann_return / max(abs(max_drawdown), 1e-12))

    return {
        "cumulative_return": total_return,
        "annual_return": ann_return,
        "annual_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
    }
