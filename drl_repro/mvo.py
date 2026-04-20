from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .data import MarketData


def mvo_weights_from_window(window_returns: pd.DataFrame) -> np.ndarray:
    mu = window_returns.mean().values
    cov = LedoitWolf().fit(window_returns.values).covariance_
    cov = nearest_psd(cov)

    n_assets = len(mu)

    def objective(w: np.ndarray) -> float:
        port_ret = float(np.dot(mu, w))
        port_vol = float(np.sqrt(np.maximum(w @ cov @ w, 1e-12)))
        return -(port_ret / port_vol)

    x0 = np.repeat(1.0 / n_assets, n_assets)
    bounds = [(0.0, 1.0)] * n_assets
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(objective, x0=x0, bounds=bounds, constraints=constraints, method="SLSQP")

    if not result.success:
        return x0
    return np.clip(result.x, 0.0, 1.0)


def nearest_psd(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def run_mvo_backtest(
    market_data: MarketData,
    lookback: int,
    initial_cash: float,
) -> tuple[pd.Series, pd.DataFrame]:
    prices = market_data.prices
    returns = market_data.asset_returns
    dates = prices.index

    portfolio_value = initial_cash
    nav = [initial_cash]
    weights_history = []
    nav_index = [dates[lookback - 1]]

    for i in range(lookback, len(dates)):
        window = returns.iloc[i - lookback : i].dropna()
        weights = mvo_weights_from_window(window)

        prev_prices = prices.iloc[i - 1].values
        next_prices = prices.iloc[i].values
        asset_rets = (next_prices / prev_prices) - 1.0
        portfolio_ret = float(np.dot(weights, asset_rets))
        portfolio_value *= 1.0 + portfolio_ret

        weights_history.append(np.append(weights, 0.0))
        nav.append(portfolio_value)
        nav_index.append(dates[i])

    nav_series = pd.Series(nav, index=nav_index, name="nav")
    weight_df = pd.DataFrame(weights_history, index=dates[lookback:])
    weight_df.columns = list(prices.columns) + ["CASH"]
    return nav_series, weight_df
