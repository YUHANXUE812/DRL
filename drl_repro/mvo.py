from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from .data import MarketData
from .env import PortfolioEnv, run_policy_backtest


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
    env = PortfolioEnv(
        market_data=market_data,
        lookback=lookback,
        initial_cash=initial_cash,
        reward_eta=1.0 / 252.0,
    )

    def policy_fn(_obs: np.ndarray, live_env: PortfolioEnv) -> np.ndarray:
        window = live_env.market_data.asset_returns.iloc[live_env.ptr - live_env.lookback : live_env.ptr].dropna()
        weights = mvo_weights_from_window(window)
        return np.append(weights, 0.0)

    return run_policy_backtest(env, policy_fn)
