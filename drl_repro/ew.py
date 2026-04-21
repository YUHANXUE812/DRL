from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MarketData


def run_ew_backtest(
    market_data: MarketData,
    lookback: int,
    initial_cash: float,
) -> tuple[pd.Series, pd.DataFrame]:
    prices = market_data.prices
    dates = prices.index
    n_assets = prices.shape[1]
    weights = np.repeat(1.0 / n_assets, n_assets)

    portfolio_value = initial_cash
    nav = [initial_cash]
    weights_history = []
    nav_index = [dates[lookback - 1]]

    for i in range(lookback, len(dates)):
        prev_prices = prices.iloc[i - 1].values
        next_prices = prices.iloc[i].values
        asset_rets = (next_prices / prev_prices) - 1.0
        portfolio_value *= 1.0 + float(np.dot(weights, asset_rets))

        weights_history.append(np.append(weights, 0.0))
        nav.append(portfolio_value)
        nav_index.append(dates[i])

    nav_series = pd.Series(nav, index=nav_index, name="nav")
    weight_df = pd.DataFrame(weights_history, index=dates[lookback:])
    weight_df.columns = list(prices.columns) + ["CASH"]
    return nav_series, weight_df
