from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import cache as yf_cache

from .config import ExperimentConfig


@dataclass
class MarketData:
    prices: pd.DataFrame
    asset_returns: pd.DataFrame
    features: pd.DataFrame


def _download_adj_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        close = data.to_frame(name=tickers[0])
    close = close.sort_index().dropna(how="all")
    if close.empty:
        raise RuntimeError(
            "No market data was downloaded. Check network access, proxy settings, or ticker symbols."
        )
    return close


def build_market_dataset(config: ExperimentConfig, refresh: bool = False) -> MarketData:
    config.ensure_dirs()
    yf_cache_dir = config.data_dir / "yfinance_cache"
    yf_cache_dir.mkdir(parents=True, exist_ok=True)
    yf_cache.set_cache_location(str(yf_cache_dir))

    cache_path = config.data_dir / "market_data.csv"
    if cache_path.exists() and not refresh:
        frame = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return _from_cached_frame(frame, config)

    tickers = config.asset_tickers + [config.market_ticker, config.vix_ticker]
    close = _download_adj_close(tickers, config.start_date, config.end_date)
    close = close.ffill().dropna()
    close.to_csv(cache_path)
    return _from_cached_frame(close, config)


def _from_cached_frame(close: pd.DataFrame, config: ExperimentConfig) -> MarketData:
    asset_prices = close[config.asset_tickers].copy()
    market_price = close[config.market_ticker].copy()
    vix_price = close[config.vix_ticker].copy()

    asset_returns = np.log(asset_prices / asset_prices.shift(1))
    market_returns = np.log(market_price / market_price.shift(1))

    vol20 = market_returns.rolling(20).std()
    vol60 = market_returns.rolling(60).std()
    vol_ratio = vol20 / vol60.replace(0.0, np.nan)

    feature_frame = pd.DataFrame(
        {
            "vol20": vol20,
            "vol_ratio": vol_ratio,
            "vix": vix_price,
        },
        index=close.index,
    )

    feature_frame = expanding_zscore(feature_frame)
    common_index = asset_returns.dropna().index.intersection(feature_frame.dropna().index)

    return MarketData(
        prices=asset_prices.loc[common_index],
        asset_returns=asset_returns.loc[common_index],
        features=feature_frame.loc[common_index],
    )


def expanding_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    expanding_mean = frame.expanding(min_periods=60).mean()
    expanding_std = frame.expanding(min_periods=60).std().replace(0.0, np.nan)
    z = (frame - expanding_mean) / expanding_std
    return z.replace([np.inf, -np.inf], np.nan)


def slice_by_dates(
    market_data: MarketData,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> MarketData:
    mask = (market_data.prices.index >= start) & (market_data.prices.index < end)
    return MarketData(
        prices=market_data.prices.loc[mask],
        asset_returns=market_data.asset_returns.loc[mask],
        features=market_data.features.loc[mask],
    )


def save_backtest_outputs(
    output_dir: Path,
    nav: pd.Series,
    weights: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    nav.to_csv(output_dir / "nav.csv")
    weights.to_csv(output_dir / "weights.csv")
    pd.Series(metrics, name="value").to_csv(output_dir / "metrics.csv")
