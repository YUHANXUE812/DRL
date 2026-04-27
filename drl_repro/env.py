from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from .data import MarketData


def action_to_target_weights(action: np.ndarray) -> np.ndarray:
    z = action.astype(np.float64) - np.max(action)
    exp = np.exp(z)
    return exp / exp.sum()


def compute_portfolio_value(shares: np.ndarray, cash: float, prices: np.ndarray) -> float:
    return float(np.dot(shares.astype(np.float64), prices.astype(np.float64)) + cash)


def compute_portfolio_weights(shares: np.ndarray, cash: float, prices: np.ndarray) -> np.ndarray:
    portfolio_value = compute_portfolio_value(shares, cash, prices)
    if portfolio_value <= 0.0:
        weights = np.zeros(len(shares) + 1, dtype=np.float64)
        weights[-1] = 1.0
        return weights

    asset_values = shares.astype(np.float64) * prices.astype(np.float64)
    weights = np.zeros(len(shares) + 1, dtype=np.float64)
    weights[:-1] = asset_values / portfolio_value
    weights[-1] = cash / portfolio_value
    return weights


def rebalance_to_target_weights(
    portfolio_value: float,
    prices: np.ndarray,
    target_weights: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    asset_budget = portfolio_value * target_weights[:-1]
    safe_prices = np.maximum(prices.astype(np.float64), 1e-12)
    shares = np.floor(asset_budget / safe_prices).astype(np.int64)
    asset_value = float(np.dot(shares.astype(np.float64), safe_prices))
    cash = float(portfolio_value - asset_value)
    actual_weights = compute_portfolio_weights(shares, cash, safe_prices)
    return shares, cash, actual_weights


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        market_data: MarketData,
        lookback: int,
        initial_cash: float,
        reward_eta: float,
    ) -> None:
        super().__init__()
        self.market_data = market_data
        self.lookback = lookback
        self.initial_cash = initial_cash
        self.reward_eta = reward_eta

        self.asset_tickers = list(market_data.prices.columns)
        self.n_assets = len(self.asset_tickers)
        self.n_actions = self.n_assets + 1  # + cash

        obs_shape = (self.n_actions, self.lookback + 1)
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=obs_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.n_actions,),
            dtype=np.float32,
        )

        self._reset_state()

    def _reset_state(self) -> None:
        self.ptr = self.lookback
        self.shares = np.zeros(self.n_assets, dtype=np.int64)
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.weights = np.zeros(self.n_actions, dtype=np.float64)
        self.weights[-1] = 1.0
        self.nav_history: list[float] = [self.initial_cash]
        self.A = 0.0
        self.B = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        ret_window = self.market_data.asset_returns.iloc[self.ptr - self.lookback : self.ptr].T.values
        feats = self.market_data.features.iloc[self.ptr].values.astype(np.float64)
        obs = np.zeros((self.n_actions, self.lookback + 1), dtype=np.float32)
        obs[:, 0] = self.weights.astype(np.float32)
        obs[:-1, 1:] = ret_window.astype(np.float32)
        obs[-1, 1:4] = feats.astype(np.float32)
        return obs

    def _advance_with_target_weights(self, target_weights: np.ndarray):
        current_prices = self.market_data.prices.iloc[self.ptr].values.astype(np.float64)
        next_prices = self.market_data.prices.iloc[self.ptr + 1].values.astype(np.float64)

        current_value = compute_portfolio_value(self.shares, self.cash, current_prices)
        shares, cash, executed_weights = rebalance_to_target_weights(current_value, current_prices, target_weights)

        next_value = compute_portfolio_value(shares, cash, next_prices)
        portfolio_return = float(next_value / max(current_value, 1e-12) - 1.0)

        reward = differential_sharpe_ratio(
            portfolio_return,
            A=self.A,
            B=self.B,
            eta=self.reward_eta,
        )
        self.A = self.A + self.reward_eta * (portfolio_return - self.A)
        self.B = self.B + self.reward_eta * (portfolio_return**2 - self.B)

        self.shares = shares
        self.cash = cash
        self.portfolio_value = next_value
        self.ptr += 1
        self.weights = compute_portfolio_weights(self.shares, self.cash, next_prices)
        self.nav_history.append(self.portfolio_value)

        terminated = self.ptr + 1 >= len(self.market_data.prices)
        truncated = False
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "weights": self.weights.copy(),
            "executed_weights": executed_weights.copy(),
            "cash": self.cash,
            "shares": self.shares.copy(),
        }
        return obs, reward, terminated, truncated, info

    def step(self, action: np.ndarray):
        target_weights = action_to_target_weights(action)
        return self._advance_with_target_weights(target_weights)

    def trade_with_target_weights(self, target_weights: np.ndarray):
        return self._advance_with_target_weights(target_weights.astype(np.float64))


def differential_sharpe_ratio(ret: float, A: float, B: float, eta: float) -> float:
    delta_A = ret - A
    delta_B = ret**2 - B
    denom_sq = max(B - A**2, 1e-12)
    denom = denom_sq ** 1.5
    return float((B * delta_A - 0.5 * A * delta_B) / denom)


def run_policy_backtest(
    env: PortfolioEnv,
    policy_fn,
) -> tuple[pd.Series, pd.DataFrame]:
    obs, _ = env.reset()
    done = False
    executed_weights = []
    nav_dates = list(env.market_data.prices.index[env.lookback:])
    trade_dates = list(env.market_data.prices.index[env.lookback:-1])

    while not done:
        target_weights = policy_fn(obs, env)
        obs, _, done, _, info = env.trade_with_target_weights(target_weights)
        executed_weights.append(info["executed_weights"])

    nav = pd.Series(env.nav_history, index=nav_dates[: len(env.nav_history)], name="nav")
    weight_df = pd.DataFrame(executed_weights, index=trade_dates[: len(executed_weights)])
    weight_df.columns = list(env.market_data.prices.columns) + ["CASH"]
    return nav, weight_df
