from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from .data import MarketData


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
        self.weights = np.zeros(self.n_actions, dtype=np.float64)
        self.weights[-1] = 1.0
        self.portfolio_value = self.initial_cash
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

    def step(self, action: np.ndarray):
        prev_prices = self.market_data.prices.iloc[self.ptr - 1].values
        next_prices = self.market_data.prices.iloc[self.ptr].values

        target_weights = _softmax(action.astype(np.float64))
        asset_returns = (next_prices / prev_prices) - 1.0
        cash_return = 0.0
        period_returns = np.concatenate([asset_returns, [cash_return]])

        portfolio_return = float(np.dot(target_weights, period_returns))
        self.portfolio_value *= 1.0 + portfolio_return

        reward = differential_sharpe_ratio(
            portfolio_return,
            A=self.A,
            B=self.B,
            eta=self.reward_eta,
        )
        self.A = self.A + self.reward_eta * (portfolio_return - self.A)
        self.B = self.B + self.reward_eta * (portfolio_return**2 - self.B)

        self.weights = target_weights
        self.ptr += 1
        self.nav_history.append(self.portfolio_value)

        terminated = self.ptr >= len(self.market_data.prices)
        truncated = False
        obs = self._get_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "weights": self.weights.copy(),
        }
        return obs, reward, terminated, truncated, info


def differential_sharpe_ratio(ret: float, A: float, B: float, eta: float) -> float:
    delta_A = ret - A
    delta_B = ret**2 - B
    denom_sq = max(B - A**2, 1e-12)
    denom = denom_sq ** 1.5
    return float((B * delta_A - 0.5 * A * delta_B) / denom)


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp = np.exp(z)
    return exp / exp.sum()


def run_policy_backtest(
    env: PortfolioEnv,
    policy_fn,
) -> tuple[pd.Series, pd.DataFrame]:
    obs, _ = env.reset()
    done = False
    weights = []
    dates = list(env.market_data.prices.index[env.lookback - 1 :])

    while not done:
        action = policy_fn(obs)
        obs, _, done, _, info = env.step(action)
        weights.append(info["weights"])

    nav = pd.Series(env.nav_history, index=dates[: len(env.nav_history)], name="nav")
    weight_df = pd.DataFrame(weights, index=dates[1 : 1 + len(weights)])
    weight_df.columns = list(env.market_data.prices.columns) + ["CASH"]
    return nav, weight_df
