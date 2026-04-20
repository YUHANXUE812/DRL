from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import ExperimentConfig
from .env import PortfolioEnv
from .data import MarketData


def build_ppo_env(market_data: MarketData, config: ExperimentConfig) -> DummyVecEnv:
    def make_env():
        return PortfolioEnv(
            market_data=market_data,
            lookback=config.lookback,
            initial_cash=config.initial_cash,
            reward_eta=config.reward_eta,
        )

    return DummyVecEnv([make_env for _ in range(config.ppo_n_envs)])


def train_ppo(market_data: MarketData, config: ExperimentConfig) -> PPO:
    env = build_ppo_env(market_data, config)
    policy_kwargs = {
        "net_arch": list(config.policy_hidden_sizes),
        "log_std_init": config.ppo_log_std_init,
    }
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.ppo_learning_rate,
        n_steps=config.ppo_n_steps,
        batch_size=config.ppo_batch_size,
        n_epochs=config.ppo_n_epochs,
        gamma=config.ppo_gamma,
        gae_lambda=config.ppo_gae_lambda,
        clip_range=config.ppo_clip_range,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=config.seed,
    )
    model.learn(total_timesteps=config.train_steps, progress_bar=False)
    return model


def model_policy_fn(model: PPO):
    def policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return np.asarray(action)

    return policy
