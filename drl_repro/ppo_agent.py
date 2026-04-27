from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor

from .config import ExperimentConfig
from .data import MarketData
from .env import PortfolioEnv, action_to_target_weights


def build_ppo_env(market_data: MarketData, config: ExperimentConfig, seed: int, eval_mode: bool = False) -> VecEnv:
    def make_env(rank: int):
        def _init():
            env = PortfolioEnv(
                market_data=market_data,
                lookback=config.lookback,
                initial_cash=config.initial_cash,
                reward_eta=config.reward_eta,
            )
            env.reset(seed=seed + rank)
            return env

        return _init

    n_envs = 1 if eval_mode else config.ppo_n_envs
    env_fns = [make_env(rank) for rank in range(n_envs)]

    if eval_mode or not config.use_subproc_vecenv or n_envs == 1:
        return VecMonitor(DummyVecEnv(env_fns))
    return VecMonitor(SubprocVecEnv(env_fns))


def make_learning_rate_schedule(initial_value: float, final_value: float):
    def schedule(progress_remaining: float) -> float:
        return final_value + progress_remaining * (initial_value - final_value)

    return schedule


def build_ppo_model(env: VecEnv, config: ExperimentConfig, seed: int) -> PPO:
    policy_kwargs = {
        "net_arch": list(config.policy_hidden_sizes),
        "log_std_init": config.ppo_log_std_init,
        "activation_fn": torch.nn.Tanh,
    }
    learning_rate = make_learning_rate_schedule(
        config.ppo_learning_rate,
        config.ppo_learning_rate_final,
    )
    return PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=config.ppo_n_steps,
        batch_size=config.ppo_batch_size,
        n_epochs=config.ppo_n_epochs,
        gamma=config.ppo_gamma,
        gae_lambda=config.ppo_gae_lambda,
        clip_range=config.ppo_clip_range,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
    )


def clone_policy_state_dict(model: PPO) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.policy.state_dict().items()}


def train_ppo(
    train_data: MarketData,
    val_data: MarketData,
    config: ExperimentConfig,
    seed: int,
    output_dir: Path,
    warm_start_policy_state: dict[str, torch.Tensor] | None = None,
) -> tuple[PPO, float, dict[str, torch.Tensor]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_ppo_env(train_data, config, seed=seed, eval_mode=False)
    eval_env = build_ppo_env(val_data, config, seed=seed, eval_mode=True)
    model = build_ppo_model(train_env, config, seed=seed)
    if warm_start_policy_state is not None:
        model.policy.load_state_dict(warm_start_policy_state)

    eval_freq = max(config.eval_frequency // max(config.ppo_n_envs, 1), 1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        log_path=str(output_dir),
        eval_freq=eval_freq,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        verbose=0,
    )

    model.learn(total_timesteps=config.train_steps, callback=eval_callback, progress_bar=False)

    best_reward = float(eval_callback.best_mean_reward)
    best_model_path = output_dir / "best_model.zip"
    if best_model_path.exists():
        best_model = PPO.load(str(best_model_path), env=train_env)
        model.policy.load_state_dict(best_model.policy.state_dict())
        best_model.env.close()

    eval_env.close()
    return model, best_reward, clone_policy_state_dict(model)


def model_policy_fn(model: PPO):
    def policy(obs: np.ndarray, _env: PortfolioEnv) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action_to_target_weights(np.asarray(action))

    return policy
