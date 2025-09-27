import dataclasses
import numpy as np
import gymnasium as gym

try:
    import pygame
except ImportError:
    raise ImportError('pygame is not installed. Please install it with pip install "vlarl-infra[classic]".')

from vlarl_infra.envs.base_env import BaseEnv, BaseEnvConfig, Action, Observation
from vlarl_infra.utils.registration import register_env, register_env_config

UID = "Classic-v1"

@register_env_config(UID)
@dataclasses.dataclass
class ClassicConfig(BaseEnvConfig):
    classic_env_name: str = "CartPole-v1"
    
@register_env(UID, max_episode_steps=10000)
class ClassicEnv(BaseEnv):
    env: gym.Env
    
    def __init__(self, config: ClassicConfig):
        super().__init__(config=config)
        env = gym.make(config.classic_env_name, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        self.env = env
        self.game_name = config.classic_env_name
        
    def prepare_obs(self, obs: np.ndarray) -> Observation:
        frame = self.env.render()
        frames = {
            f"env": np.array(frame)[None, ...]
        }
        states = {"obs": obs[None, ...]}
        return Observation(
            images=frames,
            states=states,
            text=self.game_name,
        )
        
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[Observation | None, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.prepare_obs(obs), info
    
    def step(self, action: Action) -> tuple[Observation | None, float, bool, bool, dict]:
        action = int(action.item())
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.prepare_obs(obs), float(reward), terminated, truncated, info