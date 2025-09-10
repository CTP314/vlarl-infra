import dataclasses
import sys
from typing import Literal
import gymnasium as gym
import tyro
from loguru import logger

import vlarl_infra
from vlarl_infra.utils.registration import REGISTERED_ENV_CONFIGS
from vlarl_infra.envs.base_env import BaseEnvConfig


@dataclasses.dataclass
class Args:
    uid: tyro.conf._markers.Suppress[str]
    env: BaseEnvConfig
    num_episodes: int = 1
    log_level: Literal["DEBUG", "INFO"] = "INFO"
    
_CONFIGS_DICT = {k.lower(): Args(uid=k, env=v) for k, v in REGISTERED_ENV_CONFIGS.items()}

def cli() -> Args:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})

def _main(args: Args):
    logger.configure(handlers=[{"sink": sys.stdout, "level": args.log_level}])

    logger.info(f"vlarl_infra version: {vlarl_infra.__version__}")
    logger.info(f"Selected env: {args.uid}")
    logger.info(f"Env config: {args.env}")

    env = gym.make(args.uid, config=args.env)

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        logger.info(f"Episode {ep}:")
        logger.info("  Info:", info)
        
        done = False
        truncated = False
        step_count = 0
        total_reward = 0.0
        
        while not (done or truncated):
            action = env.unwrapped.fake_action()
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1
            total_reward += reward
            
            if step_count % 100 == 0 or done or truncated:
                logger.debug(f"    Step {step_count}: reward={reward}, done={done}, truncated={truncated}")
        
        logger.info(f"  Episode {ep} finished after {step_count} steps with total reward {total_reward}")
    
def main():
    _main(cli())