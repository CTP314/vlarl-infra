import dataclasses
import sys
from typing import Literal
import gymnasium as gym
import tyro
from loguru import logger

import vlarl_infra
from vlarl_infra.utils.registration import REGISTERED_ENV_CONFIGS
from vlarl_infra.envs.base_env import BaseEnvConfig
from vlarl_client.websocket_worker_agent import WebSocketWorkerAgent


@dataclasses.dataclass
class Args:
    uid: tyro.conf._markers.Suppress[str]
    env: BaseEnvConfig
    num_episodes: int = 1
    log_level: Literal["DEBUG", "INFO"] = "INFO"
    
    host: str =  "0.0.0.0"
    port: int = 8000
    
_CONFIGS_DICT = {k.lower(): Args(uid=k, env=v) for k, v in REGISTERED_ENV_CONFIGS.items()}

def cli() -> Args:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})

def _main(args: Args):
    logger.configure(handlers=[{"sink": sys.stdout, "level": args.log_level}])

    logger.info(f"vlarl_infra version: {vlarl_infra.__version__}")
    logger.info(f"Selected env: {args.uid}")
    logger.info(f"Env config: {args.env}")

    try: 
        worker_agent = WebSocketWorkerAgent(host=args.host, port=args.port)
        logger.info(f"Connected to server with metadata: {worker_agent.get_server_metadata()}")
    except Exception as e:
        logger.error(f"Failed to connect to server: {e}")
        return

    env = gym.make(args.uid, config=args.env)

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        logger.info(f"Episode {ep}:")
        logger.info("  Info:", info)
        
        terminated = False
        truncated = False
        step_count = 0
        total_reward = 0.0
        
        while not (terminated or truncated):
            action_data = worker_agent.infer(dataclasses.asdict(obs))
            action = action_data["action"]
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            total_reward += float(reward)

            worker_agent.feedback(dataclasses.asdict(obs), float(reward), terminated, truncated, info)

            if step_count % 100 == 0 or terminated or truncated:
                logger.debug(f"    Step {step_count}: reward={reward}, terminated={terminated}, truncated={truncated}")

        logger.info(f"  Episode {ep} finished after {step_count} steps with total reward {total_reward}")
    
def main():
    _main(cli())