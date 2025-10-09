import dataclasses
import sys
import collections
from typing import Literal
import gymnasium as gym
import tyro
from loguru import logger

import vlarl_infra
from vlarl_infra.utils.registration import REGISTERED_ENV_CONFIGS
from vlarl_infra.envs.base_env import BaseEnvConfig
from vlarl_client.websocket_worker_agent import WebSocketWorkerAgent
import vlarl_infra.utils.wrappers as _wrappers


@dataclasses.dataclass
class Args:
    uid: tyro.conf._markers.Suppress[str]
    env: BaseEnvConfig
    num_episodes: int = 1
    log_level: Literal["debug", "info"] = "info"

    server_host: str =  "0.0.0.0"
    server_port: int = 8000
    
    use_remote_viewer: bool = False
    
    viewer_host: str = "0.0.0.0"
    viewer_port: int = 8001
    
    use_real_time: bool = False
    fps: float = 30.0
    
    replan_steps: int | None = None
    max_episode_steps: int | None = None

_CONFIGS_DICT = {k.lower(): Args(uid=k, env=v) for k, v in REGISTERED_ENV_CONFIGS.items()}

def cli() -> Args:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})

def _main(args: Args):
    logger.configure(handlers=[{"sink": sys.stdout, "level": args.log_level.upper()}])

    logger.info(f"vlarl_infra version: {vlarl_infra.__version__}")
    logger.info(f"Selected env: {args.uid}")
    logger.info(f"Env config: {args.env}")

    try: 
        worker_agent = WebSocketWorkerAgent(host=args.server_host, port=args.server_port)
        logger.info(f"Connected to server with metadata: {worker_agent.get_server_metadata()}")
    except Exception as e:
        logger.error(f"Failed to connect to server: {e}")
        return

    env = gym.make(args.uid, config=args.env, max_episode_steps=args.max_episode_steps)
    if args.use_remote_viewer:
        env = _wrappers.RemoteViewerWrapper(env, websocket_uri=f"ws://{args.viewer_host}:{args.viewer_port}/ws/env")
        logger.info(f"Remote viewer enabled at {args.viewer_host}:{args.viewer_port}")

    if args.use_real_time:
        env = _wrappers.RealTimeWrapper(env, fps=args.fps)
        logger.info(f"Real-time mode enabled at {args.fps} FPS")

    for ep in range(args.num_episodes):
        obs, info = env.reset()
        action_plan = collections.deque()
        logger.info(f"Episode {ep}:")
        logger.info("  Info:", info)
        
        reward, terminated, truncated = 0.0, False, False
        step_count, total_reward, sum_reward = 0, 0., 0.
        
        while not (terminated or truncated):
            if not action_plan:
                sum_reward += float(reward)
                action_data = worker_agent.infer(dataclasses.asdict(obs))
                action_chunk = action_data["action"].swapaxes(1, 0)
                replan_steps = args.replan_steps or len(action_chunk)
                assert (
                    len(action_chunk) >= replan_steps
                ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk)
            action = action_plan.popleft()
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            total_reward += float(reward)        
            sum_reward += float(reward)

            if not action_plan or terminated or truncated:
                worker_agent.feedback(dataclasses.asdict(obs), float(reward), terminated, truncated, info)

            if step_count % 100 == 0 or terminated or truncated:
                logger.debug(f"    Step {step_count}: reward={reward}, terminated={terminated}, truncated={truncated}, info={info}")

        logger.info(f"  Episode {ep} finished after {step_count} steps with total reward {total_reward} and info {info}")
    
def main():
    _main(cli())