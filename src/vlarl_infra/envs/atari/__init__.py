import os
import json
import pathlib
import numpy as np
import gymnasium as gym

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    raise ImportError(
        "Atari is not installed. Please install it with the 'atari' extra, e.g. 'pip install vlarl_infra[atari]'"
    )
    
from vlarl_infra.envs.base_env import BaseEnv, BaseEnvConfig, Action, Observation
import dataclasses
from vlarl_infra.utils.registration import register_env, register_env_config

UID = "Atari-v1"