import os
os.environ["MUJOCO_GL"] = "egl"
import json
import pathlib
import numpy as np

try:
    import robomimic
    import robomimic.envs.env_robosuite
    import robomimic.utils.env_utils as _env_utils
    import robomimic.utils.obs_utils as _obs_utils
except:
    raise ImportError(
        "Robomimic is not installed. Please install it with the 'robomimic' extra, e.g. 'pip install vlarl_infra[robomimic]'"
    )
from vlarl_infra.envs.base_env import BaseEnv, BaseEnvConfig, Action, Observation
import dataclasses
from vlarl_infra.utils.registration import register_env, register_env_config

UID = "Robomimic-v1"
ENV_META_DIR = pathlib.Path(__file__).parent / "env_meta"

@register_env_config(UID)
@dataclasses.dataclass
class RobomimicConfig(BaseEnvConfig):
    robomimic_env_name: str = "can-img"
    low_dim_keys: list[str] = dataclasses.field(default_factory=lambda: [
        'robot0_eef_pos',
        'robot0_eef_quat',
        'robot0_gripper_qpos',
        'object'
    ])
    agentview_image_size: tuple[int, int] = (720, 1280)
    
@register_env(UID)
class RobomimicEnv(BaseEnv):
    env: robomimic.envs.env_robosuite.EnvRobosuite
    image_keys: list[str]
    task: str
    agentview_image_size: tuple[int, int]
    
    def __init__(self, config: RobomimicConfig):
        super().__init__(config=config)
        
        try:
            env_meta_json_path = ENV_META_DIR / f"{config.robomimic_env_name}.json"
            env_meta = json.load(open(env_meta_json_path, 'r'))
        except FileNotFoundError:
            raise ValueError(f"Environment metadata file not found for env name: {config.robomimic_env_name}")
        
        _obs_utils.initialize_obs_modality_mapping_from_dict(dict(
            low_dim=config.low_dim_keys,
        ))
        
        env = _env_utils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=True,
            use_image_obs=True,
        )
        assert isinstance(env, robomimic.envs.env_robosuite.EnvRobosuite)
        self.env = env
        
        self.env.env.hard_reset = False
        
        self.image_keys = env_meta['env_kwargs']['camera_names']
        self.task = env_meta['env_name']
        self.agentview_image_size = config.agentview_image_size
    
    def prepare_obs(self, obs, agentview_image) -> Observation:
        images = {}
        for key in self.image_keys:
            images[key] = obs.pop(f"{key}_image")[None]
        images["agentview"] = agentview_image[None]
        states = {k: obs[k][None] for k in obs}
        text = self.task
        return Observation(
            images=images,
            states=states,
            text=text,
        )
        
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple:
        obs = self.env.reset()
        agentview_image = self.render()
        return self.prepare_obs(obs, agentview_image), {}
    
    def step(self, action: Action) -> tuple:
        assert action.shape[0] == 1, "Batch size must be 1 for robomimic env"
        action = action[0].tolist()
        obs, reward, done, info = self.env.step(action)
        agentview_image = self.render()
        return self.prepare_obs(obs, agentview_image), reward, done, False, info
    
    def render(self) -> np.ndarray:
        return self.env.render(
            mode="rgb_array",
            height=self.agentview_image_size[0],
            width=self.agentview_image_size[1],
            camera_name="agentview",
        )