import os
os.environ["MUJOCO_GL"] = "egl"

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import json

from vlarl_infra import PACKAGE_DIR

env_name = "transport-img"

env_meta_json_path = PACKAGE_DIR / "envs" / "robomimic" / "env_meta" / f"{env_name}.json"

env_meta = json.load(open(env_meta_json_path, 'r'))
obs_modality_dict = {
    "low_dim": (
        'robot0_eef_pos',
        'robot0_eef_quat',
        'robot0_gripper_qpos',
        'object'
    ),
}

ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)

env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    render=True,
    render_offscreen=True,
    use_image_obs=True,
)

env.env.hard_reset = False

while True:
    obs = env.reset()
    import ipdb; ipdb.set_trace()
    done = False
    while not done:
        action = [0] * 7
        obs, reward, done, info = env.step(action)
        env.render()
