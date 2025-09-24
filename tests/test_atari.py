import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array")

from vlarl_infra.envs.atari.atari_wrappers import *
import ipdb; ipdb.set_trace()
env = gym.wrappers.RecordEpisodeStatistics(env)
env = NoopResetEnv(env, noop_max=30)
env = MaxAndSkipEnv(env, skip=4)
env = EpisodicLifeEnv(env)
if "FIRE" in env.unwrapped.get_action_meanings():
    env = FireResetEnv(env)
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.GrayscaleObservation(env)
env = gym.wrappers.FrameStackObservation(env, 4)

print(env.action_space)

while True:
    obs, info = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        import ipdb; ipdb.set_trace()