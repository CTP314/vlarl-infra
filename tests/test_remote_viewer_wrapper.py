import gymnasium as gym
import time
import numpy as np
from vlarl_infra.utils.wrappers.remote_viewer_wrapper import RemoteViewerWrapper
from vlarl_infra.envs.base_env import BaseEnv, Action, Observation # 确保这些类被正确导入

# 假设你的 MyCustomEnv 类继承自 BaseEnv
# 这里仅为演示目的，用一个模拟类代替
class MyCustomEnv(BaseEnv):
    def __init__(self):
        # 模拟一个符合 Observation 约定的环境
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "images": gym.spaces.Dict({"rgb": gym.spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8)}),
            "states": gym.spaces.Dict({"robot_pose": gym.spaces.Box(0, 1, shape=(6,), dtype=np.float32)}),
            "text": gym.spaces.Text(100)
        })

    def reset(self, **kwargs):
        # 模拟 reset 返回一个 Observation 对象
        return Observation(
            images={"rgb": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)},
            states={"robot_pose": np.random.rand(6).astype(np.float32)},
            text="Env reset!"
        ), {}

    def step(self, action):
        # 模拟 step 返回 Observation 和其他信息
        obs = Observation(
            images={"rgb": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)},
            states={"robot_pose": np.random.rand(6).astype(np.float32)},
            text=f"Step with action {action}"
        )
        return obs, 0.5, False, False, {}

    def close(self):
        print("Environment closed.")

# 1. 实例化你的环境
env = MyCustomEnv()

# 2. 用 RemoteViewerWrapper 包装你的环境
# 这里的 URI 必须指向你的 WebSocket 服务器地址
viewer_env = RemoteViewerWrapper(env, websocket_uri="ws://localhost:8001/ws/env")

# 3. 运行你的训练/测试循环
print("Starting environment loop...")
obs, info = viewer_env.reset()

for i in range(10):
    action = viewer_env.action_space.sample()
    obs, reward, terminated, truncated, info = viewer_env.step(action)
    print(f"Step {i+1}: Obs sent. Reward: {reward}")
    time.sleep(0.5)

# 4. 循环结束后，关闭环境
viewer_env.close()
print("Environment loop finished.")