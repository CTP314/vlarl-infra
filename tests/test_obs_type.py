# 文件: robot_data.py

from typing import TypedDict, Dict, Annotated, TypeVar
import numpy as np
import numpy.typing as npt

# 定义一个类型变量，让类型更具通用性
DType = TypeVar("DType", bound=np.generic)

# 使用 Annotated 来给 ndarray 添加形状信息
# 这里我们用字符串来命名维度，方便理解
# "b" 代表批量维度（可以是不定长）
ImageArray = Annotated[npt.NDArray[DType], ("b", "h", "w", "c")]
StateArray = Annotated[npt.NDArray[DType], ("b", "d")]

# 使用 TypedDict 来定义数据结构
class Observation(TypedDict):
    images: Dict[str, ImageArray]
    states: Dict[str, StateArray]
    instruction: str

# 一个使用类型注解的函数
def process_observation(obs: Observation):
    """
    处理一个机器人观察数据，类型注解会确保数据结构正确。
    """
    # 你的处理逻辑...
    print("数据类型和形状检查通过！")
    print(f"图像尺寸: {obs['images']['base'].shape}")
    print(f"状态尺寸: {obs['states']['robot_state'].shape}")
    return obs

# --- 正确的例子 ---
# 这个数据符合 Observation 的类型定义
correct_data: Observation = {
    'images': {
        'base': np.zeros((16, 224, 224, 3), dtype=np.float32),
        'wrist': np.zeros((16, 100, 100, 3), dtype=np.float32)
    },
    'states': {
        'robot_state': np.zeros((16, 10), dtype=np.float32),
        'joint_angles': np.zeros((16, 7), dtype=np.float32)
    },
    'instruction': 'pick up the red block'
}
process_observation(correct_data)

# --- 错误的例子 ---
# 这个数据不符合类型定义
incorrect_data: Observation = {
    'images': {
        # 错误：图像通道数不正确，应该是 3
        'base': np.zeros((16, 224, 224, 4), dtype=np.float32),
        'wrist': np.zeros((16, 100, 100, 3), dtype=np.float32)
    },
    'states': {
        'robot_state': np.zeros((16, 10), dtype=np.float32),
        'joint_angles': np.zeros((16, 7), dtype=np.float32)
    },
    # 错误：缺少 instruction 键
}
# 我们故意不在这里调用 process_observation(incorrect_data)
# 而是让 Mypy 来检查