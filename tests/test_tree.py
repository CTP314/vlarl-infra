import tree
import numpy as np

# 假设有两个这样的数据样本
sample_1 = {
    'images': {
        'base': np.random.rand(1, 3, 224, 224),
        'wrist': np.random.rand(1, 3, 100, 100)
    },
    'states': {
        'robot_state': np.random.rand(1, 10),
        'joint_angles': np.random.rand(1, 7)
    },
    'instruction': 'pick up the red block'
}

sample_2 = {
    'images': {
        'base': np.random.rand(1, 3, 224, 224),
        'wrist': np.random.rand(1, 3, 100, 100)
    },
    'states': {
        'robot_state': np.random.rand(1, 10),
        'joint_angles': np.random.rand(1, 7)
    },
    'instruction': 'place it on the shelf'
}

# 将多个样本放入一个列表中
list_of_samples = [sample_1, sample_2]

# 定义一个合并函数
def merge_fn(*nodes):
    # 检查第一个节点，以确定类型
    first_node = nodes[0]
    
    if isinstance(first_node, np.ndarray):
        # 如果是 NumPy 数组，在第 0 维上拼接
        return np.concatenate(nodes, axis=0)
    elif isinstance(first_node, str):
        # 如果是字符串，则直接返回一个列表
        return list(nodes)
    else:
        # 对于其他类型，抛出错误或按需处理
        raise TypeError(f"Unsupported node type: {type(first_node)}")

# 使用 tree.map_structure 来合并整个结构
batch = tree.map_structure(merge_fn, *list_of_samples)

print("合并后的图像形状:", batch['images']['base'].shape)
print("合并后的状态形状:", batch['states']['robot_state'].shape)
print("合并后的指令列表:", batch['instruction'])