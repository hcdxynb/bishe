import numpy as np


def skew_symmetric(v):
    """向量 v 的反对称矩阵，满足 k×x = [k]_x x"""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def so3_exp(v):
    """SO(3) 指数映射，小角误差向量到旋转矩阵"""
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)

    k = v / theta
    K = skew_symmetric(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
