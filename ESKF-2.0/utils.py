# 叉乘矩阵工具函数
import numpy as np
from mytypes import ArrayLike


def cross_product_matrix(n: ArrayLike) -> np.ndarray:
    assert len(n) == 3, f"utils.cross_product_matrix: Vector not of length 3: {n}"
    vector = np.array(n, dtype=float).reshape(3)

    #S = np.zeros((3, 3))  # 待办：构造叉乘矩阵
    S=np.array([[0,-vector[2],vector[1]],
               [vector[2],0,-vector[0]],
               [-vector[1],vector[0],0]])
    
    return S
