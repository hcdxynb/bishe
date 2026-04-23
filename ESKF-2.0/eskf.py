# 导入相关库
from typing import Tuple, Sequence, Any
from dataclasses import dataclass, field
from cat_slice import CatSlice
import numpy as np
import scipy.linalg as la
import math  
from quaternion import (
    euler_to_quaternion,
    quaternion_product,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
)

# 从 utils.py 导入 cross_product_matrix 函数 用于叉乘矩阵的构造
from utils import cross_product_matrix

# 索引 返回cat_slice对象，便于后续切片操作

# 名义状态
POS_IDX = CatSlice(start=0, stop=3) # 返回[0,1,2]，对应位置索引
VEL_IDX = CatSlice(start=3, stop=6)
ATT_IDX = CatSlice(start=6, stop=10) # 四元数占4维，索引为[6,7,8,9]
ACC_BIAS_IDX = CatSlice(start=10, stop=13)
GYRO_BIAS_IDX = CatSlice(start=13, stop=16)

# 误差状态
ERR_ATT_IDX = CatSlice(start=6, stop=9)
ERR_ACC_BIAS_IDX = CatSlice(start=9, stop=12)
ERR_GYRO_BIAS_IDX = CatSlice(start=12, stop=15)

# ESKF 类定义
@dataclass
class ESKF:
    # 用于构造Q_err的参数
    sigma_acc: float
    sigma_gyro: float 
    sigma_acc_bias: float 
    sigma_gyro_bias: float 

    p_acc: float = 0 
    p_gyro: float = 0 

    S_a: np.ndarray = np.eye(3) # 加速度测量标定矩阵，默认为单位矩阵（暂时不做修正）
    S_g: np.ndarray = np.eye(3) # 角速度测量标定矩阵，默认为单位矩阵（暂时不做修正）

    g: np.ndarray = np.array([0, 0, 9.82])  # 在 NED 坐标系中重力向下为正
    Q_err: np.array = field(init=False, repr=False) # 误差状态噪声协方差矩阵，由 sigma_* 参数在 __post_init__ 中构造

# 构造Q_err矩阵 精细化设计Q
    def __post_init__(self):
        # Q构造实现
        self.Q_err = (
            la.block_diag(
                self.sigma_acc * np.eye(3),
                self.sigma_gyro * np.eye(3),
                self.sigma_acc_bias * np.eye(3),
                self.sigma_gyro_bias * np.eye(3),
            )
            ** 2
        )

# 名义状态预测
    def predict_nominal(
        self,
        x_nominal: np.ndarray,
        acceleration: np.ndarray,
        omega: np.ndarray,
        Ts: float,
    ) -> np.ndarray:
        """
        离散时间名义状态预测，对应公式 (10.58)。

        Args:
            x_nominal (np.ndarray): 待预测的名义状态，形状为 (16,)
            acceleration (np.ndarray): 预测时间段内机体系估计加速度，形状为 (3,)
            omega (np.ndarray): 预测时间段内机体系估计角速度，形状为 (3,)
            Ts (float): 采样时间

        Returns:
            np.ndarray: 预测后的名义状态，形状为 (16,)
        """
        # 提取各状态分量
        position = x_nominal[POS_IDX]
        velocity = x_nominal[VEL_IDX]
        quaternion = x_nominal[ATT_IDX]
        acceleration_bias = x_nominal[ACC_BIAS_IDX]
        gyroscope_bias = x_nominal[GYRO_BIAS_IDX]
        
        # 预测位置与速度
        R = quaternion_to_rotation_matrix(quaternion)
        acceleration=R@acceleration+self.g
        position_prediction=position+Ts*velocity+Ts**2/2*acceleration#加速度模型可能仍需进一步核对
        velocity_prediction = velocity+Ts*acceleration #同上，可能仍需进一步核对

        k=Ts*omega # 机体系局部旋转向量增量
        absk=la.norm(k) # 取模长（应为非负）

        # 四元数更新：q_pred = q ⊗ exp(k/2)
        exp_kdiv2 = np.array(np.array([np.cos(absk/2),*(np.sin(absk/2)*k.T/absk)]))  # 待办：计算预测四元数
        quaternion_prediction = quaternion_product(quaternion, exp_kdiv2)

        # 四元数归一化
        quaternion_prediction/=la.norm(quaternion_prediction) # 待办：归一化
        acceleration_bias_prediction=(1 - Ts * self.p_acc) * acceleration_bias
        gyroscope_bias_prediction=(1 - Ts * self.p_gyro) * gyroscope_bias

        # 首尾拼接预测结果
        x_nominal_predicted = np.concatenate(
            (
                position_prediction,
                velocity_prediction,
                quaternion_prediction,
                acceleration_bias_prediction,
                gyroscope_bias_prediction,
            )
        )
        return x_nominal_predicted

# 连续时间误差状态动力学雅可比矩阵 A 用来推导后续的离散误差状态转移矩阵
    def Aerr(
        self, x_nominal: np.ndarray, acceleration: np.ndarray, omega: np.ndarray,
    ) -> np.ndarray:
        """计算连续时间误差状态动力学雅可比矩阵。

        Args:
            x_nominal (np.ndarray): 名义状态，形状为 (16,)
            acceleration (np.ndarray): 预测时间段内估计加速度，形状为 (3,)
            omega (np.ndarray): 预测时间段内估计角速度，形状为 (3,)

        Returns:
            np.ndarray: 连续时间误差状态动力学雅可比矩阵，形状为 (15, 15)
        """
        # 旋转矩阵
        R = quaternion_to_rotation_matrix(x_nominal[ATT_IDX])

        # 分配矩阵
        A = np.zeros((15, 15))

        # 填充各子块
        A[POS_IDX * VEL_IDX] = np.eye(3) #已完成
        A[VEL_IDX * ERR_ATT_IDX] = -R@cross_product_matrix(acceleration)
        A[VEL_IDX * ERR_ACC_BIAS_IDX] = -R
        A[ERR_ATT_IDX * ERR_ATT_IDX] = -cross_product_matrix(omega)
        A[ERR_ATT_IDX * ERR_GYRO_BIAS_IDX] = -np.eye(3)
        A[ERR_ACC_BIAS_IDX * ERR_ACC_BIAS_IDX] = -self.p_acc*np.eye(3)
        A[ERR_GYRO_BIAS_IDX * ERR_GYRO_BIAS_IDX] = -self.p_gyro*np.eye(3)
    
        # 偏置修正
        A[VEL_IDX * ERR_ACC_BIAS_IDX] = A[VEL_IDX * ERR_ACC_BIAS_IDX] @ self.S_a
        A[ERR_ATT_IDX * ERR_GYRO_BIAS_IDX] = (
            A[ERR_ATT_IDX * ERR_GYRO_BIAS_IDX] @ self.S_g
        )
        return A

# 连续时间误差状态噪声输入矩阵 用于推导后续的离散误差状态噪声协方差矩阵Q
    def Gerr(self, x_nominal: np.ndarray,) -> np.ndarray:
        """计算连续时间误差状态噪声输入矩阵。

        Args:
            x_nominal (np.ndarray): 名义状态，形状为 (16,)

        Returns:
            np.ndarray: 连续时间误差状态噪声输入矩阵，形状为 (15, 12)
        """
        R = quaternion_to_rotation_matrix(x_nominal[ATT_IDX])

        G = np.zeros((15, 12))
        diagonal=la.block_diag(-R,-np.eye(3),np.eye(3),np.eye(3))
        G=np.vstack([np.zeros((3,12)),diagonal])
        return G

# 离散化误差状态转移矩阵与噪声协方差矩阵
    def discrete_error_matrices(
        self,
        x_nominal: np.ndarray,
        acceleration: np.ndarray,
        omega: np.ndarray,
        Ts: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算离散时间线性化误差状态转移矩阵与噪声协方差。

        Args:
            x_nominal (np.ndarray): 名义状态，形状为 (16,)
            acceleration (np.ndarray): 预测时间段内机体系估计加速度，形状为 (3,)
            omega (np.ndarray): 预测时间段内机体系估计角速度，形状为 (3,)
            Ts (float): 采样时间

        Returns:
            Tuple[np.ndarray, np.ndarray]: 离散误差矩阵二元组 (Ad, GQGd)
                Ad: 离散时间误差状态系统矩阵，形状为 (15, 15)
                GQGd: 离散时间噪声协方差矩阵，形状为 (15, 15)
        """
        A = self.Aerr(x_nominal, acceleration, omega)
        G = self.Gerr(x_nominal)
        V = np.zeros((30, 30)) # 使用 Van Loan 组块矩阵
        V[CatSlice(0,15)*CatSlice(0,15)]=-A*Ts
        V[CatSlice(0,15)*CatSlice(15,30)]=G@self.Q_err@G.T*Ts
        V[CatSlice(15,30)*CatSlice(15,30)]=A.T*Ts

        #VanLoanMatrix = la.expm(V)  # 该写法更精确，但速度较慢
        VanLoanMatrix=np.eye(30)+V+1/2.0*V@V

        V1_T = VanLoanMatrix[CatSlice(15,30)**2].T # 即 Ad = exp(A' * deltaT)
        V2 = VanLoanMatrix[CatSlice(0,15)*CatSlice(15,30)]

        # 由 Van Loan 结果得到离散噪声协方差 Qd
        Ad = V1_T
        GQGd = V1_T@V2 # Qd = V1' * V2（定理 4.5.2）
        
        return Ad, GQGd
    
# Q自适应调整
    def Q_adaptation(self, P: np.ndarray, W: np.ndarray, GQGd: np.ndarray, v_prior: np.ndarray, v_post: np.ndarray, GNSSk: int) -> np.ndarray:
        """根据 GNSS 测量残差对误差状态噪声协方差矩阵进行自适应调整。

        Args:
            P (np.ndarray): 当前误差状态协方差矩阵，形状为 (15, 15)
            W (np.ndarray): 当前卡尔曼增益，形状为 (15, 3)
            GQGd (np.ndarray): 原始离散时间噪声协方差矩阵，形状为 (15, 15)
            v_prior (np.ndarray): 当前 GNSS 测量的先验残差，形状为 (3,)
            v_post (np.ndarray): 当前 GNSS 测量的后验残差，形状为 (3,)
            GNSSk (int): 当前 GNSS 测量索引

        Returns:
            np.ndarray: 调整后的离散时间噪声协方差矩阵，形状为 (15, 15)
        """
        # 待办：自适应调整 Q 的实现
        b = 0.95
        d = (1-b)/(1-b**GNSSk)

        GQGd_adjusted = (1-d)* GQGd + d*(W@np.outer(v_prior[GNSSk], v_prior[GNSSk])@W.T) #注意是外积

        return GQGd_adjusted

# 误差协方差矩阵P预测
    def predict_covariance(
        self,
        x_nominal: np.ndarray,
        P: np.ndarray,
        acceleration: np.ndarray,
        omega: np.ndarray,
        Ts: float,
        W: np.ndarray,
        GNSSk: int,
        v_prior: np.ndarray,
        v_post: np.ndarray,
        do_auto: bool

    ) -> np.ndarray:
        """利用线性化连续时间动力学，将误差协方差向前预测 Ts 时间。

        Args:
            x_nominal (np.ndarray): 名义状态，形状为 (16,)
            P (np.ndarray): 误差状态协方差，形状为 (15, 15)
            acceleration (np.ndarray): 预测时间段内估计加速度，形状为 (3,)
            omega (np.ndarray): 预测时间段内估计角速度，形状为 (3,)
            Ts (float): 采样时间
            W (np.ndarray): 卡尔曼增益，形状为 (15, 3)，用于自适应调整 Q
            GNSSk (int): 当前 GNSS 测量索引（用于自适应调整）
            v_prior (np.ndarray): 当前 GNSS 测量的先验残差，形状为 (3,)
            v_post (np.ndarray): 当前 GNSS 测量的后验残差，形状为 (3,)

        Returns:
            np.ndarray: 预测后的误差状态协方差矩阵，形状为 (15, 15)
        """
        Ad, GQGd = self.discrete_error_matrices(x_nominal, acceleration, omega, Ts)

        if GNSSk > 0 and do_auto:
            GQGd = self.Q_adaptation(P, W, GQGd, v_prior, v_post, GNSSk)

        P_predicted=Ad@P@Ad.T+GQGd

        return P_predicted

# 整体预测接口：利用 IMU 测量向前预测名义状态与误差协方差
    def predict(
        self,
        x_nominal: np.ndarray,
        P: np.ndarray,
        z_acc: np.ndarray,
        z_gyro: np.ndarray,
        Ts: float,
        W: np.ndarray,
        GNSSk: int,
        v_prior: np.ndarray,
        v_post: np.ndarray,
        do_auto: bool
    ) -> Tuple[np.array, np.array]:
        """利用 IMU 测量 z_* 向前预测 Ts 时间的名义状态与误差协方差。

        Args:
            x_nominal (np.ndarray): 待预测的名义状态，形状为 (16,)
            P (np.ndarray): 待预测的误差状态协方差，形状为 (15, 15)
            z_acc (np.ndarray): 预测时间段内加速度测量，形状为 (3,)
            z_gyro (np.ndarray): 预测时间段内角速度测量，形状为 (3,)
            Ts (float): 采样时间
            W (np.ndarray): 卡尔曼增益，形状为 (15, 3)，用于自适应调整 Q
            GNSSk (int): 当前 GNSS 测量索引（用于自适应调整）
            v_prior (np.ndarray): 当前 GNSS 测量的先验残差，形状为 (3,)
            v_post (np.ndarray): 当前 GNSS 测量的后验残差，形状为 (3,)
        Returns:
            Tuple[ np.array, np.array ]: 预测结果二元组 (x_nominal_predicted, P_predicted)
                x_nominal_predicted: 预测后的名义状态，形状为 (16,)
                P_predicted: 预测后的误差状态协方差，形状为 (15, 15)
        """
        # 对测量做标定矩阵修正
        
        r_z_acc = self.S_a @ z_acc
        r_z_gyro = self.S_g @ z_gyro

        # 对偏置做标定矩阵修正
        acc_bias = self.S_a @ x_nominal[ACC_BIAS_IDX]
        gyro_bias = self.S_g @ x_nominal[GYRO_BIAS_IDX]

        # 去除惯导测量偏置
        acceleration = r_z_acc-acc_bias
        omega =r_z_gyro-gyro_bias

        # 执行预测
        x_nominal_predicted = self.predict_nominal(x_nominal,acceleration,omega,Ts)
        P_predicted = self.predict_covariance(x_nominal,P,acceleration,omega,Ts, W, GNSSk,v_prior,v_post, do_auto) # 这里协方差预测基于当前名义状态线性化

        return x_nominal_predicted, P_predicted

# 误差注入
    def inject(
        self, x_nominal: np.ndarray, delta_x: np.ndarray, P: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """将估计得到的误差状态注入名义状态，并对协方差做一致性补偿。

        Args:
            x_nominal (np.ndarray): 待注入误差的名义状态，形状为 (16,)
            delta_x (np.ndarray): 误差状态增量，形状为 (15,)
            P (np.ndarray): 误差状态协方差矩阵

        Returns:
            Tuple[ np.ndarray, np.ndarray ]: 注入结果二元组 (x_injected, P_injected):
                x_injected: 注入后的名义状态，形状为 (16,)
                P_injected: 注入后的误差状态协方差矩阵，形状为 (15, 15)
        """
        ### 便于注入的索引拼接
        # 需要直接加到名义状态中的索引（不含姿态四元数）
        INJ_IDX = POS_IDX + VEL_IDX + ACC_BIAS_IDX + GYRO_BIAS_IDX
        # 对应的误差状态索引（不含姿态误差）
        DTX_IDX = POS_IDX + VEL_IDX + ERR_ACC_BIAS_IDX + ERR_GYRO_BIAS_IDX
        
        x_injected = x_nominal.copy()
        x_injected[INJ_IDX]=x_injected[INJ_IDX]+delta_x[DTX_IDX]
        
        x_injected[ATT_IDX] = quaternion_product(x_nominal[ATT_IDX],np.array([1, *delta_x[ERR_ATT_IDX]/2]))
        # 待办：将误差状态注入名义状态（除姿态四元数外）
        # 待办：姿态注入
        # 待办：四元数归一化
        x_injected[ATT_IDX] = x_injected[ATT_IDX]/la.norm(x_injected[ATT_IDX])

        # 协方差补偿
        G = la.block_diag(np.eye(6),np.eye(3)-cross_product_matrix(delta_x[ERR_ATT_IDX]/2),np.eye(6))  # 待办：对注入后的协方差进行补偿
        P_injected =G@P@G.T
        
        return x_injected, P_injected
    
# GNSS R 适应性调整
    def R_GNSS_adaptation(
        self,
        H: np.ndarray,
        v_prior: np.ndarray,
        v_post: np.ndarray,
        P: np.ndarray,
        R_GNSS: np.ndarray,
        GNSSk: int
        ) -> np.ndarray:
        """根据 GNSS 测量索引 GNSSk 对观测噪声协方差矩阵 R_GNSS 进行适应性调整。

        Args:
            H (np.ndarray): GNSS 位置观测矩阵，形状为 (3, 15)
            v_prior (np.ndarray): GNSS 位置观测的先验残差，形状为 (3,)
            v_post (np.ndarray): GNSS 位置观测的后验残差，形状为 (3,)
            P (np.ndarray): 误差状态协方差矩阵，形状为 (15, 15)
            R_GNSS (np.ndarray): 原始观测噪声协方差矩阵，形状为 (3, 3)
            GNSSk (int): 当前 GNSS 测量索引

        Returns:
            np.ndarray: 调整后的观测噪声协方差矩阵，形状为 (3, 3)
        """
        # 待办：自适应调整 R_GNSS 的实现

        lambda_min = 0.98 #遗忘因子

        L = 0

        if GNSSk > 10 :
            for i in range(10):
                L = L + -(v_prior[GNSSk - i]@v_prior[GNSSk - i].T) #误差调整参数
            L = L/10
        else :
            L = -(v_prior[GNSSk]@v_prior[GNSSk].T) #误差调整参数
        b = lambda_min +(1-lambda_min)*(2**L)
        d = (1-b)/(1-b**GNSSk)

        alpha = (v_prior[GNSSk]@v_prior[GNSSk].T)/np.trace(H@P@H.T+R_GNSS) # 观测残差与理论创新协方差的比值（调整因子）

        R_GNSS = (1-d)* R_GNSS + d*(np.outer(v_post, v_post) + H@P@H.T) #注意是外积

        if ((v_prior[GNSSk]@v_prior[GNSSk].T) > np.trace(H@P@H.T+R_GNSS)) :
            R_GNSS = alpha*R_GNSS
        return R_GNSS
    
    def R_GNSS_sage_husa(
        self,
        H: np.ndarray,
        v_prior: np.ndarray,
        v_post: np.ndarray,
        P: np.ndarray,
        R_GNSS: np.ndarray,
        GNSSk: int
    ) -> np.ndarray:
        """根据 GNSS 测量索引 GNSSk 使用 SAGE-HUSA 方法对观测噪声协方差矩阵 R_GNSS 进行适应性调整。

        Args:
            H (np.ndarray): GNSS 位置观测矩阵，形状为 (3, 15)
            v_prior (np.ndarray): GNSS 位置观测的先验残差，形状为 (3,)
            v_post (np.ndarray): GNSS 位置观测的后验残差，形状为 (3,)
            P (np.ndarray): 误差状态协方差矩阵，形状为 (15, 15)
            R_GNSS (np.ndarray): 原始观测噪声协方差矩阵，形状为 (3, 3)
            GNSSk (int): 当前 GNSS 测量索引

        Returns:
            np.ndarray: 调整后的观测噪声协方差矩阵，形状为 (3, 3)
        """
        # SAGE-HUSA 自适应调整实现
        # 这里只是一个简化的示例，实际应用中可能需要更复杂的实现


        b = 0.9 # 遗忘因子
        d = (1-b)/(1 - b**GNSSk) # 这里的 b 可以根据实际情况调整，通常在 0.9 到 0.99 之间
        R_GNSS = (1 - d) * R_GNSS + d * (np.outer(v_prior[GNSSk], v_prior[GNSSk]) + H@P@H.T)  # 注意是外积
        return R_GNSS

# GNSS 位置观测更新
    def update_GNSS_position(
        self,
        x_nominal: np.ndarray,
        P: np.ndarray,
        R_GNSS: np.ndarray,
        GNSSk: int,
        v_prior: np.ndarray,
        v_post: np.ndarray,
        do_auto: bool,
        do_sage_husa: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """利用 GNSS 位置观测更新状态与协方差。

        Args:
            x_nominal (np.ndarray): 待更新的名义状态，形状为 (16,)
            P (np.ndarray): 待更新的误差状态协方差，形状为 (15, 15)
            R_GNSS (np.ndarray): 观测噪声协方差矩阵，形状为 (3, 3)
            GNSSk (int): 当前 GNSS 测量索引
            v_prior (np.ndarray): 先验残差，形状为 (3,N)
            v_post (np.ndarray): 后验残差，形状为 (3,)
            do_auto (bool): 是否启用自适应调整 R_GNSS
            do_sage_husa (bool): 是否启用 Sage-Husa 风格的自适应调整（仅在 do_auto=True 时有效）

        Returns:
            Tuple[np.ndarray, np.ndarray]: 更新结果二元组 (x_injected, P_injected):
                x_injected: 注入更新后误差状态的名义状态，形状为 (16,)
                P_injected: 误差状态更新并注入后的协方差，形状为 (15, 15)
                R_GNSS_auto: 适应性调整后的观测噪声协方差矩阵，形状为 (3, 3)
                W: 卡尔曼增益，形状为 (15, 3)
        """

        H = np.block([np.eye(3), np.zeros((3,12))])
        if do_auto and GNSSk > 0 :
            R_GNSS_auto = self.R_GNSS_adaptation(H, v_prior, v_post, P, R_GNSS, GNSSk)
        elif do_sage_husa and GNSSk > 0:
            R_GNSS_auto = self.R_GNSS_sage_husa(H, v_prior, v_post, P, R_GNSS, GNSSk)
        else:
            R_GNSS_auto = R_GNSS

        I = np.eye(*P.shape)

        S = H@P@H.T+R_GNSS_auto # 创新协方差

        # 卡尔曼滤波误差状态更新
        W = P@H.T@np.linalg.inv(S) # 待办：卡尔曼增益
        delta_x = W@v_prior[GNSSk] # 待办：误差状态增量

        Jo = I - W @ H  # 约瑟夫形式

        P_update = Jo@P@Jo.T+W@R_GNSS_auto@W.T # 待办：协方差更新

        # 误差状态注入
        x_injected, P_injected = self.inject(x_nominal, delta_x, P_update)

        return x_injected, P_injected, R_GNSS_auto, W

    @classmethod
    def delta_x(cls, x_nominal: np.ndarray, x_true: np.ndarray,) -> np.ndarray:
        """计算 x_nominal 与 x_true 之间的误差状态。

        Args:
            x_nominal (np.ndarray): 名义估计状态，形状为 (16,)
            x_true (np.ndarray): 真实状态，形状为 (16,)

        Returns:
            np.ndarray: 误差状态形式的状态差，形状为 (15,)
        """
        delta_position = x_true[POS_IDX]-x_nominal[POS_IDX]  # 待办：位置误差
        delta_velocity = x_true[VEL_IDX]-x_nominal[VEL_IDX]  # 待办：速度误差

        quaternion_conj = np.diag([1,-1,-1,-1])@x_nominal[ATT_IDX]  # 待办：四元数共轭

        delta_quaternion = quaternion_product(quaternion_conj,x_true[ATT_IDX])  # 待办：姿态误差四元数
        # 由于该误差状态定义，姿态可按此方式计算
        
        delta_theta = 2*delta_quaternion[1:]

        # 拼接偏置索引
        BIAS_IDX = ACC_BIAS_IDX + GYRO_BIAS_IDX
        delta_bias = x_true[BIAS_IDX]-x_nominal[BIAS_IDX] # 待办：偏置误差

        d_x = np.concatenate((delta_position, delta_velocity, delta_theta, delta_bias))

        return d_x