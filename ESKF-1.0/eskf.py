# %% 导入相关库
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

# %% 索引 返回cat_slice对象，便于后续切片操作
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



# %% ESKF 类定义
@dataclass
class ESKF:
    # 用于构造Q_err的参数
    sigma_acc: float # 加速度计测量噪声标准差（待传）
    sigma_gyro: float # 陀螺仪测量噪声标准差 （待传）
    sigma_acc_bias: float # 加速度计偏置随机游走标准差（待传）
    sigma_gyro_bias: float  # 陀螺仪偏置随机游走标准差（待传）

    p_acc: float = 0 # 加速度计偏置随机游走参数 (目前为定值)
    p_gyro: float = 0 # 陀螺仪偏置随机游走参数 (目前为定值)

    S_a: np.ndarray = np.eye(3) # 加速度测量标定矩阵，默认为单位矩阵
    S_g: np.ndarray = np.eye(3) # 角速度测量标定矩阵，默认为单位矩阵
    debug: bool = True

    g: np.ndarray = np.array([0, 0, 9.82])  # 在 NED 坐标系中重力向下为正

    Q_err: np.array = field(init=False, repr=False) # 误差状态噪声协方差矩阵，由 sigma_* 参数在 __post_init__ 中构造

# 构造Q_err矩阵
    def __post_init__(self):
        if self.debug:
            print(
                "ESKF 处于调试模式：会进行更多数值检查，但会牺牲计算速度"
            )
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

        Raises:
            AssertionError: 输入形状错误；调试模式下还会检查部分数值性质

        Returns:
            np.ndarray: 预测后的名义状态，形状为 (16,)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.predict_nominal: x_nominal incorrect shape {x_nominal.shape}"
        assert acceleration.shape == (
            3,
        ), f"ESKF.predict_nominal: acceleration incorrect shape {acceleration.shape}"
        assert omega.shape == (
            3,
        ), f"ESKF.predict_nominal: omega incorrect shape {omega.shape}"

        # 提取各状态分量
        position = x_nominal[POS_IDX]
        velocity = x_nominal[VEL_IDX]
        quaternion = x_nominal[ATT_IDX]
        acceleration_bias = x_nominal[ACC_BIAS_IDX]
        gyroscope_bias = x_nominal[GYRO_BIAS_IDX]

        if self.debug:
            assert np.allclose(
                np.linalg.norm(quaternion), 1, rtol=0, atol=1e-15
            ), "ESKF.predict_nominal: Quaternion not normalized."
            assert np.allclose(
                np.sum(quaternion ** 2), 1, rtol=0, atol=1e-15
            ), "ESKF.predict_nominal: Quaternion not normalized and norm failed to catch it."
        
        # 预测位置与速度
        R = quaternion_to_rotation_matrix(quaternion, debug=self.debug)
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

        assert x_nominal_predicted.shape == (
            16,
        ), f"ESKF.predict_nominal: x_nominal_predicted shape incorrect {x_nominal_predicted.shape}"
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

        Raises:
            AssertionError: 输入形状错误；调试模式下还会检查部分数值性质

        Returns:
            np.ndarray: 连续时间误差状态动力学雅可比矩阵，形状为 (15, 15)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.Aerr: x_nominal shape incorrect {x_nominal.shape}"
        assert acceleration.shape == (
            3,
        ), f"ESKF.Aerr: acceleration shape incorrect {acceleration.shape}"
        assert omega.shape == (3,), f"ESKF.Aerr: omega shape incorrect {omega.shape}"

        # 旋转矩阵
        R = quaternion_to_rotation_matrix(x_nominal[ATT_IDX], debug=self.debug)

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

        assert A.shape == (
            15,
            15,
        ), f"ESKF.Aerr: A-error matrix shape incorrect {A.shape}"
        return A

# 连续时间误差状态噪声输入矩阵 用于推导后续的离散误差状态噪声协方差矩阵Q
    def Gerr(self, x_nominal: np.ndarray,) -> np.ndarray:
        """计算连续时间误差状态噪声输入矩阵。

        Args:
            x_nominal (np.ndarray): 名义状态，形状为 (16,)

        Raises:
            AssertionError: 输入形状错误；调试模式下还会检查部分数值性质

        Returns:
            np.ndarray: 连续时间误差状态噪声输入矩阵，形状为 (15, 12)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.Gerr: x_nominal shape incorrect {x_nominal.shape}"

        R = quaternion_to_rotation_matrix(x_nominal[ATT_IDX], debug=self.debug)

        G = np.zeros((15, 12))
        diagonal=la.block_diag(-R,-np.eye(3),np.eye(3),np.eye(3))
        G=np.vstack([np.zeros((3,12)),diagonal])
        assert G.shape == (15, 12), f"ESKF.Gerr: G-matrix shape incorrect {G.shape}"
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

        Raises:
            AssertionError: 输入形状错误；调试模式下还会检查部分数值性质

        Returns:
            Tuple[np.ndarray, np.ndarray]: 离散误差矩阵二元组 (Ad, GQGd)
                Ad: 离散时间误差状态系统矩阵，形状为 (15, 15)
                GQGd: 离散时间噪声协方差矩阵，形状为 (15, 15)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.discrete_error_matrices: x_nominal shape incorrect {x_nominal.shape}"
        assert acceleration.shape == (
            3,
        ), f"ESKF.discrete_error_matrices: acceleration shape incorrect {acceleration.shape}"
        assert omega.shape == (
            3,
        ), f"ESKF.discrete_error_matrices: omega shape incorrect {omega.shape}"

        A = self.Aerr(x_nominal, acceleration, omega)
        G = self.Gerr(x_nominal)
        V = np.zeros((30, 30)) # 使用 Van Loan 组块矩阵
        V[CatSlice(0,15)*CatSlice(0,15)]=-A*Ts
        V[CatSlice(0,15)*CatSlice(15,30)]=G@self.Q_err@G.T*Ts
        V[CatSlice(15,30)*CatSlice(15,30)]=A.T*Ts
        assert V.shape == (
            30,
            30,
        ), f"ESKF.discrete_error_matrices: Van Loan matrix shape incorrect {omega.shape}"
        #VanLoanMatrix = la.expm(V)  # 该写法更精确，但速度较慢
        VanLoanMatrix=np.eye(30)+V+1/2.0*V@V

        V1_T = VanLoanMatrix[CatSlice(15,30)**2].T # 即 Ad = exp(A' * deltaT)
        V2 = VanLoanMatrix[CatSlice(0,15)*CatSlice(15,30)]

        # 由 Van Loan 结果得到离散噪声协方差 Qd
        Ad = V1_T
        GQGd = V1_T@V2 # Qd = V1' * V2（定理 4.5.2）
        
        assert Ad.shape == (
            15,
            15,
        ), f"ESKF.discrete_error_matrices: Ad-matrix shape incorrect {Ad.shape}"
        assert GQGd.shape == (
            15,
            15,
        ), f"ESKF.discrete_error_matrices: GQGd-matrix shape incorrect {GQGd.shape}"

        return Ad, GQGd

# 误差协方差矩阵P预测
    def predict_covariance(
        self,
        x_nominal: np.ndarray,
        P: np.ndarray,
        acceleration: np.ndarray,
        omega: np.ndarray,
        Ts: float,
    ) -> np.ndarray:
        """利用线性化连续时间动力学，将误差协方差向前预测 Ts 时间。

        Args:
            x_nominal (np.ndarray): 名义状态，形状为 (16,)
            P (np.ndarray): 误差状态协方差，形状为 (15, 15)
            acceleration (np.ndarray): 预测时间段内估计加速度，形状为 (3,)
            omega (np.ndarray): 预测时间段内估计角速度，形状为 (3,)
            Ts (float): 采样时间

        Raises:
            AssertionError: 输入形状错误；调试模式下还会检查部分数值性质

        Returns:
            np.ndarray: 预测后的误差状态协方差矩阵，形状为 (15, 15)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.predict_covariance: x_nominal shape incorrect {x_nominal.shape}"
        assert P.shape == (
            15,
            15,
        ), f"ESKF.predict_covariance: P shape incorrect {P.shape}"
        assert acceleration.shape == (
            3,
        ), f"ESKF.predict_covariance: acceleration shape incorrect {acceleration.shape}"
        assert omega.shape == (
            3,
        ), f"ESKF.predict_covariance: omega shape incorrect {omega.shape}"

        Ad, GQGd = self.discrete_error_matrices(x_nominal, acceleration, omega, Ts)

        P_predicted=Ad@P@Ad.T+GQGd

        assert P_predicted.shape == (
            15,
            15,
        ), f"ESKF.predict_covariance: P_predicted shape incorrect {P_predicted.shape}"
        return P_predicted

# 整体预测接口：利用 IMU 测量向前预测名义状态与误差协方差
    def predict(
        self,
        x_nominal: np.ndarray,
        P: np.ndarray,
        z_acc: np.ndarray,
        z_gyro: np.ndarray,
        Ts: float,
    ) -> Tuple[np.array, np.array]:
        """利用 IMU 测量 z_* 向前预测 Ts 时间的名义状态与误差协方差。

        Args:
            x_nominal (np.ndarray): 待预测的名义状态，形状为 (16,)
            P (np.ndarray): 待预测的误差状态协方差，形状为 (15, 15)
            z_acc (np.ndarray): 预测时间段内加速度测量，形状为 (3,)
            z_gyro (np.ndarray): 预测时间段内角速度测量，形状为 (3,)
            Ts (float): 采样时间

        Raises:
            AssertionError: 输入形状错误；调试模式下还会检查部分数值性质

        Returns:
            Tuple[ np.array, np.array ]: 预测结果二元组 (x_nominal_predicted, P_predicted)
                x_nominal_predicted: 预测后的名义状态，形状为 (16,)
                P_predicted: 预测后的误差状态协方差，形状为 (15, 15)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.predict: x_nominal shape incorrect {x_nominal.shape}"
        assert P.shape == (15, 15), f"ESKF.predict: P shape incorrect {P.shape}"
        assert z_acc.shape == (3,), f"ESKF.predict: zAcc shape incorrect {z_acc.shape}"
        assert z_gyro.shape == (
            3,
        ), f"ESKF.predict: zGyro shape incorrect {z_gyro.shape}"

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
        P_predicted = self.predict_covariance(x_nominal,P,acceleration,omega,Ts) # 这里协方差预测基于当前名义状态线性化

        assert x_nominal_predicted.shape == (
            16,
        ), f"ESKF.predict: x_nominal_predicted shape incorrect {x_nominal_predicted.shape}"
        assert P_predicted.shape == (
            15,
            15,
        ), f"ESKF.predict: P_predicted shape incorrect {P_predicted.shape}"

        return x_nominal_predicted, P_predicted

# 
    def inject(
        self, x_nominal: np.ndarray, delta_x: np.ndarray, P: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """将估计得到的误差状态注入名义状态，并对协方差做一致性补偿。

        Args:
            x_nominal (np.ndarray): 待注入误差的名义状态，形状为 (16,)
            delta_x (np.ndarray): 误差状态增量，形状为 (15,)
            P (np.ndarray): 误差状态协方差矩阵

        Raises:
            AssertionError: 输入形状错误；调试模式下还会检查部分数值性质

        Returns:
            Tuple[ np.ndarray, np.ndarray ]: 注入结果二元组 (x_injected, P_injected):
                x_injected: 注入后的名义状态，形状为 (16,)
                P_injected: 注入后的误差状态协方差矩阵，形状为 (15, 15)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.inject: x_nominal shape incorrect {x_nominal.shape}"
        assert delta_x.shape == (
            15,
        ), f"ESKF.inject: delta_x shape incorrect {delta_x.shape}"
        assert P.shape == (15, 15), f"ESKF.inject: P shape incorrect {P.shape}"

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
        

        assert x_injected.shape == (
            16,
        ), f"ESKF.inject: x_injected shape incorrect {x_injected.shape}"
        assert P_injected.shape == (
            15,
            15,
        ), f"ESKF.inject: P_injected shape incorrect {P_injected.shape}"

        return x_injected, P_injected

# GNSS 位置观测创新计算
    def innovation_GNSS_position(
        self,
        x_nominal: np.ndarray,
        P: np.ndarray,
        z_GNSS_position: np.ndarray,
        R_GNSS: np.ndarray,
        lever_arm: np.ndarray = np.zeros(3),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算 GNSS 位置观测的创新及其协方差。

        Args:
            x_nominal (np.ndarray): 用于计算创新的名义状态，形状为 (16,)
            P (np.ndarray): 用于计算创新协方差的误差状态协方差，形状为 (15, 15)
            z_GNSS_position (np.ndarray): 测得的三维位置，形状为 (3,)
            R_GNSS (np.ndarray): 观测噪声协方差矩阵，形状为 (3, 3)
            lever_arm (np.ndarray, optional): GNSS 天线相对 IMU 参考点的杆臂，默认 np.zeros(3)

        Raises:
            AssertionError: 输入形状错误；调试模式下还会检查部分数值性质

        Returns:
            Tuple[ np.ndarray, np.ndarray ]: 创新二元组 (v, S):
                v: 创新向量，形状为 (3,)
                S: 创新协方差，形状为 (3, 3)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.innovation_GNSS: x_nominal shape incorrect {x_nominal.shape}"
        assert P.shape == (15, 15), f"ESKF.innovation_GNSS: P shape incorrect {P.shape}"

        assert z_GNSS_position.shape == (
            3,
        ), f"ESKF.innovation_GNSS: z_GNSS_position shape incorrect {z_GNSS_position.shape}"
        assert R_GNSS.shape == (
            3,
            3,
        ), f"ESKF.innovation_GNSS: R_GNSS shape incorrect {R_GNSS.shape}"
        assert lever_arm.shape == (
            3,
        ), f"ESKF.innovation_GNSS: lever_arm shape incorrect {lever_arm.shape}"

        #H = np.zeros((1,))  # 待办：观测矩阵
        H = np.block([np.eye(3), np.zeros((3,12))])
        v = z_GNSS_position-x_nominal[POS_IDX]

        # 待办：创新

        # 杆臂补偿
        if not np.allclose(lever_arm, 0):
            R = quaternion_to_rotation_matrix(x_nominal[ATT_IDX], debug=self.debug)
            H[:, ERR_ATT_IDX] = -R @ cross_product_matrix(lever_arm, debug=self.debug)
            v -= R @ lever_arm

        S = H@P@H.T+R_GNSS # 待办：创新协方差

        assert v.shape == (3,), f"ESKF.innovation_GNSS: v shape incorrect {v.shape}"
        assert S.shape == (3, 3), f"ESKF.innovation_GNSS: S shape incorrect {S.shape}"
        return v, S

# GNSS 位置观测更新
    def update_GNSS_position(
        self,
        x_nominal: np.ndarray,
        P: np.ndarray,
        z_GNSS_position: np.ndarray,
        R_GNSS: np.ndarray,
        lever_arm: np.ndarray = np.zeros(3),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """利用 GNSS 位置观测更新状态与协方差。

        Args:
            x_nominal (np.ndarray): 待更新的名义状态，形状为 (16,)
            P (np.ndarray): 待更新的误差状态协方差，形状为 (15, 15)
            z_GNSS_position (np.ndarray): 测得的三维位置，形状为 (3,)
            R_GNSS (np.ndarray): 观测噪声协方差矩阵，形状为 (3, 3)
            lever_arm (np.ndarray, optional): GNSS 天线相对 IMU 参考点的杆臂，形状为 (3,)，默认 np.zeros(3)

        Raises:
            AssertionError: 输入形状错误；调试模式下还会检查部分数值性质

        Returns:
            Tuple[np.ndarray, np.ndarray]: 更新结果二元组 (x_injected, P_injected):
                x_injected: 注入更新后误差状态的名义状态，形状为 (16,)
                P_injected: 误差状态更新并注入后的协方差，形状为 (15, 15)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.update_GNSS: x_nominal shape incorrect {x_nominal.shape}"
        assert P.shape == (15, 15), f"ESKF.update_GNSS: P shape incorrect {P.shape}"
        assert z_GNSS_position.shape == (
            3,
        ), f"ESKF.update_GNSS: z_GNSS_position shape incorrect {z_GNSS_position.shape}"
        assert R_GNSS.shape == (
            3,
            3,
        ), f"ESKF.update_GNSS: R_GNSS shape incorrect {R_GNSS.shape}"
        assert lever_arm.shape == (
            3,
        ), f"ESKF.update_GNSS: lever_arm shape incorrect {lever_arm.shape}"

        I = np.eye(*P.shape)

        innovation, S = self.innovation_GNSS_position(
            x_nominal, P, z_GNSS_position, R_GNSS, lever_arm
        )

        H = np.block([np.eye(3), np.zeros((3,12))])


        # 若给定杆臂，则修正 H 中姿态误差相关项
        if not np.allclose(lever_arm, 0):
            R = quaternion_to_rotation_matrix(x_nominal[ATT_IDX], debug=self.debug)
            H[:, ERR_ATT_IDX] = -R @ cross_product_matrix(lever_arm, debug=self.debug)

        # 卡尔曼滤波误差状态更新
        W = P@H.T@np.linalg.inv(S) # 待办：卡尔曼增益
        delta_x = W@innovation # 待办：误差状态增量

        Jo = I - W @ H  # 约瑟夫形式

        P_update = Jo@P@Jo.T+W@R_GNSS@W.T # 待办：协方差更新

        # 误差状态注入
        x_injected, P_injected = self.inject(x_nominal, delta_x, P_update)

        assert x_injected.shape == (
            16,
        ), f"ESKF.update_GNSS: x_injected shape incorrect {x_injected.shape}"
        assert P_injected.shape == (
            15,
            15,
        ), f"ESKF.update_GNSS: P_injected shape incorrect {P_injected.shape}"

        return x_injected, P_injected

# GNSS 位置观测创新计算
    def NIS_GNSS_position(
        self,
        x_nominal: np.ndarray,
        P: np.ndarray,
        z_GNSS_position: np.ndarray,
        R_GNSS: np.ndarray,
        lever_arm: np.ndarray = np.zeros(3),
    ) -> float:
        """计算 GNSS 位置观测的 NIS。

        Args:
            x_nominal (np.ndarray): 用于计算创新的名义状态，形状为 (16,)
            P (np.ndarray): 用于计算创新协方差的误差状态协方差，形状为 (15, 15)
            z_GNSS_position (np.ndarray): 测得的三维位置，形状为 (3,)
            R_GNSS (np.ndarray): 观测噪声协方差矩阵，形状为 (3,)
            lever_arm (np.ndarray, optional): GNSS 天线相对 IMU 参考点的杆臂，形状为 (3,)，默认 np.zeros(3)

        Raises:
            AssertionError: 输入形状错误；调试模式下还会检查部分数值性质

        Returns:
            float: 归一化创新平方（NIS）
        """

        assert x_nominal.shape == (
            16,
        ), "ESKF.NIS_GNSS: x_nominal shape incorrect " + str(x_nominal.shape)
        assert P.shape == (15, 15), "ESKF.NIS_GNSS: P shape incorrect " + str(P.shape)
        assert z_GNSS_position.shape == (
            3,
        ), "ESKF.NIS_GNSS: z_GNSS_position shape incorrect " + str(
            z_GNSS_position.shape
        )
        assert R_GNSS.shape == (3, 3), "ESKF.NIS_GNSS: R_GNSS shape incorrect " + str(
            R_GNSS.shape
        )
        assert lever_arm.shape == (
            3,
        ), "ESKF.NIS_GNSS: lever_arm shape incorrect " + str(lever_arm.shape)

        v, S = self.innovation_GNSS_position(
            x_nominal, P, z_GNSS_position, R_GNSS, lever_arm
        )

        NIS = v.T@np.linalg.inv(S)@v  # 待办：计算归一化创新平方 # S 可能不可逆

        assert NIS >= 0, "EKSF.NIS_GNSS_positionNIS: NIS not positive"

        return NIS

    @classmethod
    def delta_x(cls, x_nominal: np.ndarray, x_true: np.ndarray,) -> np.ndarray:
        """计算 x_nominal 与 x_true 之间的误差状态。

        Args:
            x_nominal (np.ndarray): 名义估计状态，形状为 (16,)
            x_true (np.ndarray): 真实状态，形状为 (16,)

        Raises:
            AssertionError: 输入形状错误；调试模式下还会检查部分数值性质

        Returns:
            np.ndarray: 误差状态形式的状态差，形状为 (15,)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.delta_x: x_nominal shape incorrect {x_nominal.shape}"
        assert x_true.shape == (
            16,
        ), f"ESKF.delta_x: x_true shape incorrect {x_true.shape}"

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

        assert d_x.shape == (15,), f"ESKF.delta_x: d_x shape incorrect {d_x.shape}"

        return d_x

    @classmethod
    def NEESes(
        cls, x_nominal: np.ndarray, P: np.ndarray, x_true: np.ndarray,
    ) -> np.ndarray:
        """计算总体 NEES 及各子状态 NEES。

        Args:
            x_nominal (np.ndarray): 名义估计状态
            P (np.ndarray): 误差状态协方差
            x_true (np.ndarray): 真实状态

        Raises:
            AssertionError: 输入形状错误；调试模式下还会检查部分数值性质

        Returns:
            np.ndarray: [总体、位置、速度、姿态、加计偏置、陀螺偏置] 的 NEES，形状为 (6,)
        """

        assert x_nominal.shape == (
            16,
        ), f"ESKF.NEES: x_nominal shape incorrect {x_nominal.shape}"
        assert P.shape == (15, 15), f"ESKF.NEES: P shape incorrect {P.shape}"
        assert x_true.shape == (
            16,
        ), f"ESKF.NEES: x_true shape incorrect {x_true.shape}"

        d_x = cls.delta_x(x_nominal, x_true)
        """
        NEES_all = d_x.T@np.linalg.inv(P)@d_x  # 待办：总体一致性指标
        NEES_pos = d_x[POS_IDX].T@np.linalg.inv(P[POS_IDX**2])@d_x[POS_IDX] # 待办：位置一致性指标
        NEES_vel = d_x[VEL_IDX].T@np.linalg.inv(P[VEL_IDX**2])@d_x[VEL_IDX]  # 待办：速度一致性指标
        NEES_att = d_x[ATT_IDX].T@np.linalg.inv(P[ATT_IDX**2])@d_x[ATT_IDX]  # 待办：姿态一致性指标
        NEES_accbias = d_x[ACC_BIAS_IDX].T@np.linalg.inv(P[ACC_BIAS_IDX**2])@d_x[ACC_BIAS_IDX]  # 待办：加计偏置一致性指标
        NEES_gyrobias = d_x[GYRO_BIAS_IDX].T@(P[GYRO_BIAS_IDX**2])@d_x[GYRO_BIAS_IDX]  # 待办：陀螺偏置一致性指标
        NEES_gyrobias=cls._NEES(P[ERR_GYRO_BIAS_IDX**2], d_x[ERR_GYRO_BIAS_IDX])
        """
        NEES_all = cls._NEES(d_x,P)
        NEES_pos = cls._NEES(d_x[POS_IDX],P[POS_IDX**2])
        NEES_vel = cls._NEES(d_x[VEL_IDX],P[VEL_IDX**2])
        NEES_att = cls._NEES(d_x[ERR_ATT_IDX],P[ERR_ATT_IDX**2])
        NEES_accbias = cls._NEES(d_x[ERR_ACC_BIAS_IDX],P[ERR_ACC_BIAS_IDX**2])
        NEES_gyrobias = cls._NEES(d_x[ERR_GYRO_BIAS_IDX],P[ERR_GYRO_BIAS_IDX**2])


        NEESes = np.array(
            [NEES_all, NEES_pos, NEES_vel, NEES_att, NEES_accbias, NEES_gyrobias]
        )
        assert np.all(NEESes >= 0), "ESKF.NEES: one or more negative NEESes"
        #print(NEESes)
        #print(NEESes的数据类型)
        return NEESes
        
    @classmethod
    def _NEES(cls, diff, P):
        #print(P)
        NEES = diff@la.solve(P,diff)
        assert NEES >= 0, "ESKF._NEES: negative NEES"
        return NEES


# %%

