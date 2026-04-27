# 导入基础依赖
import scipy
import scipy.io
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
from pathlib import Path

# 读取配置参数
with open(r'params.yaml', encoding='utf-8') as file:
    params = yaml.load(file, Loader=yaml.FullLoader)

# 导入进度条
from tqdm import tqdm

# ESKF类及相关导入
from eskf import (
    ESKF,
    POS_IDX,
    VEL_IDX,
    ATT_IDX,
    ACC_BIAS_IDX,
    GYRO_BIAS_IDX,
    ERR_ATT_IDX,
    ERR_ACC_BIAS_IDX,
    ERR_GYRO_BIAS_IDX,
)

# 四元数与切片工具导入
from quaternion import quaternion_to_euler
from cat_slice import CatSlice

# 设置绘图样式
import scienceplots
plt_styles = ["science", "grid", "bright", "no-latex"]
plt.style.use(plt_styles)
print(f"pyplot using style set {plt_styles}")

# 控制变量区
doGNSS: bool = True # 是否执行 GNSS 更新
do_auto_R: bool = True # 是否启用自适应测量噪声（R）调整
do_sage_husa: bool = False # 是否使用 SAGE-HUSA 进行自适应（R）噪声调整
do_auto_Q: bool = True # 是否启用自适应过程噪声（Q）调整
# filename_to_load = "./task_simulation_9000/task_simulation_part_01.mat" # 要加载的仿真数据文件名
# filename_to_load = "./task_simulation_50000/task_simulation_random_02.mat" # 要加载的仿真数据文件名
# filename_to_load = "task_simulation.mat" # 要加载的仿真数据文件名
filename_to_load = "./task_simulation_90000/task_simulation_scene5.mat"
gnss_downsample_factor: int = 1 # GNSS 数据下采样因子
do_emergency: bool = False # 突发状况 GNSS一段时间缺失
gnss_dropout_start = 5.0   # 失锁开始时间 [s]
gnss_dropout_end = 25.0     # 失锁结束时间 [s]
steps=90000
window = 10 # 误差调整窗口大小（仅在 do_auto_R=True 时使用）

# 批处理模式可通过环境变量覆盖输入和图像输出
filename_to_load = os.environ.get("SIM_DATA_FILE", filename_to_load)
save_fig_dir = os.environ.get("SAVE_FIG_DIR", "").strip()
fig_prefix = os.environ.get("FIG_PREFIX", "").strip()
no_show_fig = os.environ.get("NO_SHOW_FIG", "0") == "1"
save_fullscreen_fig = os.environ.get("SAVE_FULLSCREEN_FIG", "1") == "1"


def apply_fullscreen_canvas(fig, width_in=19.2, height_in=10.8):
    # 使用 16:9 大画布导出，近似全屏截图尺寸
    fig.set_size_inches(width_in, height_in, forward=True)

# 加载数据
loaded_data = scipy.io.loadmat(filename_to_load)
S_a = loaded_data["S_a"]
S_g = loaded_data["S_g"]
timeGNSS = loaded_data["timeGNSS"].ravel() # 100Hz
timeIMU = loaded_data["timeIMU"].ravel() # 1Hz
x_true = loaded_data["xtrue"].T
z_acceleration = loaded_data["zAcc"].T
z_GNSS = loaded_data["zGNSS"].T
z_gyroscope = loaded_data["zGyro"].T
dt = np.mean(np.diff(timeIMU))

timeGNSS = timeGNSS[::gnss_downsample_factor]
z_GNSS = z_GNSS[::gnss_downsample_factor]
gnss_steps = len(z_GNSS)

# 测量噪声 STIM300 的 IMU 噪声参数，依据数据手册与仿真采样率设置 （连续时间噪声）
# TODO: 可继续调参
cont_gyro_noise_std = float(params["cont_gyro_noise_std"]) #7e-5 #4.36e-5  # 单位: (rad/s)/sqrt(Hz)
cont_acc_noise_std = float(params["cont_acc_noise_std"]) #4e-3#1.167e-3  # 单位: (m/s**2)/sqrt(Hz)
# 仿真采样率下使用的离散噪声
rate_std = 0.5 * cont_gyro_noise_std * np.sqrt(1 / dt)
acc_std = 0.5 * cont_acc_noise_std * np.sqrt(1 / dt)
# 偏置相关参数
rate_bias_driving_noise_std = float(params["rate_bias_driving_noise_std"]) #10e-2
# 调试用：打印放大后的陀螺偏置驱动噪声
#print(float(rate_bias_driving_noise_std)*2)
cont_rate_bias_driving_noise_std = (
    (1 / 3) * rate_bias_driving_noise_std / np.sqrt(1 / dt)
)
acc_bias_driving_noise_std = float(params["acc_bias_driving_noise_std"]) #3 # 该参数已做过较大幅度调参
cont_acc_bias_driving_noise_std = 6 * acc_bias_driving_noise_std / np.sqrt(1 / dt)
# 位置测量噪声
p_std = np.array(params["p_std"]) #np.array([0.3, 0.3, 0.5]) # np.array([0.3, 0.3, 0.5]) # 测量噪声
R_GNSS = np.diag(p_std ** 2)
p_acc = float(params["p_acc"]) #1e-17 #1e-16
p_gyro = float(params["p_gyro"]) #1e-17 #1e-16

# 估计器初始化
eskf = ESKF(
    acc_std,
    rate_std,
    cont_acc_bias_driving_noise_std,
    cont_rate_bias_driving_noise_std,
    p_acc,
    p_gyro,
    S_a=S_a, # 设置加速度计修正矩阵
    S_g=S_g, # 设置陀螺仪修正矩阵
    )


# 预分配数组
x_est = np.zeros((steps, 16))
P_est = np.zeros((steps, 15, 15))
x_pred = np.zeros((steps, 16))
P_pred = np.zeros((steps, 15, 15))
v_prior = np.zeros((steps, 3)) # 记录每次 GNSS 更新前的测量残差（先验残差）
v_post = np.zeros((steps, 3)) # 记录每次 GNSS 更新后的测量残差（后验残差）
delta_x = np.zeros((steps, 15))
R_GNSS_history = np.zeros((steps, 3, 3)) # 可选：记录每次更新后的 R_GNSS 以分析自适应调整效果

# 初始化
# 状态变量初始化
x_pred[0, POS_IDX] = x_true[0, POS_IDX]  # 使用数据中的首个位置作为初值
x_pred[0, VEL_IDX] = x_true[0, VEL_IDX]  # 使用数据中的首个速度作为初值
x_pred[0, ATT_IDX] = x_true[0, ATT_IDX]  # 使用数据中的首个四元数姿态作为初值       
# P矩阵初始化
P_pred[0][POS_IDX ** 2] = params["P_pred0_pos"]*np.eye(3)# TODO: 继续调参
P_pred[0][VEL_IDX ** 2] = params["P_pred0_vel"]*np.eye(3)# TODO: 继续调参
P_pred[0][ERR_ATT_IDX ** 2] = params["P_pred0_att"]*np.eye(3)# TODO: 继续调参 # 误差旋转向量（非四元数）
P_pred[0][ERR_ACC_BIAS_IDX ** 2] = params["P_pred0_accbias"]*np.eye(3)# TODO: 继续调参
P_pred[0][ERR_GYRO_BIAS_IDX ** 2] = params["P_pred0_gyrobias"]*np.eye(3)# TODO: 继续调参

# 可选：固定估计步数
N: int = steps # TODO: 可先从较小值开始（如 500），结果稳定后再逐步增大

do_auto_R: bool = False # 是否启用自适应测量噪声（R）调整
do_sage_husa: bool = False # 是否使用 SAGE-HUSA 进行自适应（R）噪声调整
do_auto_Q: bool = False # 是否启用自适应过程噪声（Q）调整

# 主循环 滤波过程
GNSSk: int = 0  # 记录当前 GNSS 测量索引
for k in tqdm(range(N)):
    gnss_available = not (
        do_emergency
        and gnss_dropout_start + timeIMU[0] <= timeIMU[k] <= gnss_dropout_end + timeIMU[0]
    )
    if doGNSS and gnss_available and GNSSk < gnss_steps and timeIMU[k] >= timeGNSS[GNSSk]:
        v_prior[GNSSk] = z_GNSS[GNSSk] - x_pred[k, POS_IDX] # 测量残差（先验残差）        
        if GNSSk == 0:
            x_est[k], P_est[k], R_GNSS, W = eskf.update_GNSS_position(x_pred[k],P_pred[k],R_GNSS,GNSSk,v_prior,np.zeros(3), do_auto_R, do_sage_husa, window)
        else:
            x_est[k], P_est[k], R_GNSS, W = eskf.update_GNSS_position(x_pred[k],P_pred[k],R_GNSS,GNSSk,v_prior,v_post[GNSSk-1], do_auto_R, do_sage_husa, window)
            v_post[GNSSk] = z_GNSS[GNSSk] - x_est[k, POS_IDX] # 测量残差（后验残差）
        R_GNSS_history[GNSSk] = R_GNSS

        GNSSk += 1
    else:
        # 无观测更新时，令估计值等于预测值
        x_est[k] = x_pred[k]
        P_est[k] = P_pred[k]

    delta_x[k] = eskf.delta_x(x_est[k], x_true[k])

    if k < N - 1:
        if GNSSk == 0:
            x_pred[k + 1], P_pred[k + 1] = eskf.predict(x_est[k],P_est[k],z_acceleration[k],z_gyroscope[k],dt, np.zeros((15, 3)), GNSSk, np.zeros(3),np.zeros(3) ,do_auto_Q)# TODO: 提示：测量来自当前与过去时刻，而不是未来
        else:
            x_pred[k + 1], P_pred[k + 1] = eskf.predict(x_est[k],P_est[k],z_acceleration[k],z_gyroscope[k],dt, W, GNSSk, v_prior,v_post ,do_auto_Q)# TODO: 提示：测量来自当前与过去时刻，而不是未来
delta_x_1 = delta_x

# 测量噪声 STIM300 的 IMU 噪声参数，依据数据手册与仿真采样率设置 （连续时间噪声）
# TODO: 可继续调参
cont_gyro_noise_std = float(params["cont_gyro_noise_std"]) #7e-5 #4.36e-5  # 单位: (rad/s)/sqrt(Hz)
cont_acc_noise_std = float(params["cont_acc_noise_std"]) #4e-3#1.167e-3  # 单位: (m/s**2)/sqrt(Hz)
# 仿真采样率下使用的离散噪声
rate_std = 0.5 * cont_gyro_noise_std * np.sqrt(1 / dt)
acc_std = 0.5 * cont_acc_noise_std * np.sqrt(1 / dt)
# 偏置相关参数
rate_bias_driving_noise_std = float(params["rate_bias_driving_noise_std"]) #10e-2
# 调试用：打印放大后的陀螺偏置驱动噪声
#print(float(rate_bias_driving_noise_std)*2)
cont_rate_bias_driving_noise_std = (
    (1 / 3) * rate_bias_driving_noise_std / np.sqrt(1 / dt)
)
acc_bias_driving_noise_std = float(params["acc_bias_driving_noise_std"]) #3 # 该参数已做过较大幅度调参
cont_acc_bias_driving_noise_std = 6 * acc_bias_driving_noise_std / np.sqrt(1 / dt)
# 位置测量噪声
p_std = np.array(params["p_std"]) #np.array([0.3, 0.3, 0.5]) # np.array([0.3, 0.3, 0.5]) # 测量噪声
R_GNSS = np.diag(p_std ** 2)
p_acc = float(params["p_acc"]) #1e-17 #1e-16
p_gyro = float(params["p_gyro"]) #1e-17 #1e-16

# 估计器初始化
eskf = ESKF(
    acc_std,
    rate_std,
    cont_acc_bias_driving_noise_std,
    cont_rate_bias_driving_noise_std,
    p_acc,
    p_gyro,
    S_a=S_a, # 设置加速度计修正矩阵
    S_g=S_g, # 设置陀螺仪修正矩阵
    )


# 预分配数组
x_est = np.zeros((steps, 16))
P_est = np.zeros((steps, 15, 15))
x_pred = np.zeros((steps, 16))
P_pred = np.zeros((steps, 15, 15))
v_prior = np.zeros((steps, 3)) # 记录每次 GNSS 更新前的测量残差（先验残差）
v_post = np.zeros((steps, 3)) # 记录每次 GNSS 更新后的测量残差（后验残差）
delta_x = np.zeros((steps, 15))
R_GNSS_history = np.zeros((steps, 3, 3)) # 可选：记录每次更新后的 R_GNSS 以分析自适应调整效果

# 初始化
# 状态变量初始化
x_pred[0, POS_IDX] = x_true[0, POS_IDX]  # 使用数据中的首个位置作为初值
x_pred[0, VEL_IDX] = x_true[0, VEL_IDX]  # 使用数据中的首个速度作为初值
x_pred[0, ATT_IDX] = x_true[0, ATT_IDX]  # 使用数据中的首个四元数姿态作为初值       
# P矩阵初始化
P_pred[0][POS_IDX ** 2] = params["P_pred0_pos"]*np.eye(3)# TODO: 继续调参
P_pred[0][VEL_IDX ** 2] = params["P_pred0_vel"]*np.eye(3)# TODO: 继续调参
P_pred[0][ERR_ATT_IDX ** 2] = params["P_pred0_att"]*np.eye(3)# TODO: 继续调参 # 误差旋转向量（非四元数）
P_pred[0][ERR_ACC_BIAS_IDX ** 2] = params["P_pred0_accbias"]*np.eye(3)# TODO: 继续调参
P_pred[0][ERR_GYRO_BIAS_IDX ** 2] = params["P_pred0_gyrobias"]*np.eye(3)# TODO: 继续调参

do_auto_R: bool = False # 是否启用自适应测量噪声（R）调整
do_sage_husa: bool = True # 是否使用 SAGE-HUSA 进行自适应（R）噪声调整
do_auto_Q: bool = True # 是否启用自适应过程噪声（Q）调整

# 主循环 滤波过程
GNSSk: int = 0  # 记录当前 GNSS 测量索引
for k in tqdm(range(N)):
    gnss_available = not (
        do_emergency
        and gnss_dropout_start + timeIMU[0] <= timeIMU[k] <= gnss_dropout_end + timeIMU[0]
    )
    if doGNSS and gnss_available and GNSSk < gnss_steps and timeIMU[k] >= timeGNSS[GNSSk]:
        v_prior[GNSSk] = z_GNSS[GNSSk] - x_pred[k, POS_IDX] # 测量残差（先验残差）        
        if GNSSk == 0:
            x_est[k], P_est[k], R_GNSS, W = eskf.update_GNSS_position(x_pred[k],P_pred[k],R_GNSS,GNSSk,v_prior,np.zeros(3), do_auto_R, do_sage_husa, window)
        else:
            x_est[k], P_est[k], R_GNSS, W = eskf.update_GNSS_position(x_pred[k],P_pred[k],R_GNSS,GNSSk,v_prior,v_post[GNSSk-1], do_auto_R, do_sage_husa, window)
            v_post[GNSSk] = z_GNSS[GNSSk] - x_est[k, POS_IDX] # 测量残差（后验残差）
        R_GNSS_history[GNSSk] = R_GNSS

        GNSSk += 1
    else:
        # 无观测更新时，令估计值等于预测值
        x_est[k] = x_pred[k]
        P_est[k] = P_pred[k]

    delta_x[k] = eskf.delta_x(x_est[k], x_true[k])

    if k < N - 1:
        if GNSSk == 0:
            x_pred[k + 1], P_pred[k + 1] = eskf.predict(x_est[k],P_est[k],z_acceleration[k],z_gyroscope[k],dt, np.zeros((15, 3)), GNSSk, np.zeros(3),np.zeros(3) ,do_auto_Q)# TODO: 提示：测量来自当前与过去时刻，而不是未来
        else:
            x_pred[k + 1], P_pred[k + 1] = eskf.predict(x_est[k],P_est[k],z_acceleration[k],z_gyroscope[k],dt, W, GNSSk, v_prior,v_post ,do_auto_Q)# TODO: 提示：测量来自当前与过去时刻，而不是未来
delta_x_2 = delta_x

# 测量噪声 STIM300 的 IMU 噪声参数，依据数据手册与仿真采样率设置 （连续时间噪声）
# TODO: 可继续调参
cont_gyro_noise_std = float(params["cont_gyro_noise_std"]) #7e-5 #4.36e-5  # 单位: (rad/s)/sqrt(Hz)
cont_acc_noise_std = float(params["cont_acc_noise_std"]) #4e-3#1.167e-3  # 单位: (m/s**2)/sqrt(Hz)
# 仿真采样率下使用的离散噪声
rate_std = 0.5 * cont_gyro_noise_std * np.sqrt(1 / dt)
acc_std = 0.5 * cont_acc_noise_std * np.sqrt(1 / dt)
# 偏置相关参数
rate_bias_driving_noise_std = float(params["rate_bias_driving_noise_std"]) #10e-2
# 调试用：打印放大后的陀螺偏置驱动噪声
#print(float(rate_bias_driving_noise_std)*2)
cont_rate_bias_driving_noise_std = (
    (1 / 3) * rate_bias_driving_noise_std / np.sqrt(1 / dt)
)
acc_bias_driving_noise_std = float(params["acc_bias_driving_noise_std"]) #3 # 该参数已做过较大幅度调参
cont_acc_bias_driving_noise_std = 6 * acc_bias_driving_noise_std / np.sqrt(1 / dt)
# 位置测量噪声
p_std = np.array(params["p_std"]) #np.array([0.3, 0.3, 0.5]) # np.array([0.3, 0.3, 0.5]) # 测量噪声
R_GNSS = np.diag(p_std ** 2)
p_acc = float(params["p_acc"]) #1e-17 #1e-16
p_gyro = float(params["p_gyro"]) #1e-17 #1e-16

# 估计器初始化
eskf = ESKF(
    acc_std,
    rate_std,
    cont_acc_bias_driving_noise_std,
    cont_rate_bias_driving_noise_std,
    p_acc,
    p_gyro,
    S_a=S_a, # 设置加速度计修正矩阵
    S_g=S_g, # 设置陀螺仪修正矩阵
    )


# 预分配数组
x_est = np.zeros((steps, 16))
P_est = np.zeros((steps, 15, 15))
x_pred = np.zeros((steps, 16))
P_pred = np.zeros((steps, 15, 15))
v_prior = np.zeros((steps, 3)) # 记录每次 GNSS 更新前的测量残差（先验残差）
v_post = np.zeros((steps, 3)) # 记录每次 GNSS 更新后的测量残差（后验残差）
delta_x = np.zeros((steps, 15))
R_GNSS_history = np.zeros((steps, 3, 3)) # 可选：记录每次更新后的 R_GNSS 以分析自适应调整效果

# 初始化
# 状态变量初始化
x_pred[0, POS_IDX] = x_true[0, POS_IDX]  # 使用数据中的首个位置作为初值
x_pred[0, VEL_IDX] = x_true[0, VEL_IDX]  # 使用数据中的首个速度作为初值
x_pred[0, ATT_IDX] = x_true[0, ATT_IDX]  # 使用数据中的首个四元数姿态作为初值       
# P矩阵初始化
P_pred[0][POS_IDX ** 2] = params["P_pred0_pos"]*np.eye(3)# TODO: 继续调参
P_pred[0][VEL_IDX ** 2] = params["P_pred0_vel"]*np.eye(3)# TODO: 继续调参
P_pred[0][ERR_ATT_IDX ** 2] = params["P_pred0_att"]*np.eye(3)# TODO: 继续调参 # 误差旋转向量（非四元数）
P_pred[0][ERR_ACC_BIAS_IDX ** 2] = params["P_pred0_accbias"]*np.eye(3)# TODO: 继续调参
P_pred[0][ERR_GYRO_BIAS_IDX ** 2] = params["P_pred0_gyrobias"]*np.eye(3)# TODO: 继续调参

do_auto_R: bool = True # 是否启用自适应测量噪声（R）调整
do_sage_husa: bool = False # 是否使用 SAGE-HUSA 进行自适应（R）噪声调整
do_auto_Q: bool = True # 是否启用自适应过程噪声（Q）调整

# 主循环 滤波过程
GNSSk: int = 0  # 记录当前 GNSS 测量索引
for k in tqdm(range(N)):
    gnss_available = not (
        do_emergency
        and gnss_dropout_start + timeIMU[0] <= timeIMU[k] <= gnss_dropout_end + timeIMU[0]
    )
    if doGNSS and gnss_available and GNSSk < gnss_steps and timeIMU[k] >= timeGNSS[GNSSk]:
        v_prior[GNSSk] = z_GNSS[GNSSk] - x_pred[k, POS_IDX] # 测量残差（先验残差）        
        if GNSSk == 0:
            x_est[k], P_est[k], R_GNSS, W = eskf.update_GNSS_position(x_pred[k],P_pred[k],R_GNSS,GNSSk,v_prior,np.zeros(3), do_auto_R, do_sage_husa, window)
        else:
            x_est[k], P_est[k], R_GNSS, W = eskf.update_GNSS_position(x_pred[k],P_pred[k],R_GNSS,GNSSk,v_prior,v_post[GNSSk-1], do_auto_R, do_sage_husa, window)
            v_post[GNSSk] = z_GNSS[GNSSk] - x_est[k, POS_IDX] # 测量残差（后验残差）
        R_GNSS_history[GNSSk] = R_GNSS

        GNSSk += 1
    else:
        # 无观测更新时，令估计值等于预测值
        x_est[k] = x_pred[k]
        P_est[k] = P_pred[k]

    delta_x[k] = eskf.delta_x(x_est[k], x_true[k])

    if k < N - 1:
        if GNSSk == 0:
            x_pred[k + 1], P_pred[k + 1] = eskf.predict(x_est[k],P_est[k],z_acceleration[k],z_gyroscope[k],dt, np.zeros((15, 3)), GNSSk, np.zeros(3),np.zeros(3) ,do_auto_Q)# TODO: 提示：测量来自当前与过去时刻，而不是未来
        else:
            x_pred[k + 1], P_pred[k + 1] = eskf.predict(x_est[k],P_est[k],z_acceleration[k],z_gyroscope[k],dt, W, GNSSk, v_prior,v_post ,do_auto_Q)# TODO: 提示：测量来自当前与过去时刻，而不是未来
delta_x_3 = delta_x

# 绘图
# 状态估计结果与真值的比较，以及误差分析

# # 轨迹比较
# fig1 = plt.figure(1)
# fig1.suptitle("Trajectory comparison")
# ax = plt.axes(projection="3d")

# ax.plot3D(x_est[:N, 1], x_est[:N, 0], -x_est[:N, 2], color='blue', label='ESKF estimate') # 前N行，第1列（东）、第0列（北）、第2列（下，取负号变为高度）
# ax.plot3D(z_GNSS[:GNSSk, 1], z_GNSS[:GNSSk, 0], -z_GNSS[:GNSSk, 2], color='red', label='GNSS')
# ax.plot3D(x_true[:N, 1], x_true[:N, 0], -x_true[:N, 2], color='yellow', label='Ground truth')
# ax.set_xlabel("East [m]")
# ax.set_ylabel("North [m]")
# ax.set_zlabel("Altitude [m]")
# ax.legend(loc='best')

# t = np.linspace(0, dt * (N - 1), N)
# eul = np.apply_along_axis(quaternion_to_euler, 1, x_est[:N, ATT_IDX])
# eul_true = np.apply_along_axis(quaternion_to_euler, 1, x_true[:N, ATT_IDX])

# # 状态估计曲线
# fig2, axs2 = plt.subplots(3, 1, num=2, clear=True)
# fig2.suptitle("States estimates")

# axs2[0].plot(t, x_est[:N, POS_IDX], linewidth=1.5)
# axs2[0].plot(t, x_true[:N, POS_IDX], linestyle='--', linewidth=1.2)
# axs2[0].set(ylabel="NED position [m]",xlabel="Time [s]")
# axs2[0].legend([
#     "North est", "East est", "Down est",
#     "North true", "East true", "Down true",
# ])


# axs2[1].plot(t, x_est[:N, VEL_IDX], linewidth=1.5)
# axs2[1].plot(t, x_true[:N, VEL_IDX], linestyle='--', linewidth=1.2)
# axs2[1].set(ylabel="Velocities [m/s]",xlabel="Time [s]")
# axs2[1].legend([
#     "North est", "East est", "Down est",
#     "North true", "East true", "Down true",
# ])


# axs2[2].plot(t, eul[:N] * 180 / np.pi, linewidth=1.5)
# axs2[2].plot(t, eul_true[:N] * 180 / np.pi, linestyle='--', linewidth=1.2)
# axs2[2].set(ylabel="Euler angles [deg]",xlabel="Time [s]")
# axs2[2].legend([
#     r"$\phi$ est", r"$\theta$ est", r"$\psi$ est",
#     r"$\phi$ true", r"$\theta$ true", r"$\psi$ true",
# ])

# 误差范数曲线(RMSE)
fig3, axs3 = plt.subplots(5, 1, num=3, clear=True)
fig3.suptitle("RMSE of all state groups")
t = np.linspace(0, dt * (N - 1), N)

# 各状态分量 RMSE（便于论文表格或终端汇总）
pos_err_norm_1 = np.linalg.norm(delta_x_1[:N, POS_IDX], axis=1)
vel_err_norm_1 = np.linalg.norm(delta_x_1[:N, VEL_IDX], axis=1)
att_err_norm_deg_1 = np.linalg.norm(delta_x_1[:N, ERR_ATT_IDX] * 180 / np.pi, axis=1)
acc_bias_err_norm_1 = np.linalg.norm(delta_x_1[:N, ERR_ACC_BIAS_IDX], axis=1)
gyro_bias_err_norm_deg_h_1 = np.linalg.norm(
    delta_x_1[:N, ERR_GYRO_BIAS_IDX] * 180 / np.pi * 3600, axis=1
)

# 各状态分量 RMSE（便于论文表格或终端汇总）
pos_err_norm_2 = np.linalg.norm(delta_x_2[:N, POS_IDX], axis=1)
vel_err_norm_2 = np.linalg.norm(delta_x_2[:N, VEL_IDX], axis=1)
att_err_norm_deg_2 = np.linalg.norm(delta_x_2[:N, ERR_ATT_IDX] * 180 / np.pi, axis=1)
acc_bias_err_norm_2 = np.linalg.norm(delta_x_2[:N, ERR_ACC_BIAS_IDX], axis=1)
gyro_bias_err_norm_deg_h_2 = np.linalg.norm(
    delta_x_2[:N, ERR_GYRO_BIAS_IDX] * 180 / np.pi * 3600, axis=1
)

# 各状态分量 RMSE（便于论文表格或终端汇总）
pos_err_norm_3 = np.linalg.norm(delta_x_3[:N, POS_IDX], axis=1)
vel_err_norm_3 = np.linalg.norm(delta_x_3[:N, VEL_IDX], axis=1)
att_err_norm_deg_3 = np.linalg.norm(delta_x_3[:N, ERR_ATT_IDX] * 180 / np.pi, axis=1)
acc_bias_err_norm_3 = np.linalg.norm(delta_x_3[:N, ERR_ACC_BIAS_IDX], axis=1)
gyro_bias_err_norm_deg_h_3 = np.linalg.norm(
    delta_x_3[:N, ERR_GYRO_BIAS_IDX] * 180 / np.pi * 3600, axis=1
)

# rmse_pos_xyz = np.sqrt(np.mean(delta_x[:N, POS_IDX] ** 2, axis=0))
# rmse_pos_xyz_gnss = np.sqrt(np.mean((x_true[99:N:100 * gnss_downsample_factor, POS_IDX] - z_GNSS[:steps//(100 * gnss_downsample_factor)])**2, axis=0))

# rmse_pos_xyz_1 = np.sqrt(np.mean(delta_x[:20000, POS_IDX] ** 2, axis=0))
# rmse_pos_xyz_2 = np.sqrt(np.mean(delta_x[20000:40000, POS_IDX] ** 2, axis=0))
# rmse_pos_xyz_3 = np.sqrt(np.mean(delta_x[40000:60000, POS_IDX] ** 2, axis=0))
# rmse_pos_xyz_4 = np.sqrt(np.mean(delta_x[60000:90000, POS_IDX] ** 2, axis=0))

# rmse_pos_xyz_1_all = np.sqrt(np.mean(np.sum(delta_x[:20000, POS_IDX] ** 2, axis=1)))
# rmse_pos_xyz_2_all = np.sqrt(np.mean(np.sum(delta_x[20000:40000, POS_IDX] ** 2, axis=1)))
# rmse_pos_xyz_3_all = np.sqrt(np.mean(np.sum(delta_x[40000:60000, POS_IDX] ** 2, axis=1)))
# rmse_pos_xyz_4_all = np.sqrt(np.mean(np.sum(delta_x[60000:90000, POS_IDX] ** 2, axis=1)))

# rmse_pos_xyz_gnss_1 = np.sqrt(np.mean((x_true[99:20000:100 * gnss_downsample_factor, POS_IDX] - z_GNSS[0:200])**2, axis=0))
# rmse_pos_xyz_gnss_2 = np.sqrt(np.mean((x_true[20099:40000:100 * gnss_downsample_factor, POS_IDX] - z_GNSS[200:400])**2, axis=0))
# rmse_pos_xyz_gnss_3 = np.sqrt(np.mean((x_true[40099:60000:100 * gnss_downsample_factor, POS_IDX] - z_GNSS[400:600])**2, axis=0))
# rmse_pos_xyz_gnss_4 = np.sqrt(np.mean((x_true[60099:N:100 * gnss_downsample_factor, POS_IDX] - z_GNSS[600:steps//(100 * gnss_downsample_factor)])**2, axis=0))

# rmse_pos_xyz_gnss_1_all = np.sqrt(np.mean(np.sum((x_true[99:20000:100 * gnss_downsample_factor, POS_IDX] - z_GNSS[0:200])**2, axis=1)))
# rmse_pos_xyz_gnss_2_all = np.sqrt(np.mean(np.sum((x_true[20099:40000:100 * gnss_downsample_factor, POS_IDX] - z_GNSS[200:400])**2, axis=1)))
# rmse_pos_xyz_gnss_3_all = np.sqrt(np.mean(np.sum((x_true[40099:60000:100 * gnss_downsample_factor, POS_IDX] - z_GNSS[400:600])**2, axis=1)))
# rmse_pos_xyz_gnss_4_all = np.sqrt(np.mean(np.sum((x_true[60099:N:100 * gnss_downsample_factor, POS_IDX] - z_GNSS[600:steps//(100 * gnss_downsample_factor)])**2, axis=1)))


# rmse_vel_xyz = np.sqrt(np.mean(delta_x[:N, VEL_IDX] ** 2, axis=0))
# rmse_att_rpy_deg = np.sqrt(np.mean((delta_x[:N, ERR_ATT_IDX] * 180 / np.pi) ** 2, axis=0))
# rmse_acc_bias_xyz = np.sqrt(np.mean(delta_x[:N, ERR_ACC_BIAS_IDX] ** 2, axis=0))
# rmse_gyro_bias_xyz_deg_h = np.sqrt(
#     np.mean((delta_x[:N, ERR_GYRO_BIAS_IDX] * 180 / np.pi * 3600) ** 2, axis=0)
# )

# print("\n==================== RMSE summary ====================")
# if(do_auto_R):
#     print("window length = ", window)
# print(f"Position RMSE [m]      (N,E,D): {rmse_pos_xyz[0]:.4f}, {rmse_pos_xyz[1]:.4f}, {rmse_pos_xyz[2]:.4f}")
# print(f"Position RMSE(Overall) [m]      (N,E,D): {np.sqrt(np.mean(np.sum(delta_x[:N, POS_IDX]**2, axis=1))):.4f}")
# print(f"Position RMSE(GNSS) [m]      (N,E,D): {rmse_pos_xyz_gnss[0]:.4f}, {rmse_pos_xyz_gnss[1]:.4f}, {rmse_pos_xyz_gnss[2]:.4f}")
# print(f"Position RMSE(GNSS)_all [m]      (N,E,D): {np.sqrt(np.mean(np.sum((x_true[99:N:100 * gnss_downsample_factor, POS_IDX] - z_GNSS[:steps//(100 * gnss_downsample_factor)])**2, axis=1))):.4f}")
# print(f"Velocity RMSE [m/s]    (N,E,D): {rmse_vel_xyz[0]:.4f}, {rmse_vel_xyz[1]:.4f}, {rmse_vel_xyz[2]:.4f}")
# print(f"Attitude RMSE [deg]  (roll,pitch,yaw): {rmse_att_rpy_deg[0]:.4f}, {rmse_att_rpy_deg[1]:.4f}, {rmse_att_rpy_deg[2]:.4f}")
# print(f"Acc bias RMSE [m/s^2]  (x,y,z): {rmse_acc_bias_xyz[0]:.6f}, {rmse_acc_bias_xyz[1]:.6f}, {rmse_acc_bias_xyz[2]:.6f}")
# print(f"Gyro bias RMSE [deg/h] (x,y,z): {rmse_gyro_bias_xyz_deg_h[0]:.4f}, {rmse_gyro_bias_xyz_deg_h[1]:.4f}, {rmse_gyro_bias_xyz_deg_h[2]:.4f}")

# print(f"Position RMSE [m]_1      (N,E,D): {rmse_pos_xyz_1[0]:.4f}, {rmse_pos_xyz_1[1]:.4f}, {rmse_pos_xyz_1[2]:.4f}")
# print(f"Position RMSE [m]_2      (N,E,D): {rmse_pos_xyz_2[0]:.4f}, {rmse_pos_xyz_2[1]:.4f}, {rmse_pos_xyz_2[2]:.4f}")
# print(f"Position RMSE [m]_3      (N,E,D): {rmse_pos_xyz_3[0]:.4f}, {rmse_pos_xyz_3[1]:.4f}, {rmse_pos_xyz_3[2]:.4f}")
# print(f"Position RMSE [m]_4      (N,E,D): {rmse_pos_xyz_4[0]:.4f}, {rmse_pos_xyz_4[1]:.4f}, {rmse_pos_xyz_4[2]:.4f}")

# print(f"Position RMSE [m]_1_all      (Overall): {rmse_pos_xyz_1_all:.4f}")
# print(f"Position RMSE [m]_2_all      (Overall): {rmse_pos_xyz_2_all:.4f}")
# print(f"Position RMSE [m]_3_all      (Overall): {rmse_pos_xyz_3_all:.4f}")
# print(f"Position RMSE [m]_4_all      (Overall): {rmse_pos_xyz_4_all:.4f}")

# print(f"Position RMSE(GNSS)_1 [m]      (N,E,D): {rmse_pos_xyz_gnss_1[0]:.4f}, {rmse_pos_xyz_gnss_1[1]:.4f}, {rmse_pos_xyz_gnss_1[2]:.4f}")
# print(f"Position RMSE(GNSS)_2 [m]      (N,E,D): {rmse_pos_xyz_gnss_2[0]:.4f}, {rmse_pos_xyz_gnss_2[1]:.4f}, {rmse_pos_xyz_gnss_2[2]:.4f}")
# print(f"Position RMSE(GNSS)_3 [m]      (N,E,D): {rmse_pos_xyz_gnss_3[0]:.4f}, {rmse_pos_xyz_gnss_3[1]:.4f}, {rmse_pos_xyz_gnss_3[2]:.4f}")
# print(f"Position RMSE(GNSS)_4 [m]      (N,E,D): {rmse_pos_xyz_gnss_4[0]:.4f}, {rmse_pos_xyz_gnss_4[1]:.4f}, {rmse_pos_xyz_gnss_4[2]:.4f}")

# print(f"Position RMSE(GNSS)_1_all [m]      (Overall): {rmse_pos_xyz_gnss_1_all:.4f}")
# print(f"Position RMSE(GNSS)_2_all [m]      (Overall): {rmse_pos_xyz_gnss_2_all:.4f}")
# print(f"Position RMSE(GNSS)_3_all [m]      (Overall): {rmse_pos_xyz_gnss_3_all:.4f}")
# print(f"Position RMSE(GNSS)_4_all [m]      (Overall): {rmse_pos_xyz_gnss_4_all:.4f}")

# print("======================================================\n")


axs3[0].plot(t, pos_err_norm_1, label=f"ESKF RMSE: {np.sqrt(np.mean(pos_err_norm_1**2)):.4f}")
axs3[0].plot(t, pos_err_norm_2, label=f"S-H_ESKF RMSE: {np.sqrt(np.mean(pos_err_norm_2**2)):.4f}")
axs3[0].plot(t, pos_err_norm_3, label=f"new RMSE: {np.sqrt(np.mean(pos_err_norm_3**2)):.4f}")
axs3[0].set(ylabel="Position error [m]", xlabel="Time [s]")
axs3[0].legend()

axs3[1].plot(t, vel_err_norm_1, label=f"ESKF RMSE: {np.sqrt(np.mean(vel_err_norm_1**2)):.4f}")
axs3[1].plot(t, vel_err_norm_2, label=f"S-H_ESKF RMSE: {np.sqrt(np.mean(vel_err_norm_2**2)):.4f}")
axs3[1].plot(t, vel_err_norm_3, label=f"new RMSE: {np.sqrt(np.mean(vel_err_norm_3**2)):.4f}")
axs3[1].set(ylabel="Velocity error [m/s]", xlabel="Time [s]")
axs3[1].legend()

axs3[2].plot(t, att_err_norm_deg_1, label=f"ESKF RMSE: {np.sqrt(np.mean(att_err_norm_deg_1**2)):.4f}")
axs3[2].plot(t, att_err_norm_deg_2, label=f"S-H_ESKF RMSE: {np.sqrt(np.mean(att_err_norm_deg_2**2)):.4f}")
axs3[2].plot(t, att_err_norm_deg_3, label=f"new RMSE: {np.sqrt(np.mean(att_err_norm_deg_3**2)):.4f}")
axs3[2].set(ylabel="Attitude error [deg]", xlabel="Time [s]")
axs3[2].legend()

axs3[3].plot(t, acc_bias_err_norm_1, label=f"ESKF RMSE: {np.sqrt(np.mean(acc_bias_err_norm_1**2)):.4f}")
axs3[3].plot(t, acc_bias_err_norm_2, label=f"S-H_ESKF RMSE: {np.sqrt(np.mean(acc_bias_err_norm_2**2)):.4f}")
axs3[3].plot(t, acc_bias_err_norm_3, label=f"new RMSE: {np.sqrt(np.mean(acc_bias_err_norm_3**2)):.4f}")
axs3[3].set(ylabel="Acc bias error [m/s^2]", xlabel="Time [s]")
axs3[3].legend()

axs3[4].plot(t, gyro_bias_err_norm_deg_h_1, label=f"ESKF RMSE: {np.sqrt(np.mean(gyro_bias_err_norm_deg_h_1**2)):.4f}")
axs3[4].plot(t, gyro_bias_err_norm_deg_h_2, label=f"S-H_ESKF RMSE: {np.sqrt(np.mean(gyro_bias_err_norm_deg_h_2**2)):.4f}")
axs3[4].plot(t, gyro_bias_err_norm_deg_h_3, label=f"new RMSE: {np.sqrt(np.mean(gyro_bias_err_norm_deg_h_3**2)):.4f}")
axs3[4].set(ylabel="Gyro bias error [deg/h]", xlabel="Time [s]")
axs3[4].legend()

# # R 对角线随时间变化
# valid_r_history = R_GNSS_history[:GNSSk]
# fig4, ax4 = plt.subplots(1, 1, num=4, clear=True)
# ax4.plot(timeGNSS[:GNSSk], valid_r_history[:, 0, 0], label='R[0,0]')
# ax4.plot(timeGNSS[:GNSSk], valid_r_history[:, 1, 1], label='R[1,1]')
# ax4.plot(timeGNSS[:GNSSk], valid_r_history[:, 2, 2], label='R[2,2]')
# ax4.set(xlabel='Time [s]', ylabel='R diagonal')
# ax4.set_title('GNSS R diagonal over time')
# ax4.grid(True)
# ax4.legend()

# if save_fig_dir:
#     out_dir = Path(save_fig_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     prefix = f"{fig_prefix}_" if fig_prefix else ""

#     if save_fullscreen_fig:
#         apply_fullscreen_canvas(fig1)
#         apply_fullscreen_canvas(fig2)
#         apply_fullscreen_canvas(fig3)
#         apply_fullscreen_canvas(fig4)

#     fig1.savefig(out_dir / f"{prefix}fig1_trajectory.png", dpi=200, bbox_inches="tight")
#     fig2.savefig(out_dir / f"{prefix}fig2_states.png", dpi=200, bbox_inches="tight")
#     fig3.savefig(out_dir / f"{prefix}fig3_rmse.png", dpi=200, bbox_inches="tight")
#     fig4.savefig(out_dir / f"{prefix}fig4_rdiag.png", dpi=200, bbox_inches="tight")
#     print(f"Saved figures to: {out_dir.resolve()}")

if no_show_fig:
    plt.close('all')
else:
    plt.show()