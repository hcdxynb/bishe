# 导入基础依赖
import scipy
import scipy.io
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

# 读取配置参数
with open(r'params.yaml') as file:
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

# 加载数据
filename_to_load = "task_simulation.mat"
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
steps = len(z_acceleration)
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
    debug=False # TODO: 设为 False 可避免开销较大的调试检查，也可用 'python -O run_INS_simulated.py' 关闭断言
)

steps=90000
# 预分配数组
x_est = np.zeros((steps, 16))
P_est = np.zeros((steps, 15, 15))

x_pred = np.zeros((steps, 16))
P_pred = np.zeros((steps, 15, 15))

v_post = np.zeros((steps, 3)) # 记录每次 GNSS 更新后的测量残差（后验残差）

delta_x = np.zeros((steps, 15))


# 初始化
x_pred[0, POS_IDX] = np.array([0, 0, -5])  # 初始高度离地 5 米
x_pred[0, VEL_IDX] = np.array([20, 0, 0])  # 初速度 20 m/s，朝正北方向
x_pred[0, 6] = 1  # 初始无旋转：机头朝北、机体右侧朝东、机腹向下

# 这些初值需要设置合理，才能获得较好的估计结果
P_pred[0][POS_IDX ** 2] = params["P_pred0_pos"]*np.eye(3)# TODO: 继续调参
P_pred[0][VEL_IDX ** 2] = params["P_pred0_vel"]*np.eye(3)# TODO: 继续调参
P_pred[0][ERR_ATT_IDX ** 2] = params["P_pred0_att"]*np.eye(3)# TODO: 继续调参 # 误差旋转向量（非四元数）
P_pred[0][ERR_ACC_BIAS_IDX ** 2] = params["P_pred0_accbias"]*np.eye(3)# TODO: 继续调参
P_pred[0][ERR_GYRO_BIAS_IDX ** 2] = params["P_pred0_gyrobias"]*np.eye(3)# TODO: 继续调参

# 使用 'python -O run_INS_simulated.py' 运行可关闭断言，长时运行时大约可提升 8/5 的速度
# 可选：固定估计步数
N: int = steps # TODO: 可先从较小值开始（如 500），结果稳定后再逐步增大
doGNSS: bool = True  # TODO: 若想检查纯预测在合理时长内是否稳定，可设为 False


# 主循环 滤波过程
GNSSk: int = 0  # 记录当前 GNSS 测量索引
for k in tqdm(range(N)):
    if doGNSS and timeIMU[k] >= timeGNSS[GNSSk]:
        v_prior = z_GNSS[GNSSk] - x_pred[k, POS_IDX] # 测量残差（先验残差）
        
        if GNSSk == 0:
            x_est[k], P_est[k], R_GNSS =eskf.update_GNSS_position(x_pred[k],P_pred[k],R_GNSS,GNSSk,v_prior,np.zeros(3))
        else:
            x_est[k], P_est[k], R_GNSS =eskf.update_GNSS_position(x_pred[k],P_pred[k],R_GNSS,GNSSk,v_prior,v_post[GNSSk-1])
            v_post[GNSSk] = z_GNSS[GNSSk] - x_est[k, POS_IDX] # 测量残差（后验残差）

        GNSSk += 1
    else:
        # 无观测更新时，令估计值等于预测值
        x_est[k] = x_pred[k]
        P_est[k] = P_pred[k]

    delta_x[k] = eskf.delta_x(x_est[k], x_true[k])

    if k < N - 1:
        x_pred[k + 1], P_pred[k + 1] = eskf.predict(x_est[k],P_est[k],z_acceleration[k],z_gyroscope[k],dt)# TODO: 提示：测量来自当前与过去时刻，而不是未来



# 绘图
# 状态估计结果与真值的比较，以及误差分析

# 轨迹比较
fig1 = plt.figure(1)
fig1.suptitle("Trajectory comparison")
ax = plt.axes(projection="3d")

ax.plot3D(x_est[:N, 1], x_est[:N, 0], -x_est[:N, 2],color='blue') # 前N行，第1列（东）、第0列（北）、第2列（下，取负号变为高度）
ax.plot3D(z_GNSS[:GNSSk, 1], z_GNSS[:GNSSk, 0], -z_GNSS[:GNSSk, 2],color='red')
ax.plot3D(x_true[:N, 1], x_true[:N, 0], -x_true[:N, 2],color='yellow')
ax.set_xlabel("East [m]")
ax.set_ylabel("North [m]")
ax.set_zlabel("Altitude [m]")

t = np.linspace(0, dt * (N - 1), N)
eul = np.apply_along_axis(quaternion_to_euler, 1, x_est[:N, ATT_IDX])
eul_true = np.apply_along_axis(quaternion_to_euler, 1, x_true[:N, ATT_IDX])

# 状态估计曲线
fig2, axs2 = plt.subplots(5, 1, num=2, clear=True)
fig2.suptitle("States estimates")

axs2[0].plot(t, x_est[:N, POS_IDX])
axs2[0].set(ylabel="NED position [m]",xlabel="Time [s]")
axs2[0].legend(["North", "East", "Down"])


axs2[1].plot(t, x_est[:N, VEL_IDX])
axs2[1].set(ylabel="Velocities [m/s]",xlabel="Time [s]")
axs2[1].legend(["North", "East", "Down"])


axs2[2].plot(t, eul[:N] * 180 / np.pi)
axs2[2].set(ylabel="Euler angles [deg]",xlabel="Time [s]")
axs2[2].legend([r"$\phi$", r"$\theta$", r"$\psi$"])


axs2[3].plot(t, x_est[:N, ACC_BIAS_IDX])
axs2[3].set(ylabel="Accl bias [m/s^2]",xlabel="Time [s]")
axs2[3].legend(["$x$", "$y$", "$z$"])


axs2[4].plot(t, x_est[:N, GYRO_BIAS_IDX] * 180 / np.pi * 3600)
axs2[4].set(ylabel="Gyro bias [deg/h]",xlabel="Time [s]")
axs2[4].legend(["$x$", "$y$", "$z$"])

# 误差范数曲线(RMSE)
fig4, axs4 = plt.subplots(2, 1, num=4, clear=True)

axs4[0].plot(t, np.linalg.norm(delta_x[:N, POS_IDX], axis=1))
axs4[0].plot(
    np.arange(0, N, 100) * dt,
    np.linalg.norm(x_true[99:N:100, :3] - z_GNSS[:GNSSk], axis=1),
)
axs4[0].set(ylabel="Position error [m]")
axs4[0].legend(
    [
        f"Estimation(ESKF) error(RMSE) ({np.sqrt(np.mean(np.sum(delta_x[:N, POS_IDX]**2, axis=1)))})",
        f"Measurement(GNSS) error(RMSE) ({np.sqrt(np.mean(np.sum((x_true[99:N:100, POS_IDX] - z_GNSS[:GNSSk])**2, axis=1)))})",
    ]
)

axs4[1].plot(t, np.linalg.norm(delta_x[:N, VEL_IDX], axis=1))
axs4[1].set(ylabel="Speed error [m/s]")
axs4[1].legend([f"RMSE: {np.sqrt(np.mean(np.sum(delta_x[:N, VEL_IDX]**2, axis=0)))}"])

plt.show()
