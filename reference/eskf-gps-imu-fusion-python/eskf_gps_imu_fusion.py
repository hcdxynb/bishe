import os
import yaml
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R

# -----------------------------------------------------------------------------
# ESKF GPS/IMU Fusion 模块
# 该文件实现了一个基于误差状态卡尔曼滤波（Error-State Kalman Filter, ESKF）的
# IMU(惯性测量单元)与GPS融合系统。包含数据结构、坐标变换、IMU/GPS读取、
# 状态预测与更新逻辑。
# -----------------------------------------------------------------------------

# 数据容器：IMU 传感器数据（观测与真实值）
class IMUData:
    """IMU 数据结构，保存时间戳、加速度、角速度及地真值"""

    def __init__(self):
        self.time = 0.0  # 时间戳
        self.linear_accel = np.zeros(3)
        self.angle_velocity = np.zeros(3)

        # 真实值（仿真或参考数据，用于评估）
        self.true_linear_accel = np.zeros(3)
        self.true_angle_velocity = np.zeros(3)
        self.true_q_enu = R.from_quat([0, 0, 0, 1])  # 真实 ENU 姿态（四元数）
        self.true_t_enu = np.zeros(3)  # 真实 ENU 位置


class GPSData:
    """GPS 数据结构,包含经纬高、速度、NED 本地坐标及真值"""

    def __init__(self):
        self.position_lla = np.zeros(3)  # [lat, lon, alt]
        self.velocity = np.zeros(3)
        self.local_position_ned = np.zeros(3)  # NED 本地坐标系
        self.true_velocity = np.zeros(3)
        self.true_position_lla = np.zeros(3)
        self.time = 0.0


class ConfigParameters:
    """配置参数类，支持从 YAML 文件读取滤波参数"""

    def __init__(self):
        self.earth_rotation_speed = 0.0
        self.earth_gravity = 9.8
        self.ref_longitude = 0.0
        self.ref_latitude = 0.0
        self.ref_altitude = 0.0
        self.position_error_prior_std = 1e-5
        self.velocity_error_prior_std = 1e-5
        self.rotation_error_prior_std = 1e-5
        self.accelerometer_bias_error_prior_std = 1e-5
        self.gyro_bias_error_prior_std = 1e-5
        self.gyro_noise_std = 1e-2
        self.accelerometer_noise_std = 1e-1
        self.gps_position_x_std = 5.0
        self.gps_position_y_std = 5.0
        self.gps_position_z_std = 8.0
        self.only_prediction = False
        self.use_earth_model = True

    def load(self, path):
        """从 YAML 文件加载参数，若字段缺失则使用默认值"""
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        self.earth_rotation_speed = float(cfg.get('earth_rotation_speed', self.earth_rotation_speed))
        self.earth_gravity = float(cfg.get('earth_gravity', self.earth_gravity))
        self.ref_longitude = float(cfg.get('ref_longitude', self.ref_longitude))
        self.ref_latitude = float(cfg.get('ref_latitude', self.ref_latitude))
        self.ref_altitude = float(cfg.get('ref_altitude', self.ref_altitude))
        self.position_error_prior_std = float(cfg.get('position_error_prior_std', self.position_error_prior_std))
        self.velocity_error_prior_std = float(cfg.get('velocity_error_prior_std', self.velocity_error_prior_std))
        self.rotation_error_prior_std = float(cfg.get('rotation_error_prior_std', self.rotation_error_prior_std))
        self.accelerometer_bias_error_prior_std = float(cfg.get('accelerometer_bias_error_prior_std', self.accelerometer_bias_error_prior_std))
        self.gyro_bias_error_prior_std = float(cfg.get('gyro_bias_error_prior_std', self.gyro_bias_error_prior_std))
        self.gyro_noise_std = float(cfg.get('gyro_noise_std', self.gyro_noise_std))
        self.accelerometer_noise_std = float(cfg.get('accelerometer_noise_std', self.accelerometer_noise_std))
        self.gps_position_x_std = float(cfg.get('gps_position_x_std', self.gps_position_x_std))
        self.gps_position_y_std = float(cfg.get('gps_position_y_std', self.gps_position_y_std))
        self.gps_position_z_std = float(cfg.get('gps_position_z_std', self.gps_position_z_std))
        self.only_prediction = bool(cfg.get('only_prediction', self.only_prediction))
        self.use_earth_model = bool(cfg.get('use_earth_model', self.use_earth_model))


def skew_symmetric(v):
    """返回向量 v 的反对称矩阵，用于叉乘矩阵表达"""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def so3_exp(v):
    """SO(3)指数映射：从小角度误差向量到旋转矩阵"""
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return np.eye(3)
    k = v / theta
    K = skew_symmetric(k)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


class GPSTool:
    """GPS坐标变换和数据读取工具"""

    def __init__(self, lon, lat, alt):
        self.ref_lon = lon
        self.ref_lat = lat
        self.ref_alt = alt
        self.a = 6378137.0
        self.f = 1.0 / 298.257223563
        self.e2 = self.f * (2 - self.f)

        self.ref_ecef = self.lla_to_ecef(np.deg2rad(self.ref_lat), np.deg2rad(self.ref_lon), self.ref_alt)

    def lla_to_ecef(self, lat_rad, lon_rad, alt):
        """从 WGS-84 大地坐标（经纬高）转换到 ECEF 地心坐标"""
        N = self.a / np.sqrt(1 - self.e2 * np.sin(lat_rad) ** 2)
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - self.e2) + alt) * np.sin(lat_rad)
        return np.array([x, y, z])

    def ecef_to_enu(self, ecef):
        """ECEF 转 ENU（局部东-北-上）坐标系"""
        dx = ecef - self.ref_ecef
        lat_rad = np.deg2rad(self.ref_lat)
        lon_rad = np.deg2rad(self.ref_lon)

        t = np.array([[-np.sin(lon_rad), np.cos(lon_rad), 0],
                      [-np.sin(lat_rad) * np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad)],
                      [np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)]])
        return t.dot(dx)

    def lla_to_local_ned_vec(self, lla):
        lat_rad = np.deg2rad(lla[0])
        lon_rad = np.deg2rad(lla[1])
        alt = lla[2]

        ecef = self.lla_to_ecef(lat_rad, lon_rad, alt)
        enu = self.ecef_to_enu(ecef)
        # ENU -> NED
        return np.array([enu[1], enu[0], -enu[2]])

    def read_gps_data(self, path, skip_rows=1):
        """从数据目录读取 GPS 测量与参考数据，并转换到 NED 本地坐标"""
        gps_path = os.path.join(path, 'gps-0.csv')
        ref_gps_path = os.path.join(path, 'ref_gps.csv')
        gps_time_path = os.path.join(path, 'gps_time.csv')

        gps_data = []
        with open(gps_path, 'r', encoding='utf-8') as f0, \
             open(ref_gps_path, 'r', encoding='utf-8') as f1, \
             open(gps_time_path, 'r', encoding='utf-8') as f2:
            for _ in range(skip_rows):
                next(f0, None)
                next(f1, None)
                next(f2, None)

            for line0, line1, line2 in zip(f0, f1, f2):
                item0 = [float(i) for i in line0.strip().split(',')]
                item1 = [float(i) for i in line1.strip().split(',')]
                t = float(line2.strip())

                d = GPSData()
                d.time = t
                d.position_lla = np.array(item0[0:3])
                d.velocity = np.array(item0[3:6])
                d.true_position_lla = np.array(item1[0:3])
                d.true_velocity = np.array(item1[3:6])
                d.local_position_ned = self.lla_to_local_ned_vec(d.position_lla)

                gps_data.append(d)
        return deque(gps_data)


class IMUTool:
    """IMU 数据读取与预处理工具"""

    DEG2RAD = np.pi / 180.0

    def read_imu_data(self, path, skip_rows=1):
        """从数据目录读取 IMU 传感器数据与参考真值，并归一化角度单位"""
        accel_path = os.path.join(path, 'accel-0.csv')
        ref_accel_path = os.path.join(path, 'ref_accel.csv')
        gyro_path = os.path.join(path, 'gyro-0.csv')
        ref_gyro_path = os.path.join(path, 'ref_gyro.csv')
        time_path = os.path.join(path, 'time.csv')
        ref_att_quat_path = os.path.join(path, 'ref_att_quat.csv')
        ref_pos_path = os.path.join(path, 'ref_pos.csv')

        with open(accel_path, 'r', encoding='utf-8') as fa, \
             open(ref_accel_path, 'r', encoding='utf-8') as fra, \
             open(gyro_path, 'r', encoding='utf-8') as fg, \
             open(ref_gyro_path, 'r', encoding='utf-8') as frg, \
             open(time_path, 'r', encoding='utf-8') as ft, \
             open(ref_att_quat_path, 'r', encoding='utf-8') as fq, \
             open(ref_pos_path, 'r', encoding='utf-8') as fp:
            for _ in range(skip_rows):
                next(fa, None)
                next(fra, None)
                next(fg, None)
                next(frg, None)
                next(ft, None)
                next(fq, None)
                next(fp, None)

            imu_data = []
            for la, lra, lg, lrg, lt, lq, lp in zip(fa, fra, fg, frg, ft, fq, fp):
                a = [float(x) for x in la.strip().split(',')]
                ra = [float(x) for x in lra.strip().split(',')]
                g = [float(x) for x in lg.strip().split(',')]
                rg = [float(x) for x in lrg.strip().split(',')]
                t = float(lt.strip())
                quat = [float(x) for x in lq.strip().split(',')]
                pos = [float(x) for x in lp.strip().split(',')]

                d = IMUData()
                d.time = t
                d.linear_accel = np.array(a)
                d.true_linear_accel = np.array(ra)
                d.angle_velocity = np.array(g) * self.DEG2RAD
                d.true_angle_velocity = np.array(rg) * self.DEG2RAD
                d.true_t_enu = np.array(pos)
                # quaternion order: w,x,y,z in C++
                d.true_q_enu = R.from_quat([quat[1], quat[2], quat[3], quat[0]])

                imu_data.append(d)
        return deque(imu_data)


class ErrorStateKalmanFilter:
    """误差状态卡尔曼滤波器实现（15 维状态）"""
    DIM_STATE = 15
    IND_POS = 0
    IND_VEL = 3
    IND_ORI = 6
    IND_GYRO_BIAS = 9
    IND_ACC_BIAS = 12

    def __init__(self, cfg: ConfigParameters):
        """初始化状态与协方差矩阵"""
        self.cfg = cfg
        self.g = np.array([0.0, 0.0, -cfg.earth_gravity])
        self.pose = np.eye(4)
        self.velocity = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)
        self.X = np.zeros((15, 1))

        self.F = np.zeros((15, 15))
        self.B = np.zeros((15, 6))
        self.Q = np.zeros((6, 6))
        self.P = np.zeros((15, 15))
        self.K = np.zeros((3, 15))
        self.C = np.eye(3)
        self.G = np.zeros((3, 15))
        self.R = np.zeros((3, 3))

        self.G[:, 0:3] = np.eye(3)
        self.Ft = np.zeros((15, 15))
        self.Y = np.zeros((3, 1))

        self.imu_data_buff = deque()
        self.curr_gps_data = None

        self.set_covariance_p(cfg.position_error_prior_std,
                              cfg.velocity_error_prior_std,
                              cfg.rotation_error_prior_std,
                              cfg.gyro_bias_error_prior_std,
                              cfg.accelerometer_bias_error_prior_std)
        self.set_covariance_r(cfg.gps_position_x_std,
                              cfg.gps_position_y_std,
                              cfg.gps_position_z_std)
        self.set_covariance_q(cfg.gyro_noise_std,
                              cfg.accelerometer_noise_std)

    def set_covariance_p(self, posi_noise, velocity_noise, ori_noise, gyro_noise, accel_noise):
        self.P = np.zeros((15, 15))
        self.P[0:3, 0:3] = np.eye(3) * (posi_noise**2)
        self.P[3:6, 3:6] = np.eye(3) * (velocity_noise**2)
        self.P[6:9, 6:9] = np.eye(3) * (ori_noise**2)
        self.P[9:12, 9:12] = np.eye(3) * (gyro_noise**2)
        self.P[12:15, 12:15] = np.eye(3) * (accel_noise**2)

    def set_covariance_r(self, x_std, y_std, z_std):
        self.R = np.diag([x_std**2, y_std**2, z_std**2])

    def set_covariance_q(self, gyro_noise, accel_noise):
        """设定过程噪声协方差矩阵 Q"""
        self.Q = np.zeros((6, 6))
        self.Q[0:3, 0:3] = np.eye(3) * (gyro_noise**2)
        self.Q[3:6, 3:6] = np.eye(3) * (accel_noise**2)

    def init(self, gps_data: GPSData, imu_data: IMUData):
        """滤波器初始化：从 GPS 和第一帧 IMU 数据设定初始位姿与速度"""
        self.velocity = gps_data.true_velocity.copy()

        self.pose = np.eye(4)
        self.pose[0:3, 0:3] = R.from_euler('xyz', [0.0, 0.0, 0.0]).as_matrix()
        self.pose[0:3, 3] = gps_data.local_position_ned.copy()

        self.imu_data_buff.clear()
        self.imu_data_buff.append(imu_data)

        self.curr_gps_data = gps_data

    def get_pose(self):
        return self.pose

    def compute_navigation_frame_angular_velocity(self):
        """计算导航系（NED）相对于地球惯性系的角速度，用于考虑地球自转与纬度效应"""
        lat = np.deg2rad(self.curr_gps_data.position_lla[1])
        h = self.curr_gps_data.position_lla[2]

        f = 1.0 / 298.257223563
        Re = 6378137.0
        Rp = (1.0 - f) * Re
        e = np.sqrt(Re*Re - Rp*Rp) / Re
        Rn = Re / np.sqrt(1 - e*e*np.sin(lat)**2)
        Rm = Re*(1-e*e)/((1-e*e*np.sin(lat)**2)**1.5)

        w_en_n = np.array([
            self.velocity[1] / (Rm + h),
            -self.velocity[0] / (Rn + h),
            -self.velocity[1] / (Rn + h) * np.tan(lat)
        ])

        w_ie_n = np.array([
            self.cfg.earth_rotation_speed * np.cos(lat),
            0.0,
            -self.cfg.earth_rotation_speed * np.sin(lat)
        ])

        return w_en_n + w_ie_n

    def compute_delta_rotation(self, imu_data_0: IMUData, imu_data_1: IMUData):
        dt = imu_data_1.time - imu_data_0.time
        assert dt > 0

        uw0 = imu_data_0.angle_velocity - self.gyro_bias
        uw1 = imu_data_1.angle_velocity - self.gyro_bias
        return 0.5 * (uw0 + uw1) * dt

    def compute_unbias_accel(self, accel):
        return accel - self.accel_bias

    def compute_unbias_gyro(self, gyro):
        return gyro - self.gyro_bias

    def compute_orientation(self, angular_delta, R_nm_nm_1):
        last_R = self.pose[0:3, 0:3].copy()
        delta_R = R.from_rotvec(angular_delta).as_matrix()
        curr_R = R_nm_nm_1.T @ last_R @ delta_R
        self.pose[0:3, 0:3] = curr_R
        return curr_R, last_R

    def compute_velocity(self, R0, R1, imu0, imu1):
        dt = imu1.time - imu0.time
        assert dt > 0

        a0 = R0 @ self.compute_unbias_accel(imu0.linear_accel) - self.g
        a1 = R1 @ self.compute_unbias_accel(imu1.linear_accel) - self.g

        last_vel = self.velocity.copy()
        self.velocity = self.velocity + 0.5 * dt * (a0 + a1)
        return last_vel, self.velocity

    def compute_position(self, R0, R1, last_vel, curr_vel, imu0, imu1):
        """基于速度-加速度积分预测当前位置（惯导导航方程）"""
        dt = imu1.time - imu0.time
        a0 = R0 @ self.compute_unbias_accel(imu0.linear_accel) - self.g
        a1 = R1 @ self.compute_unbias_accel(imu1.linear_accel) - self.g

        self.pose[0:3, 3] += 0.5*dt*(curr_vel + last_vel) + 0.25*dt*dt*(a0 + a1)

    def update_odometry_estimation(self, w_in):
        """利用最新两帧 IMU 更新姿态、速度、位置（预积分）"""
        imu0 = self.imu_data_buff[0]
        imu1 = self.imu_data_buff[1]
        dt = imu1.time - imu0.time

        delta_rotation = self.compute_delta_rotation(imu0, imu1)
        phi_in = w_in * dt
        norm_phi = np.linalg.norm(phi_in)
        if norm_phi < 1e-12:
            R_nm_nm_1 = np.eye(3)
        else:
            R_nm_nm_1 = R.from_rotvec(phi_in).as_matrix().T

        curr_R, last_R = self.compute_orientation(delta_rotation, R_nm_nm_1)
        last_vel, curr_vel = self.compute_velocity(last_R, curr_R, imu0, imu1)
        self.compute_position(last_R, curr_R, last_vel, curr_vel, imu0, imu1)

    def update_error_state(self, dt, accel, w_in_n):
        F23 = skew_symmetric(accel)
        F33 = -skew_symmetric(w_in_n)

        self.F = np.zeros((15, 15))
        self.F[0:3, 3:6] = np.eye(3)
        self.F[3:6, 6:9] = F23
        self.F[6:9, 6:9] = F33
        self.F[3:6, 12:15] = self.pose[0:3, 0:3]
        self.F[6:9, 9:12] = -self.pose[0:3, 0:3]

        self.B = np.zeros((15, 6))
        self.B[3:6, 3:6] = self.pose[0:3, 0:3]
        self.B[6:9, 0:3] = -self.pose[0:3, 0:3]

        Fk = np.eye(15) + self.F*dt
        Bk = self.B*dt

        self.Ft = self.F * dt

        self.X = Fk @ self.X
        self.P = Fk @ self.P @ Fk.T + Bk @ self.Q @ Bk.T

    def estimate(self, imu_data):
        """预测步骤：用 IMU 数据预积分并更新误差状态"""
        self.imu_data_buff.append(imu_data)

        w_in = np.zeros(3)
        if self.cfg.use_earth_model:
            w_in = self.compute_navigation_frame_angular_velocity()

        self.update_odometry_estimation(w_in)

        dt = imu_data.time - self.imu_data_buff[0].time
        accel_world = self.pose[0:3, 0:3] @ imu_data.linear_accel
        self.update_error_state(dt, accel_world, w_in)

        self.imu_data_buff.popleft()

    def correct(self, gps_data):
        """校正步骤：用 GPS 位置测量值更新滤波器状态"""
        self.curr_gps_data = gps_data
        residual = gps_data.local_position_ned.reshape((3, 1)) - self.pose[0:3, 3].reshape((3, 1))
        self.Y = residual

        S = self.G @ self.P @ self.G.T + self.C @ self.R @ self.C.T
        K = self.P @ self.G.T @ np.linalg.inv(S)

        self.P = (np.eye(15) - K @ self.G) @ self.P
        self.X = self.X + K @ (self.Y - self.G @ self.X)

        self.pose[0:3, 3] += self.X[0:3, 0]
        self.velocity += self.X[3:6, 0]

        delta_ori = -self.X[6:9, 0]
        C_nn = so3_exp(delta_ori)
        self.pose[0:3, 0:3] = C_nn @ self.pose[0:3, 0:3]

        self.gyro_bias += self.X[9:12, 0]
        self.accel_bias += self.X[12:15, 0]

        self.X[:] = 0


class ESKFFlow:
    """ESKF 主流程：从文件读取数据，运行预测+校正，写出结果"""

    def __init__(self, config_file_path, data_file_path):
        self.config = ConfigParameters()
        self.config.load(config_file_path)
        self.gps_tool = GPSTool(self.config.ref_longitude, self.config.ref_latitude, self.config.ref_altitude)
        self.imu_tool = IMUTool()
        self.eskf = ErrorStateKalmanFilter(self.config)
        self.data_file_path = data_file_path
        self.imu_data_buff = deque()
        self.gps_data_buff = deque()

    def read_data(self):
        raw_path = self.data_file_path
        # allow passing either parent folder (which contains raw_data) or raw_data folder itself
        if os.path.basename(raw_path).lower() != 'raw_data':
            raw_path = os.path.join(raw_path, 'raw_data')

        self.imu_data_buff = self.imu_tool.read_imu_data(raw_path)
        self.gps_data_buff = self.gps_tool.read_gps_data(raw_path)

    def valid_gps_and_imu(self):
        imu = self.imu_data_buff[0]
        gps = self.gps_data_buff[0]

        dt = imu.time - gps.time
        if dt > 0.05:
            self.gps_data_buff.popleft()
            return False
        if dt < -0.05:
            self.imu_data_buff.popleft()
            return False

        self.imu_data_buff.popleft()
        self.gps_data_buff.popleft()

        return imu, gps

    def run(self):
        self.read_data()

        # init
        while self.imu_data_buff and self.gps_data_buff:
            valid = self.valid_gps_and_imu()
            if not valid:
                continue
            imu_data, gps_data = valid
            self.eskf.init(gps_data, imu_data)
            break

        gc_path = os.path.join(self.data_file_path, 'gt.txt')
        fused_path = os.path.join(self.data_file_path, 'fused.txt')
        meas_path = os.path.join(self.data_file_path, 'gps_measurement.txt')

        with open(gc_path, 'w', encoding='utf-8') as f_gt, \
             open(fused_path, 'w', encoding='utf-8') as f_fused, \
             open(meas_path, 'w', encoding='utf-8') as f_meas:
            while self.imu_data_buff and self.gps_data_buff:
                imu_data = self.imu_data_buff[0]
                gps_data = self.gps_data_buff[0]

                self.eskf.estimate(imu_data)
                self.imu_data_buff.popleft()

                if imu_data.time >= gps_data.time:
                    if not self.config.only_prediction:
                        self.eskf.correct(gps_data)

                    self.save_tum_pose(f_fused, self.eskf.get_pose(), imu_data.time)
                    self.save_tum_pose(f_meas, np.eye(4), gps_data.time, position=gps_data.local_position_ned)
                    gt_q = np.eye(4)
                    gt_q[0:3, 3] = imu_data.true_t_enu
                    self.save_tum_pose(f_gt, gt_q, imu_data.time)

                    self.gps_data_buff.popleft()

    @staticmethod
    def save_tum_pose(f, pose, timestamp, position=None):
        if position is None:
            t = pose[0:3, 3]
            q = R.from_matrix(pose[0:3, 0:3]).as_quat()
        else:
            t = position
            q = np.array([0.0, 0.0, 0.0, 1.0])

        f.write(f"{timestamp:.10f} {t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='ESKF GPS-IMU fusion python version')
    parser.add_argument('config', help='config file path')
    parser.add_argument('data', help='data folder path')
    args = parser.parse_args()

    flow = ESKFFlow(args.config, args.data)
    flow.run()
