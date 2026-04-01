import numpy as np
from scipy.spatial.transform import Rotation as R
from .utils import skew_symmetric, so3_exp


class ErrorStateKalmanFilter:
    DIM_STATE = 15
    IND_POS = 0
    IND_VEL = 3
    IND_ORI = 6
    IND_GYRO_BIAS = 9
    IND_ACC_BIAS = 12

    def __init__(self, cfg):
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
        self.C = np.eye(3)
        self.G = np.zeros((3, 15))
        self.R = np.zeros((3, 3))

        self.G[:, 0:3] = np.eye(3)
        self.Ft = np.zeros((15, 15))
        self.Y = np.zeros((3, 1))

        self.imu_data_buff = []
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
        self.Q = np.zeros((6, 6))
        self.Q[0:3, 0:3] = np.eye(3) * (gyro_noise**2)
        self.Q[3:6, 3:6] = np.eye(3) * (accel_noise**2)

    def init(self, gps_data, imu_data):
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

    def compute_delta_rotation(self, imu_data_0, imu_data_1):
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
        dt = imu1.time - imu0.time
        a0 = R0 @ self.compute_unbias_accel(imu0.linear_accel) - self.g
        a1 = R1 @ self.compute_unbias_accel(imu1.linear_accel) - self.g

        self.pose[0:3, 3] += 0.5 * dt * (curr_vel + last_vel) + 0.25 * dt * dt * (a0 + a1) # 使用当前 IMU 数据进行位置更新

    def update_odometry_estimation(self, w_in):
        imu0 = self.imu_data_buff[0] # 获取当前缓冲区中的第一条 IMU 数据作为基准
        imu1 = self.imu_data_buff[1] # 获取当前缓冲区中的第二条 IMU 数据作为当前时刻的 IMU 数据
        dt = imu1.time - imu0.time

        delta_rotation = self.compute_delta_rotation(imu0, imu1) # 计算两条 IMU 数据之间的旋转增量
        phi_in = w_in * dt
        norm_phi = np.linalg.norm(phi_in)
        if norm_phi < 1e-12:
            R_nm_nm_1 = np.eye(3)
        else:
            R_nm_nm_1 = R.from_rotvec(phi_in).as_matrix().T

        curr_R, last_R = self.compute_orientation(delta_rotation, R_nm_nm_1) # 计算当前时刻的旋转矩阵，并返回当前和上一个时刻的旋转矩阵
        last_vel, curr_vel = self.compute_velocity(last_R, curr_R, imu0, imu1) # 计算当前时刻的速度，并返回当前和上一个时刻的速度
        self.compute_position(last_R, curr_R, last_vel, curr_vel, imu0, imu1) # 使用 当前 IMU 数据进行位置更新

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

        Fk = np.eye(15) + self.F * dt
        Bk = self.B * dt

        self.Ft = self.F * dt

        self.X = Fk @ self.X
        self.P = Fk @ self.P @ Fk.T + Bk @ self.Q @ Bk.T

    def estimate(self, imu_data):
        self.imu_data_buff.append(imu_data) # 将当前 IMU 数据添加到缓冲区

        w_in = np.zeros(3)
        if self.cfg.use_earth_model:
            w_in = self.compute_navigation_frame_angular_velocity()

        self.update_odometry_estimation(w_in) # 使用当前 IMU 数据进行 ESKF 预测

        dt = imu_data.time - self.imu_data_buff[0].time
        accel_world = self.pose[0:3, 0:3] @ imu_data.linear_accel
        self.update_error_state(dt, accel_world, w_in)

        self.imu_data_buff.pop(0) 

    def correct(self, gps_data):
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
