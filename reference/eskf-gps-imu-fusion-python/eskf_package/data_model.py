import numpy as np
from scipy.spatial.transform import Rotation as R


class IMUData:
    """IMU 数据结构，保存时间戳、观测值与真值"""

    def __init__(self):
        self.time = 0.0
        self.linear_accel = np.zeros(3)
        self.angle_velocity = np.zeros(3)

        self.true_linear_accel = np.zeros(3)
        self.true_angle_velocity = np.zeros(3)
        self.true_q_enu = R.from_quat([0, 0, 0, 1])
        self.true_t_enu = np.zeros(3)


class GPSData:
    """GPS 数据结构，保存经纬高、速度、本地 NED 及真值"""

    def __init__(self):
        self.position_lla = np.zeros(3)
        self.velocity = np.zeros(3)
        self.local_position_ned = np.zeros(3)
        self.true_velocity = np.zeros(3)
        self.true_position_lla = np.zeros(3)
        self.time = 0.0
