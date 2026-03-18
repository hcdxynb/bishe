import os
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation as R
from .config import ConfigParameters
from .gps_tool import GPSTool
from .imu_tool import IMUTool
from .eskf import ErrorStateKalmanFilter


class ESKFFlow:
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
        if os.path.basename(raw_path).lower() != 'raw_data':
            raw_path = os.path.join(raw_path, 'raw_data')

        self.imu_data_buff = deque(self.imu_tool.read_imu_data(raw_path))
        self.gps_data_buff = deque(self.gps_tool.read_gps_data(raw_path))

    def valid_gps_and_imu(self):
        if not self.imu_data_buff or not self.gps_data_buff:
            return False

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

    def save_tum_pose(self, f, pose, timestamp, position=None):
        if position is None:
            t = pose[0:3, 3]
            q = R.from_matrix(pose[0:3, 0:3]).as_quat()
        else:
            t = position
            q = np.array([0.0, 0.0, 0.0, 1.0])

        f.write(f"{timestamp:.10f} {t[0]:.10f} {t[1]:.10f} {t[2]:.10f} {q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f}\n")

    def run(self):
        self.read_data()

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
