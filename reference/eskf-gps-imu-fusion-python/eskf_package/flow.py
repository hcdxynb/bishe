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
        self.config = ConfigParameters() # 创建配置参数实例
        self.config.load(config_file_path) # 从配置文件加载参数

        # 数据来源待修改
        self.gps_tool = GPSTool(self.config.ref_longitude, self.config.ref_latitude, self.config.ref_altitude) # 创建 GPS 工具实例，传入参考点坐标
        self.imu_tool = IMUTool()


        self.eskf = ErrorStateKalmanFilter(self.config) # 创建 ESKF 实例，传入配置参数
        self.data_file_path = data_file_path # 数据文件路径
        self.imu_data_buff = deque() # IMU 数据缓冲区，使用双端队列实现
        self.gps_data_buff = deque() # GPS 数据缓冲区，使用双端队列实现

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
        self.read_data() # 读取 IMU 和 GPS 数据到缓冲区

        while self.imu_data_buff and self.gps_data_buff: # 循环处理数据，直到任一缓冲区为空
            valid = self.valid_gps_and_imu()
            if not valid:
                continue
            imu_data, gps_data = valid
            self.eskf.init(gps_data, imu_data) # 使用第一条 GPS 和 IMU 数据初始化 ESKF
            break

        gc_path = os.path.join(self.data_file_path, 'gt.txt')
        fused_path = os.path.join(self.data_file_path, 'fused.txt')
        meas_path = os.path.join(self.data_file_path, 'gps_measurement.txt')

        with open(gc_path, 'w', encoding='utf-8') as f_gt, \
             open(fused_path, 'w', encoding='utf-8') as f_fused, \
             open(meas_path, 'w', encoding='utf-8') as f_meas:
            while self.imu_data_buff and self.gps_data_buff: # 循环处理数据，直到任一缓冲区为空
                imu_data = self.imu_data_buff[0] 
                gps_data = self.gps_data_buff[0]

                self.eskf.estimate(imu_data) # 使用当前 IMU 数据进行 ESKF 预测
                self.imu_data_buff.popleft() # 处理完当前 IMU 数据后将其从缓冲区移除

                if imu_data.time >= gps_data.time:
                    if not self.config.only_prediction:
                        self.eskf.correct(gps_data) # 使用当前 GPS 数据进行 ESKF 校正

                    self.save_tum_pose(f_fused, self.eskf.get_pose(), imu_data.time) # 保存融合后的位姿到文件
                    self.save_tum_pose(f_meas, np.eye(4), gps_data.time, position=gps_data.local_position_ned) # 保存 GPS 测量位置到文件，姿态部分使用单位矩阵表示
                    gt_q = np.eye(4)
                    gt_q[0:3, 3] = imu_data.true_t_enu # 从 IMU 数据中获取真实位置作为地面真值
                    self.save_tum_pose(f_gt, gt_q, imu_data.time)

                    self.gps_data_buff.popleft() # 处理完当前 GPS 数据后将其从缓冲区移除
