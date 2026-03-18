import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from .data_model import IMUData


class IMUTool:
    DEG2RAD = np.pi / 180.0

    def read_imu_data(self, path, skip_rows=1):
        accel_path = os.path.join(path, 'accel-0.csv')
        ref_accel_path = os.path.join(path, 'ref_accel.csv')
        gyro_path = os.path.join(path, 'gyro-0.csv')
        ref_gyro_path = os.path.join(path, 'ref_gyro.csv')
        time_path = os.path.join(path, 'time.csv')
        ref_att_quat_path = os.path.join(path, 'ref_att_quat.csv')
        ref_pos_path = os.path.join(path, 'ref_pos.csv')

        imu_data = []
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
                d.true_q_enu = R.from_quat([quat[1], quat[2], quat[3], quat[0]])

                imu_data.append(d)
        return imu_data
