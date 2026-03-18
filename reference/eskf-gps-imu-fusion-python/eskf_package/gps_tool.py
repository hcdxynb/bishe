import os
import numpy as np
from .data_model import GPSData


class GPSTool:
    """GPS 坐标和文件读取模块"""

    def __init__(self, lon, lat, alt):
        self.ref_lon = lon
        self.ref_lat = lat
        self.ref_alt = alt

        self.a = 6378137.0
        self.f = 1.0 / 298.257223563
        self.e2 = self.f * (2 - self.f)

        self.ref_ecef = self.lla_to_ecef(np.deg2rad(self.ref_lat), np.deg2rad(self.ref_lon), self.ref_alt)

    def lla_to_ecef(self, lat_rad, lon_rad, alt):
        N = self.a / np.sqrt(1 - self.e2 * np.sin(lat_rad) ** 2)
        x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - self.e2) + alt) * np.sin(lat_rad)
        return np.array([x, y, z])

    def ecef_to_enu(self, ecef):
        dx = ecef - self.ref_ecef
        lat = np.deg2rad(self.ref_lat)
        lon = np.deg2rad(self.ref_lon)

        t = np.array([
            [-np.sin(lon), np.cos(lon), 0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        ])
        return t.dot(dx)

    def lla_to_local_ned_vec(self, lla):
        lat_rad = np.deg2rad(lla[0])
        lon_rad = np.deg2rad(lla[1])
        alt = lla[2]

        ecef = self.lla_to_ecef(lat_rad, lon_rad, alt)
        enu = self.ecef_to_enu(ecef)
        return np.array([enu[1], enu[0], -enu[2]])

    def read_gps_data(self, path, skip_rows=1):
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
        return gps_data
