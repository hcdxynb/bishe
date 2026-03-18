import glob
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# -----------------------------
# 1. 读取 KITTI OXTS 文件夹中每帧的 txt 数据
# -----------------------------
column_names = [
    'lat', 'lon', 'alt', 'roll', 'pitch', 'yaw',
    'vn', 've', 'vf', 'vl', 'vu',
    'ax', 'ay', 'az', 'af', 'al', 'au',
    'wx', 'wy', 'wz', 'wf', 'wl', 'wu',
    'pos_accuracy', 'vel_accuracy',
    'navstat', 'numsats', 'posmode', 'velmode', 'orimode'
]

oxts_dir = "data"  # 文件夹路径，包含多个 txt 文件
file_paths = sorted(glob.glob(os.path.join(oxts_dir, "*.txt")))
rows = []
for fp in file_paths:
    row = np.loadtxt(fp)
    # 每个 txt 一帧，如果文件只包含一行也有效
    if row.ndim == 1:
        rows.append(row)
    # else:
        # 可能某些 txt 文件是多行，逐行追加
        # rows.extend(row)

if len(rows) == 0:
    raise ValueError(f"未找到 oxts 数据文件：{oxts_dir}/*.txt")

data = np.vstack(rows)

df = pd.DataFrame(data, columns=column_names) # 转成 DataFrame 方便处理


# -----------------------------
# 2. 添加时间戳（假设已知起始时间 10Hz，使用整数起点便于后续处理）
# -----------------------------
start_time = 0.0  # 从0秒开始计时，不需要具体实际时刻
imu_hz = 10

dt_imu = 1.0 / imu_hz
# 时间轴以秒数形式存储，便于数学运算和滤波输入
df['time'] = [start_time + i * dt_imu for i in range(len(df))]


# -----------------------------
# 3. 提取用于 ESKF 的 IMU 和 GPS 数据
# -----------------------------
imu_hz = 10
gps_hz = 1
gps_interval = 1.0
next_gps_time = df['time'].iloc[0]

imu_data_list = []
gps_data_list = []

for idx, row in df.iterrows():
    # --- IMU 10Hz ---
    imu_data_list.append({
        'time': row['time'],
        'af': row['af'],
        'al': row['al'],
        'au': row['au'],
        'wf': row['wf'],
        'wl': row['wl'],
        'wu': row['wu']
    })

    # --- GPS 1Hz ---
    if row['time'] >= next_gps_time:
        gps_data_list.append({
            'time': row['time'],
            'lat': row['lat'],
            'lon': row['lon'],
            'alt': row['alt'],
            'vf': row['vf'],
            'vl': row['vl'],
            'vu': row['vu']
        })
        next_gps_time += gps_interval

# -----------------------------
# 4. 转成 DataFrame
# -----------------------------
imu_df = pd.DataFrame(imu_data_list)
gps_df = pd.DataFrame(gps_data_list)

# -----------------------------
# 5. 保存 CSV（可选）
# -----------------------------
# imu_df.to_csv("eskf_imu_10hz.csv", index=False)
# gps_df.to_csv("eskf_gps_1hz.csv", index=False)

# -----------------------------
# 6. 查看数据
# -----------------------------
print("IMU 10Hz (ESKF) 示例：")
print(imu_df.head(100))

print("\nGPS 1Hz (ESKF) 示例：")
print(gps_df.head(10))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       