import os
import numpy as np
import matplotlib.pyplot as plt


def load_kitti_poses(poses_file):
    """读取 KITTI odometry poses.txt，返回 N x 4x4 矩阵"""
    with open(poses_file, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    poses = []
    for line in lines:
        if not line.strip():
            continue
        vals = [float(x) for x in line.strip().split()]
        if len(vals) != 12:
            raise ValueError('poses.txt 每行必须是 12 个浮点数')
        T = np.eye(4)
        T[:3, :4] = np.array(vals).reshape(3, 4)
        poses.append(T)
    return np.stack(poses, axis=0)


def load_kitti_oxts(oxts_file):
    """读取 KITTI oxts data (通常有 6+ 术语)，返回 dict 列表"""
    cols = [
        'lat', 'lon', 'alt', 'roll', 'pitch', 'yaw',
        'vn', 've', 'vf', 'vl', 'vu', 'ax', 'ay', 'az',
        'af', 'al', 'au', 'wx', 'wy', 'wz', 'wf', 'wl', 'wu'
    ]

    data = []
    with open(oxts_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            vals = [float(x) for x in line.strip().split()]
            row = {col: (vals[i] if i < len(vals) else np.nan) for i, col in enumerate(cols)}
            data.append(row)
    return data


def plot_trajectory(poses, label='trajectory', color='b'):
    """可视化pose轨迹，默认绘制XY平面"""
    t = poses[:, :3, 3]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.plot(t[:, 0], t[:, 1], '-', color=color, label=label)
    ax.scatter(t[0, 0], t[0, 1], c='g', marker='o', label='start')
    ax.scatter(t[-1, 0], t[-1, 1], c='r', marker='x', label='end')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('KITTI Trajectory')
    ax.axis('equal')
    ax.legend()
    ax.grid(True)
    plt.show()


def write_csv_from_oxts(oxts_data, output_dir):
    """从 oxts 数据写入一个简单 csv 供 ESKF 使用（可选）"""
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, 'kitti_oxts.csv')
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write('lat,lon,alt,vn,ve,vf,ax,ay,az,wx,wy,wz\n')
        for d in oxts_data:
            f.write(f"{d['lat']},{d['lon']},{d['alt']},{d['vn']},{d['ve']},{d['vf']},{d['ax']},{d['ay']},{d['az']},{d['wx']},{d['wy']},{d['wz']}\n")
    print('写入:', csv_file)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='KITTI 轨迹可视化')
    parser.add_argument('--kitti_path', required=True, help='KITTI odometry 数据集根目录，例如 .../kitti/odometry/dataset/sequence/00')
    parser.add_argument('--plot', action='store_true', help='显示轨迹图')
    parser.add_argument('--dump_oxts', action='store_true', help='输出 OXTS csv 文件')
    args = parser.parse_args()

    seq_path = args.kitti_path
    poses_file = os.path.join(seq_path, 'poses.txt')
    oxts_file = os.path.join(seq_path, 'oxts/data.txt')

    if not os.path.exists(poses_file):
        raise FileNotFoundError('找不到 poses.txt: ' + poses_file)

    poses = load_kitti_poses(poses_file)
    print('加载pose数量：', poses.shape[0])

    if args.plot:
        plot_trajectory(poses, label=os.path.basename(seq_path), color='b')

    if os.path.exists(oxts_file):
        oxts_data = load_kitti_oxts(oxts_file)
        print('加载 oxts 行数：', len(oxts_data))

        if args.dump_oxts:
            write_csv_from_oxts(oxts_data, os.path.join(seq_path, 'converted'))

        if args.plot:
            # 使用 oxts 速度轨迹示例（VN, VE）, 展示里程计速度方向的走向
            vn = np.array([d['vn'] for d in oxts_data])
            ve = np.array([d['ve'] for d in oxts_data])
            plt.figure(figsize=(6, 6))
            plt.quiver(np.arange(len(vn)), np.zeros_like(vn), vn, ve, scale=10)
            plt.title('OXTS speed vectors (video sample)')
            plt.xlabel('frame index')
            plt.ylabel('vector')
            plt.grid(True)
            plt.show()
    else:
        print('未找到 oxts 数据，跳过 oxts 可视化')

    print('完成')


if __name__ == '__main__':
    main()
