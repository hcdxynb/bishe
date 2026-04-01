import argparse # 用于解析命令行参数
from .flow import ESKFFlow # 从当前包中导入 ESKFFlow 类


def main():
    parser = argparse.ArgumentParser(description='ESKF GPS-IMU fusion python package')
    parser.add_argument('config', help='config file path')
    parser.add_argument('data', help='data folder path')
    args = parser.parse_args()

    flow = ESKFFlow(args.config, args.data) # 参数导入并创建 ESKFFlow 实例
    flow.run() # 运行 ESKFFlow 的主流程


if __name__ == '__main__':
    main()
