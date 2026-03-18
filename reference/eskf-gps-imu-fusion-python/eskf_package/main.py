import argparse
from .flow import ESKFFlow


def main():
    parser = argparse.ArgumentParser(description='ESKF GPS-IMU fusion python package')
    parser.add_argument('config', help='config file path')
    parser.add_argument('data', help='data folder path')
    args = parser.parse_args()

    flow = ESKFFlow(args.config, args.data)
    flow.run()


if __name__ == '__main__':
    main()
