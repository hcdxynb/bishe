import yaml


class ConfigParameters:
    """配置参数类，支持从 YAML 文件读取滤波参数"""

    def __init__(self):
        self.earth_rotation_speed = 0.0
        self.earth_gravity = 9.8
        self.ref_longitude = 0.0
        self.ref_latitude = 0.0
        self.ref_altitude = 0.0

        self.position_error_prior_std = 1e-5
        self.velocity_error_prior_std = 1e-5
        self.rotation_error_prior_std = 1e-5
        self.accelerometer_bias_error_prior_std = 1e-5
        self.gyro_bias_error_prior_std = 1e-5
        self.gyro_noise_std = 1e-2
        self.accelerometer_noise_std = 1e-1

        self.gps_position_x_std = 5.0
        self.gps_position_y_std = 5.0
        self.gps_position_z_std = 8.0

        self.only_prediction = False
        self.use_earth_model = True

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        self.earth_rotation_speed = float(cfg.get('earth_rotation_speed', self.earth_rotation_speed))
        self.earth_gravity = float(cfg.get('earth_gravity', self.earth_gravity))
        self.ref_longitude = float(cfg.get('ref_longitude', self.ref_longitude))
        self.ref_latitude = float(cfg.get('ref_latitude', self.ref_latitude))
        self.ref_altitude = float(cfg.get('ref_altitude', self.ref_altitude))

        self.position_error_prior_std = float(cfg.get('position_error_prior_std', self.position_error_prior_std))
        self.velocity_error_prior_std = float(cfg.get('velocity_error_prior_std', self.velocity_error_prior_std))
        self.rotation_error_prior_std = float(cfg.get('rotation_error_prior_std', self.rotation_error_prior_std))
        self.accelerometer_bias_error_prior_std = float(cfg.get('accelerometer_bias_error_prior_std', self.accelerometer_bias_error_prior_std))
        self.gyro_bias_error_prior_std = float(cfg.get('gyro_bias_error_prior_std', self.gyro_bias_error_prior_std))

        self.gyro_noise_std = float(cfg.get('gyro_noise_std', self.gyro_noise_std))
        self.accelerometer_noise_std = float(cfg.get('accelerometer_noise_std', self.accelerometer_noise_std))

        self.gps_position_x_std = float(cfg.get('gps_position_x_std', self.gps_position_x_std))
        self.gps_position_y_std = float(cfg.get('gps_position_y_std', self.gps_position_y_std))
        self.gps_position_z_std = float(cfg.get('gps_position_z_std', self.gps_position_z_std))

        self.only_prediction = bool(cfg.get('only_prediction', self.only_prediction))
        self.use_earth_model = bool(cfg.get('use_earth_model', self.use_earth_model))
