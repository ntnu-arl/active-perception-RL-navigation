import warp as wp
from src.sensors.base_sensor import BaseSensor

from aerial_gym.utils.math import (
    quat_from_euler_xyz,
    quat_mul,
    tf_apply,
    torch_rand_float_tensor,
    quat_from_euler_xyz_tensor,
)

import torch

from src.sensors.warp.warp_cam import WarpCam
from aerial_gym.sensors.warp.warp_stereo_cam import WarpStereoCam
from aerial_gym.sensors.warp.warp_lidar import WarpLidar
from aerial_gym.sensors.warp.warp_normal_faceID_cam import WarpNormalFaceIDCam
from aerial_gym.sensors.warp.warp_normal_faceID_lidar import WarpNormalFaceIDLidar

from aerial_gym.utils.logging import CustomLogger, logging

logger = CustomLogger("WarpSensor")
logger.setLoggerLevel(logging.INFO)


class WarpSensor(BaseSensor):
    def __init__(self, sensor_config, num_envs, mesh_id_list, device):
        super().__init__(sensor_config=sensor_config, num_envs=num_envs, device=device)
        self.mesh_id_list = mesh_id_list
        self.device = device
        self.num_sensors = self.cfg.num_sensors

        self.mesh_ids_array = wp.array(mesh_id_list, dtype=wp.uint64)

        if self.cfg.sensor_type == "lidar":
            self.sensor = WarpLidar(
                num_envs=self.num_envs,
                mesh_ids_array=self.mesh_ids_array,
                config=self.cfg,
            )
            logger.info("Lidar sensor initialized")
            logger.debug(f"Sensor config: {self.cfg.__dict__}")

        elif self.cfg.sensor_type == "camera":
            self.sensor = WarpCam(
                num_envs=self.num_envs,
                mesh_ids_array=self.mesh_ids_array,
                config=self.cfg,
            )
            logger.info("Camera sensor initialized")
            logger.debug(f"Sensor config: {self.cfg.__dict__}")
        
        elif self.cfg.sensor_type == "stereo_camera":
            self.sensor = WarpStereoCam(
                num_envs=self.num_envs,
                mesh_ids_array=self.mesh_ids_array,
                config=self.cfg,
            )
            logger.info("Camera sensor initialized")
            logger.debug(f"Sensor config: {self.cfg.__dict__}")

        elif self.cfg.sensor_type == "normal_faceID_lidar":
            self.sensor = WarpNormalFaceIDLidar(
                num_envs=self.num_envs,
                mesh_ids_array=self.mesh_ids_array,
                config=self.cfg,
            )
            logger.info("Normal FaceID Lidar sensor initialized")
            logger.debug(f"Sensor config: {self.cfg.__dict__}")

        elif self.cfg.sensor_type == "normal_faceID_camera":
            self.sensor = WarpNormalFaceIDCam(
                num_envs=self.num_envs,
                mesh_ids_array=self.mesh_ids_array,
                config=self.cfg,
            )
            logger.info("Normal FaceID Camera sensor initialized")
            logger.debug(f"Sensor config: {self.cfg.__dict__}")

        else:
            raise NotImplementedError

    def init_tensors(self, global_tensor_dict):
        super().init_tensors(global_tensor_dict)
        logger.debug(f"Initializing sensor tensors")
        # here a new view of robot position and orienentation is created since the robot has multiple sensors
        self.robot_position = self.robot_position.unsqueeze(1).expand(-1, self.num_sensors, -1)
        self.robot_orientation = self.robot_orientation.unsqueeze(1).expand(
            -1, self.num_sensors, -1
        )

        self.sensor_min_translation = torch.tensor(
            self.cfg.min_translation, device=self.device, requires_grad=False
        ).expand(self.num_envs, self.num_sensors, -1)
        self.sensor_max_translation = torch.tensor(
            self.cfg.max_translation, device=self.device, requires_grad=False
        ).expand(self.num_envs, self.num_sensors, -1)
        self.sensor_min_rotation = torch.deg2rad(
            torch.tensor(self.cfg.min_euler_rotation_deg, device=self.device, requires_grad=False)
        ).expand(self.num_envs, self.num_sensors, -1)
        self.sensor_max_rotation = torch.deg2rad(
            torch.tensor(self.cfg.max_euler_rotation_deg, device=self.device, requires_grad=False)
        ).expand(self.num_envs, self.num_sensors, -1)
        euler_sensor_frame_rot = self.cfg.euler_frame_rot_deg
        sensor_frame_rot_rad = torch.deg2rad(
            torch.tensor(euler_sensor_frame_rot, device=self.device, requires_grad=False)
        )
        sensor_quat = quat_from_euler_xyz_tensor(sensor_frame_rot_rad)
        self.sensor_data_frame_quat = sensor_quat.expand(self.num_envs, self.num_sensors, -1)

        self.sensor_local_position = torch.zeros(
            (self.num_envs, self.num_sensors, 3),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_local_orientation = torch.zeros(
            (self.num_envs, self.num_sensors, 4),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_local_orientation[..., 3] = 1.0
        self.sensor_local_orientation_euler = torch.zeros(
            (self.num_envs, self.num_sensors, 3),
            device=self.device,
            requires_grad=False,
        )
        mean_euler_rotation = (self.sensor_min_rotation + self.sensor_max_rotation) / 2.0
        self.sensor_local_orientation[:] = quat_from_euler_xyz(
            mean_euler_rotation[..., 0],
            mean_euler_rotation[..., 1],
            mean_euler_rotation[..., 2],
        )
        self.sensor_position = torch.zeros(
            (self.num_envs, self.num_sensors, 3),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_orientation = torch.zeros(
            (self.num_envs, self.num_sensors, 4),
            device=self.device,
            requires_grad=False,
        )
        self.sensor_orientation[..., 3] = 1.0
        self.sensor.set_pose_tensor(
            positions=self.sensor_position, orientations=self.sensor_orientation
        )
        self.sensor.set_image_tensors(
            pixels=self.pixels, segmentation_pixels=self.segmentation_pixels, occupancy_map=self.occupancy_map
        )
        self.get_camera_local_orientation(orientation=self.camera_orientation)
        self.tau = 0.2 * torch.ones((self.num_envs), device=self.device)
        self.reset()

        logger.debug(f"[DONE] Initializing sensor tensors")

    def reset(self):
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_ids)

    def reset_idx(self, env_ids):
        self.tau[env_ids] = 0.2 + torch.rand_like(self.tau[env_ids]) * 0.1  # tau in [0.2, 0.4]
        if self.cfg.randomize_placement == True:
            # sample local position from min and max translations
            self.sensor_local_position[env_ids] = torch_rand_float_tensor(
                self.sensor_min_translation[env_ids],
                self.sensor_max_translation[env_ids],
            )
            # sample local orientation from min and max rotations
            local_euler_rotation = torch_rand_float_tensor(
                self.sensor_min_rotation[env_ids], self.sensor_max_rotation[env_ids]
            )
            self.sensor_local_orientation[env_ids] = quat_from_euler_xyz(
                local_euler_rotation[..., 0],
                local_euler_rotation[..., 1],
                local_euler_rotation[..., 2],
            )
            self.sensor_local_orientation_euler[env_ids] = local_euler_rotation
            self.sensor.reset_maps(env_ids)
        else:
            # Do nothing
            pass
        return

    def initialize_sensor(self):
        self.sensor.capture()

    def update(self, action):
        # transform local position and orientation to world frame before performing ray_casting
        # tf_apply(self.root_quats, self.root_positions, self.sensor_local_pos)
        self.sensor_position[:] = tf_apply(
            self.robot_orientation, self.robot_position, self.sensor_local_position
        )
        self.servo_speed = 1.0 / self.tau  # T = 0.2s, so servo_speed = 5.0
        self.dt = 0.1  # 10Hz
        # --- First-order servo simulation ---
        target_pitch = action[..., 4]
        actual_pitch = self.sensor_local_orientation_euler[..., 0, 1]
        error_pitch = target_pitch - actual_pitch
        servo_instantaneous_speed = self.servo_speed * error_pitch
        self.max_servo_speed = torch.pi/1.5  # 90 degrees per second
        servo_clamped_speed = torch.clamp(servo_instantaneous_speed, -self.max_servo_speed, self.max_servo_speed)
        actual_pitch = actual_pitch + servo_clamped_speed * self.dt
        actual_pitch = torch.clamp(actual_pitch, -torch.pi/3, torch.pi/3)
        self.sensor_local_orientation_euler[..., 0, 1] = actual_pitch

        target_yaw = action[..., 5]
        actual_yaw = self.sensor_local_orientation_euler[..., 0, 2]
        error_yaw = target_yaw - actual_yaw
        servo_instantaneous_speed = self.servo_speed * error_yaw
        self.max_servo_speed = torch.pi/1.5  # 90 degrees per second
        servo_clamped_speed = torch.clamp(servo_instantaneous_speed, -self.max_servo_speed, self.max_servo_speed)
        actual_yaw = actual_yaw + servo_clamped_speed * self.dt
        actual_yaw = torch.clamp(actual_yaw, -torch.pi/4, torch.pi/4)
        self.sensor_local_orientation_euler[..., 0, 2] = actual_yaw
        # # --- Velocity control simulation ---
        # # action[..., 4] and action[..., 5] are angular velocities (rad/s)
        # self.dt = 0.1  # 10Hz
        # pitch_vel = action[..., 4]
        # yaw_vel = action[..., 5]
        # actual_pitch = self.sensor_local_orientation_euler[..., 0, 1] + pitch_vel * self.dt
        # actual_pitch = torch.clamp(actual_pitch, -torch.pi/2, torch.pi/2)
        # self.sensor_local_orientation_euler[..., 0, 1] = actual_pitch

        # actual_yaw = self.sensor_local_orientation_euler[..., 0, 2] + yaw_vel * self.dt
        # actual_yaw = torch.clamp(actual_yaw, -torch.pi/2, torch.pi/2)
        # self.sensor_local_orientation_euler[..., 0, 2] = actual_yaw

        self.sensor_local_orientation = quat_from_euler_xyz(self.sensor_local_orientation_euler[..., 0],
                                                            self.sensor_local_orientation_euler[..., 1],
                                                            self.sensor_local_orientation_euler[..., 2])
        # quat_mul(self.root_quats, quat_mul(self.sensor_local_quat, self.correct_sensor_frame_quat))
        self.sensor_orientation[:] = quat_mul(
            self.robot_orientation,
            quat_mul(self.sensor_local_orientation, self.sensor_data_frame_quat),
        )

        # logger.debug(
        #     f"Sensor position: {self.sensor_position[0]}, Sensor orientation: {self.sensor_orientation[0]}"
        # )

        # logger.debug("Capturing sensor data")
        self.sensor.capture()
        # logger.debug("[DONE] Capturing sensor data")

        self.apply_noise()
        if self.cfg.sensor_type in ["camera", "lidar", "stereo_camera"]:
            self.apply_range_limits()
            self.normalize_observation()

    def apply_range_limits(self):
        if self.cfg.return_pointcloud == True:
            # if pointcloud is in the world frame, the pointcloud range will not be normalized
            if self.cfg.pointcloud_in_world_frame == False:
                # logger.debug("Pointcoud is not in world frame")
                self.pixels[
                    self.pixels.norm(dim=4, keepdim=True).expand(-1, -1, -1, -1, 3)
                    > self.cfg.max_range
                ] = self.cfg.far_out_of_range_value
                self.pixels[
                    self.pixels.norm(dim=4, keepdim=True).expand(-1, -1, -1, -1, 3)
                    < self.cfg.min_range
                ] = self.cfg.near_out_of_range_value
                # logger.debug("[DONE] Clipping pointcloud values to sensor range")
        else:
            # logger.debug("Pointcloud is in world frame")
            self.pixels[self.pixels > self.cfg.max_range] = self.cfg.far_out_of_range_value
            self.pixels[self.pixels < self.cfg.min_range] = self.cfg.near_out_of_range_value
            # logger.debug("[DONE] Clipping pointcloud values to sensor range")

    def normalize_observation(self):
        if self.cfg.normalize_range and self.cfg.pointcloud_in_world_frame == False:
            # logger.debug("Normalizing pointcloud values")
            self.pixels[:] = self.pixels / self.cfg.max_range
        # if self.cfg.pointcloud_in_world_frame == True:
        #     logger.debug("Pointcloud is in world frame. not normalizing")

    def apply_noise(self):
        if self.cfg.sensor_noise.enable_sensor_noise == True:
            # logger.debug("Applying sensor noise")
            sensor_noise_params = self.cfg.sensor_noise
            std_a = sensor_noise_params.std_a
            std_b = sensor_noise_params.std_b
            std_c = sensor_noise_params.std_c
            mean_offset = sensor_noise_params.mean_offset
            std_val = std_a * self.pixels**2 + std_b * self.pixels + std_c
            self.pixels[:] = torch.normal(
                mean= (self.pixels - mean_offset),
                std=std_val
            )
            self.pixels[
                torch.bernoulli(
                    torch.ones_like(self.pixels) * self.cfg.sensor_noise.pixel_dropout_prob
                )
                > 0
            ] = self.cfg.near_out_of_range_value

    def get_observation(self):
        return self.pixels, self.segmentation_pixels

    def get_camera_local_orientation(self, orientation):
        self.sensor_local_orientation_euler = orientation