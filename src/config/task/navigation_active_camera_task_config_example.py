import torch
from aerial_gym import AERIAL_GYM_DIRECTORY


class task_config_example:
    seed = -1
    sim_name = "base_sim"
    env_name = "env_active_camera_with_obstacles"
    robot_name = "active_camera_quadrotor"
    controller_name = "velocity_control"
    args = {}
    num_envs = 16
    use_warp = True
    headless = False
    device = "cuda:0"
    observation_space_dim = 13 + 6 + 2 + 64 # root_state + action_dim + camera_orientation + latent_dims
    privileged_observation_space_dim = 0
    action_space_dim = 4 + 1 + 1  # x, y, z, yawrate, camera_pitch, camera_yaw
    episode_len_steps = 150  # real physics time for simulation is this value multiplied by sim.dt

    return_state_before_reset = (
        False  # False as usually state is returned for next episode after reset
    )
    # user can set the above to true if they so desire

    target_min_ratio = [0.90, 0.1, 0.1]  # target ratio w.r.t environment bounds in x,y,z
    target_max_ratio = [0.94, 0.90, 0.90]  # target ratio w.r.t environment bounds in x,y,z

    reward_parameters = {
        "pos_reward_magnitude": 5.0,
        "pos_reward_exponent": 1.0,
        "very_close_to_goal_reward_magnitude": 5.0,
        "very_close_to_goal_reward_exponent": 5.0,
        "getting_closer_reward_multiplier": 10.0,
        "x_action_diff_penalty_magnitude": 0.25,
        "x_action_diff_penalty_exponent": 5.0,
        "z_action_diff_penalty_magnitude": 0.25,
        "z_action_diff_penalty_exponent": 5.0,
        "yawrate_action_diff_penalty_magnitude": 0.25,
        "yawrate_action_diff_penalty_exponent": 5.0,
        "camera_action_diff_penalty_magnitude": 0.25,
        "camera_action_diff_penalty_exponent": 5.0,
        "collision_penalty": -2.0,
    }
    
    sub_voxelmap_size = 21
    sub_voxelmap_cell_size = 0.1
    min_value_x = -6.0
    max_value_x = 6.0
    min_value_y = -4.0
    max_value_y = 4.0
    min_value_z = -3.0
    max_value_z = 3.0
    observations_map_dim = [sub_voxelmap_size, sub_voxelmap_size, sub_voxelmap_size]

    class vae_config:
        use_vae = True
        latent_dims = 64
        model_file = (
            AERIAL_GYM_DIRECTORY
            + "/aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
        )
        model_folder = AERIAL_GYM_DIRECTORY
        image_res = (270, 480)
        interpolation_mode = "nearest"
        return_sampled_latent = True

    class curriculum:
        min_level = 49
        max_level = 50
        check_after_log_instances = 2048
        increase_step = 2
        decrease_step = 1
        success_rate_for_increase = 0.7
        success_rate_for_decrease = 0.6

        def update_curriculim_level(self, success_rate, current_level):
            if success_rate > self.success_rate_for_increase:
                return min(current_level + self.increase_step, self.max_level)
            elif success_rate < self.success_rate_for_decrease:
                return max(current_level - self.decrease_step, self.min_level)
            return current_level


    def action_transformation_function(action, previous_action=None):
        max_speed = 1.0  # [m/s]
        max_yawrate = torch.pi / 3  # [rad/s]
        
        # if previous_action is not None:
        #     clamped_action = torch.clamp(action, -0.1, 0.1)
        #     # Apply smoothing to the action
        #     clamped_action += previous_action
        clamped_action = torch.clamp(action, -1.0, 1.0)

        clamped_action[:, 0] *= max_speed  # x velocity in range [0, 2]

        clamped_action[:, 1] *= max_speed  # y velocity in range [-1, 1]

        clamped_action[:, 2] *= max_speed  # z velocity in range [-1, 1]
        
        velocity_magnitude = torch.norm(clamped_action[:, 0:3], dim=1, keepdim=True)
        scaling_factor = torch.clamp(max_speed / (velocity_magnitude + 1e-6), max=1.0)
        clamped_action[:, 0:3] *= scaling_factor

        clamped_action[:, 3] *= 1.0 # yaw rate

        clamped_action[:, 4] *= torch.pi / 3.0  # camera pitch

        clamped_action[:, 5] *= torch.pi / 4.0  # camera yaw
        
        return clamped_action
