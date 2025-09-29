from aerial_gym.task.base_task import BaseTask
from src.sim.sim_builder import SimBuilder
import torch, torchvision
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("navigation_active_camera_task")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class NavigationActiveCameraTask(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        # overwrite the params if user has provided them
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp
        super().__init__(task_config)
        self.device = self.task_config.device
        # set the each of the elements of reward parameter to a torch tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        logger.info("Building environment for navigation task.")
        logger.info(
            "Sim Name: {}, Env Name: {}, Robot Name: {}, Controller Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        self.target_min_ratio = torch.tensor(
            self.task_config.target_min_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs, -1)
        self.target_max_ratio = torch.tensor(
            self.task_config.target_max_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs, -1)

        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0

        self.pos_error_vehicle_frame_prev = torch.zeros_like(self.target_position)
        self.pos_error_vehicle_frame = torch.zeros_like(self.target_position)
        # occupancy grid variables
        indices_0 = torch.arange(self.task_config.num_envs, device=self.device)
        indices_1 = torch.arange(self.task_config.sub_voxelmap_size, device=self.device)
        center = torch.floor(self.task_config.sub_voxelmap_size * 0.5 * torch.ones((1, 3), device=self.device))
        # Generate all combinations of indices using cartesian product
        index_base = torch.cartesian_prod(indices_0, indices_1, indices_1, indices_1)[:self.task_config.sub_voxelmap_size**3, 1:4].unsqueeze(0).expand(self.task_config.num_envs, -1, -1)
        # Define variables for local map extraction
        self.index_base = index_base.clamp(0, self.task_config.sub_voxelmap_size - 1)
        self.position_base = ((torch.cartesian_prod(indices_0, indices_1, indices_1, indices_1)[
                              :self.task_config.sub_voxelmap_size**3, 1:4] - center) * self.task_config.sub_voxelmap_cell_size).unsqueeze(0).expand(self.task_config.num_envs, -1, -1)
        if self.task_config.vae_config.use_vae:
            self.vae_model = VAEImageEncoder(config=self.task_config.vae_config, device=self.device)
            self.image_latents = torch.zeros(
                (self.sim_env.num_envs, self.task_config.vae_config.latent_dims),
                device=self.device,
                requires_grad=False,
            )
        else:
            self.vae_model = lambda x: x

        # Get the dictionary once from the environment and use it to get the observations later.
        # This is to avoid constant retuning of data back anf forth across functions as the tensors update and can be read in-place.
        self.obs_dict = self.sim_env.get_obs()
        if "curriculum_level" not in self.obs_dict.keys():
            self.curriculum_level = self.task_config.curriculum.min_level
            self.obs_dict["curriculum_level"] = self.curriculum_level
        else:
            self.curriculum_level = self.obs_dict["curriculum_level"]
        self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
        self.curriculum_progress_fraction = (
            self.curriculum_level - self.task_config.curriculum.min_level
        ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)
        self.obs_dict["robot_prev_actions"][:] = 0.0

        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-np.Inf,
                    high=np.Inf,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                ),
                "observations_map": Box(
                    low=-np.Inf,
                    high=np.Inf,
                    shape=(1, self.task_config.observations_map_dim[0], self.task_config.observations_map_dim[1], self.task_config.observations_map_dim[2]),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.task_config.action_space_dim,), dtype=np.float32)
        self.action_transformation_function = self.task_config.action_transformation_function

        self.num_envs = self.sim_env.num_envs
        self.old_visited_voxles = torch.zeros(self.num_envs, device=self.device)
        # Currently only the "observations" are sent to the actor and critic.
        # The "priviliged_obs" are not handled so far in sample-factory

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "observations_map": torch.zeros(
                (self.sim_env.num_envs, 1, self.task_config.observations_map_dim[0], self.task_config.observations_map_dim[1], self.task_config.observations_map_dim[2]),
                device=self.device,
                requires_grad=False,
            ),
        }

        self.num_task_steps = 0

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.reset_idx(torch.arange(self.sim_env.num_envs))
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        target_ratio = torch_rand_float_tensor(self.target_min_ratio, self.target_max_ratio)
        self.target_position[env_ids] = torch_interpolate_ratio(
            min=self.obs_dict["env_bounds_min"][env_ids],
            max=self.obs_dict["env_bounds_max"][env_ids],
            ratio=target_ratio[env_ids],
        )
        # logger.warning(f"reset envs: {env_ids}")
        self.infos = {}
        self.pos_error_vehicle_frame_prev[env_ids, ...] = 1000.0
        self.pos_error_vehicle_frame[env_ids, ...] = 0.0
        self.obs_dict["robot_prev_actions"][env_ids, ...] = 0.0
        self.old_visited_voxles[env_ids] = 0.0
        return

    def render(self):
        return self.sim_env.render()

    def logging_sanity_check(self, infos):
        successes = infos["successes"]
        crashes = infos["crashes"]
        timeouts = infos["timeouts"]
        time_at_crash = torch.where(
            crashes > 0,
            self.sim_env.sim_steps,
            self.task_config.episode_len_steps * torch.ones_like(self.sim_env.sim_steps),
        )
        env_list_for_toc = (time_at_crash < 5).nonzero(as_tuple=False).squeeze(-1)
        crash_envs = crashes.nonzero(as_tuple=False).squeeze(-1)
        success_envs = successes.nonzero(as_tuple=False).squeeze(-1)
        timeout_envs = timeouts.nonzero(as_tuple=False).squeeze(-1)

        if len(env_list_for_toc) > 0:
            logger.critical("Crash is happening too soon.")
            logger.critical(f"Envs crashing too soon: {env_list_for_toc}")
            logger.critical(f"Time at crash: {time_at_crash[env_list_for_toc]}")

        if torch.sum(torch.logical_and(successes, crashes)) > 0:
            logger.critical("Success and crash are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, successes))}"
            )
        if torch.sum(torch.logical_and(successes, timeouts)) > 0:
            logger.critical("Success and timeout are occuring at the same time")
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of timeouts: {torch.count_nonzero(timeouts)}, Timeout envs: {timeout_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(successes, timeouts))}"
            )
        if torch.sum(torch.logical_and(crashes, timeouts)) > 0:
            logger.critical("Crash and timeout are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of timeouts: {torch.count_nonzero(timeouts)}, Timeout envs: {timeout_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, timeouts))}"
            )
        return

    def check_and_update_curriculum_level(self, successes, crashes, timeouts):
        self.success_aggregate += torch.sum(successes)
        self.crashes_aggregate += torch.sum(crashes)
        self.timeouts_aggregate += torch.sum(timeouts)

        instances = self.success_aggregate + self.crashes_aggregate + self.timeouts_aggregate

        if instances >= self.task_config.curriculum.check_after_log_instances:
            success_rate = self.success_aggregate / instances
            crash_rate = self.crashes_aggregate / instances
            timeout_rate = self.timeouts_aggregate / instances

            if success_rate > self.task_config.curriculum.success_rate_for_increase:
                self.curriculum_level += self.task_config.curriculum.increase_step
            elif success_rate < self.task_config.curriculum.success_rate_for_decrease:
                self.curriculum_level -= self.task_config.curriculum.decrease_step

            # clamp curriculum_level
            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                self.task_config.curriculum.max_level,
            )
            self.obs_dict["curriculum_level"] = self.curriculum_level
            self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
            self.curriculum_progress_fraction = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

            logger.warning(
                f"Curriculum Level: {self.curriculum_level}, Curriculum progress fraction: {self.curriculum_progress_fraction}"
            )
            logger.warning(
                f"\nSuccess Rate: {success_rate}\nCrash Rate: {crash_rate}\nTimeout Rate: {timeout_rate}"
            )
            logger.warning(
                f"\nSuccesses: {self.success_aggregate}\nCrashes : {self.crashes_aggregate}\nTimeouts: {self.timeouts_aggregate}"
            )
            self.success_aggregate = 0
            self.crashes_aggregate = 0
            self.timeouts_aggregate = 0

    def process_image_observation(self):
        image_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
        # torchvision.utils.save_image(image_obs[0], "images/depth_image_0_"+str(self.num_task_steps)+".png", normalize=True)
        if self.task_config.vae_config.use_vae:
            self.image_latents[:] = self.vae_model.encode(image_obs)

    def step(self, actions):
        # this uses the action, gets observations
        # calculates rewards, returns tuples
        # In this case, the episodes that are terminated need to be
        # first reset, and the first observation of the new episode
        # needs to be returned.

        transformed_action = self.action_transformation_function(actions, self.obs_dict["robot_prev_actions"])
        logger.debug(f"raw_action: {actions[0]}, transformed action: {transformed_action[0]}")
        self.sim_env.step(actions=transformed_action)

        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)
        self.pos_error_vehicle_frame_prev[:] = self.pos_error_vehicle_frame.clone()
        self.obs_dict["robot_prev_actions"] = self.obs_dict["robot_actions"].clone()
        # logger.info(f"Curricluum Level: {self.curriculum_level}")

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        # successes are the sum of the environments which are to be truncated and have reached the target within a distance threshold
        successes = self.truncations * (
            torch.norm(self.target_position - self.obs_dict["robot_position"], dim=1) < 1.0
        )
        successes = torch.where(self.terminations > 0, torch.zeros_like(successes), successes)
        timeouts = torch.where(
            self.truncations > 0, torch.logical_not(successes), torch.zeros_like(successes)
        )
        timeouts = torch.where(
            self.terminations > 0, torch.zeros_like(timeouts), timeouts
        )  # timeouts are not counted if there is a crash
        # visited_voxles = torch.where(self.obs_dict["occupancy_map"] > 0.0, 1.0, 0.0).sum(dim=(1, 2, 3))
        # visited_voxles /= (self.obs_dict["occupancy_map"].shape[1] * self.obs_dict["occupancy_map"].shape[2] * self.obs_dict["occupancy_map"].shape[3])
        # print("Mean coverage: ", torch.mean(visited_voxles))
        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = self.terminations

        self.logging_sanity_check(self.infos)
        self.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )
        # rendering happens at the post-reward calculation step since the newer measurement is required to be
        # sent to the RL algorithm as an observation and it helps if the camera image is updated then
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)
        self.num_task_steps += 1
        # do stuff with the image observations here
        self.process_image_observation()
        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()
        return return_tuple

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        vec_to_tgt = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        dist_to_tgt = torch.norm(vec_to_tgt, dim=-1)
        unit_vec_to_tgt = vec_to_tgt / dist_to_tgt.unsqueeze(1)
        self.task_obs["observations"][:, 0:3] = unit_vec_to_tgt
        self.task_obs["observations"][:, 3] = dist_to_tgt
        euler_angles = ssa(self.obs_dict["robot_euler_angles"])
        perturbed_euler_angles = euler_angles + 0.1 * (torch.rand_like(euler_angles) - 0.5)
        self.task_obs["observations"][:, 4] = perturbed_euler_angles[:, 0]
        self.task_obs["observations"][:, 5] = perturbed_euler_angles[:, 1]
        self.task_obs["observations"][:, 6] = 0.0
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13:19] = self.obs_dict["robot_actions"]
        self.task_obs["observations"][:, 19:21] = self.obs_dict["camera_orientation"][:, 0, 1:3]
        if self.task_config.vae_config.use_vae:
            self.task_obs["observations"][:, 21:] = self.image_latents

        self.local_grid  = extract_centered_tensor(self.obs_dict["occupancy_map"], 
                                                   self.task_config.sub_voxelmap_size, 
                                                   self.obs_dict["robot_position"], self.obs_dict["robot_orientation"], 
                                                   self.task_config.min_value_x, self.task_config.max_value_x, 
                                                   self.task_config.min_value_y, self.task_config.max_value_y, 
                                                   self.task_config.min_value_z, self.task_config.max_value_z, 
                                                   self.index_base, self.position_base)
        self.task_obs["observations_map"] = self.local_grid.unsqueeze(1)

    def compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        target_orientation = torch.zeros_like(robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0
        self.pos_error_vehicle_frame[:] = quat_rotate_inverse(
            robot_vehicle_orientation, (target_position - robot_position)
        )
        visited_voxles = torch.where(self.obs_dict["occupancy_map"] > 0.0, 1.0, 0.0).sum(dim=(1, 2, 3))
        visited_voxles /= (self.obs_dict["occupancy_map"].shape[1] * self.obs_dict["occupancy_map"].shape[2] * self.obs_dict["occupancy_map"].shape[3])
        visited_voxles_diff = visited_voxles - self.old_visited_voxles
        self.old_visited_voxles = visited_voxles.clone()       
        return compute_reward(
            self.pos_error_vehicle_frame,
            self.pos_error_vehicle_frame_prev,
            obs_dict["crashes"],
            obs_dict["robot_actions"],
            obs_dict["robot_prev_actions"],
            self.curriculum_progress_fraction,
            self.task_config.reward_parameters,
            self.local_grid,
            visited_voxles_diff
        )
        
def extract_centered_tensor(global_voxel_map, sub_voxelmap_size, 
                            position, orientation, 
                            min_value_x, max_value_x, 
                            min_value_y, max_value_y, 
                            min_value_z, max_value_z, 
                            index_base, position_base):
    n = global_voxel_map.shape[0]
    orientation_expanded = orientation.unsqueeze(1).expand(-1, sub_voxelmap_size**3, -1)
    position_expanded = position.unsqueeze(1).expand(-1, sub_voxelmap_size**3, -1)
    position_world = position_expanded + quat_rotate(orientation_expanded.reshape(-1, 4), position_base.reshape(-1, 3)).reshape(n, sub_voxelmap_size**3, 3)
    voxel_indices_globalmap_x = ((position_world - min_value_x) / (max_value_x - min_value_x) * (global_voxel_map.shape[1] - 1)).round().long()
    voxel_indices_globalmap_y = ((position_world - min_value_y) / (max_value_y - min_value_y) * (global_voxel_map.shape[2] - 1)).round().long()
    voxel_indices_globalmap_z = ((position_world - min_value_z) / (max_value_z - min_value_z) * (global_voxel_map.shape[3] - 1)).round().long()
    voxel_indices_globalmap_x = voxel_indices_globalmap_x.clamp(0, global_voxel_map.shape[1] - 1)
    voxel_indices_globalmap_y = voxel_indices_globalmap_y.clamp(0, global_voxel_map.shape[2] - 1)
    voxel_indices_globalmap_z = voxel_indices_globalmap_z.clamp(0, global_voxel_map.shape[3] - 1)
    expanded_indices_x = voxel_indices_globalmap_x.view(voxel_indices_globalmap_x.size(0), -1, 3)
    expanded_indices_y = voxel_indices_globalmap_y.view(voxel_indices_globalmap_y.size(0), -1, 3)
    expanded_indices_z = voxel_indices_globalmap_z.view(voxel_indices_globalmap_z.size(0), -1, 3)
    idx_0 = expanded_indices_x[:, :, 0]
    idx_1 = expanded_indices_y[:, :, 1]
    idx_2 = expanded_indices_z[:, :, 2]
    probability_occupancy = global_voxel_map[torch.arange(n).unsqueeze(1), idx_0, idx_1, idx_2]
    sub_voxel_map = torch.zeros((n, sub_voxelmap_size, sub_voxelmap_size, sub_voxelmap_size), device=global_voxel_map.device, requires_grad=False)
    idx_0_base = index_base[:, :, 0]
    idx_1_base = index_base[:, :, 1]
    idx_2_base = index_base[:, :, 2]
    sub_voxel_map[torch.arange(n).unsqueeze(1), idx_0_base, idx_1_base, idx_2_base] = probability_occupancy
    return sub_voxel_map

@torch.jit.script
def exponential_reward_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function"""
    return magnitude * torch.exp(-(value * value) * exponent)


@torch.jit.script
def exponential_penalty_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function"""
    return magnitude * (torch.exp(-(value * value) * exponent) - 1.0)


@torch.jit.script
def compute_reward(
    pos_error,
    prev_pos_error,
    crashes,
    action,
    prev_action,
    curriculum_progress_fraction,
    parameter_dict,
    local_grid,
    visited_voxles_diff
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor], Tensor, Tensor) -> Tuple[Tensor, Tensor]
    MULTIPLICATION_FACTOR_REWARD = 1.0 #+ (2.0) * curriculum_progress_fraction
    dist = torch.norm(pos_error, dim=1)
    prev_dist_to_goal = torch.norm(prev_pos_error, dim=1)
    pos_reward = exponential_reward_function(
        parameter_dict["pos_reward_magnitude"],
        parameter_dict["pos_reward_exponent"],
        dist,
    )
    very_close_to_goal_reward = exponential_reward_function(
        parameter_dict["very_close_to_goal_reward_magnitude"],
        parameter_dict["very_close_to_goal_reward_exponent"],
        dist,
    )

    getting_closer = prev_dist_to_goal - dist
    getting_closer_reward = torch.where(
        getting_closer > 0,
        parameter_dict["getting_closer_reward_multiplier"] * getting_closer,
        2.0 * parameter_dict["getting_closer_reward_multiplier"] * getting_closer,
    )
    getting_closer_reward = torch.where(prev_dist_to_goal > 100.0, 0.0, getting_closer_reward)

    action_diff = action - prev_action
    x_diff_penalty = exponential_penalty_function(
        parameter_dict["x_action_diff_penalty_magnitude"],
        parameter_dict["x_action_diff_penalty_exponent"],
        action_diff[:, 0],
    )
    y_diff_penalty = exponential_penalty_function(
        parameter_dict["x_action_diff_penalty_magnitude"],
        parameter_dict["x_action_diff_penalty_exponent"],
        action_diff[:, 1],
    )
    z_diff_penalty = exponential_penalty_function(
        parameter_dict["z_action_diff_penalty_magnitude"],
        parameter_dict["z_action_diff_penalty_exponent"],
        action_diff[:, 2],
    )
    yawrate_diff_penalty = exponential_penalty_function(
        parameter_dict["yawrate_action_diff_penalty_magnitude"],
        parameter_dict["yawrate_action_diff_penalty_exponent"],
        action_diff[:, 3],
    )
    pitch_camera_diff_penalty = exponential_penalty_function(
        parameter_dict["camera_action_diff_penalty_magnitude"],
        parameter_dict["camera_action_diff_penalty_exponent"],
        action_diff[:, 4],
    )
    
    yaw_camera_diff_penalty = exponential_penalty_function(
        parameter_dict["camera_action_diff_penalty_magnitude"],
        parameter_dict["camera_action_diff_penalty_exponent"],
        action_diff[:, 5],
    )
    action_diff_penalty = x_diff_penalty + y_diff_penalty + z_diff_penalty + yawrate_diff_penalty + pitch_camera_diff_penalty + yaw_camera_diff_penalty

    # collision penalty
    center_x = local_grid.shape[1] // 2
    center_y = local_grid.shape[2] // 2
    center_z = local_grid.shape[3] // 2
    shift = 3
    local_grid = local_grid[:, center_x-shift:center_x+1+shift, center_y-shift:center_y+1+shift, center_z-shift:center_z+1+shift]
    obstacle_penalty = torch.where((local_grid > 1.1) | (local_grid < 0.1), -1.0, 0.0).sum(dim=(1,2,3))
    obstacle_penalty /= (local_grid.shape[1] * local_grid.shape[2] * local_grid.shape[3])
    # combined reward
    reward = (
        MULTIPLICATION_FACTOR_REWARD
        * (
            pos_reward
            + very_close_to_goal_reward
            + getting_closer_reward
        )
        + 0.01 * action_diff_penalty
        + 100.0 * visited_voxles_diff
    )
    reward = torch.where(obstacle_penalty < 0, obstacle_penalty - 1.0, reward)
    reward[:] = torch.where(
        crashes > 0,
        parameter_dict["collision_penalty"] * torch.ones_like(reward),
        reward,
    )
    return reward, crashes
