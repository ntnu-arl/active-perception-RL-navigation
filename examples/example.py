import time
from aerial_gym.utils.logging import CustomLogger

logger = CustomLogger(__name__)
import torch

from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.robots.base_multirotor import BaseMultirotor


from src.task.navigation_active_camera_task import NavigationActiveCameraTask
from src.config.task.navigation_active_camera_task_config import (
    task_config as navigation_active_camera_task_config,
)
from src.config.env.env_active_camera_with_obstacles import EnvActiveCameraWithObstaclesCfg
from src.config.robot.active_camera_quad_config import LMF2ActiveCameraCfg

env_config_registry.register("env_active_camera_with_obstacles", EnvActiveCameraWithObstaclesCfg)
task_registry.register_task("navigation_active_camera_task", NavigationActiveCameraTask, navigation_active_camera_task_config)
robot_registry.register("active_camera_quadrotor", BaseMultirotor, LMF2ActiveCameraCfg)


if __name__ == "__main__":
    logger.print_example_message()
    start = time.time()
    rl_task_env = task_registry.make_task(
        "inspection_task",
        # other params are not set here and default values from the task config file are used
    )
    rl_task_env.reset()
    actions = torch.zeros(
        (
            rl_task_env.sim_env.num_envs,
            rl_task_env.sim_env.robot_manager.robot.controller_config.num_actions,
        )
    ).to("cuda:0")
    actions[:] = 0.0
    with torch.no_grad():
        for i in range(10000):
            if i == 100:
                start = time.time()
            obs, reward, terminated, truncated, info = rl_task_env.step(actions=actions)
    end = time.time()