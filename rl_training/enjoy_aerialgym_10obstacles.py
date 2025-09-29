# this is here just to guarantee that isaacgym is imported before PyTorch
# isort: off
# noinspection PyUnresolvedReferences
import isaacgym

# isort: on

import sys

from sample_factory.enjoy import enjoy
from rl_training.train_aerialgym import (
    parse_aerialgym_cfg,
    register_aerialgym_custom_components,
)

from aerial_gym.registry.env_registry import env_config_registry
from aerial_gym.registry.task_registry import task_registry
from aerial_gym.registry.robot_registry import robot_registry
from aerial_gym.robots.base_multirotor import BaseMultirotor

from src.task.navigation_active_camera_task import NavigationActiveCameraTask
from src.config.task.navigation_active_camera_task_config_example import (
    task_config_example as navigation_active_camera_task_config,
)
from src.config.env.env_active_camera_with_10obstacles import EnvActiveCameraWith10ObstaclesCfg
from src.config.robot.active_camera_quad_config import LMF2ActiveCameraCfg

env_config_registry.register("env_active_camera_with_obstacles", EnvActiveCameraWith10ObstaclesCfg)
task_registry.register_task("navigation_active_camera_task", NavigationActiveCameraTask, navigation_active_camera_task_config)
robot_registry.register("active_camera_quadrotor", BaseMultirotor, LMF2ActiveCameraCfg)

def main():
    """Script entry point."""
    register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
