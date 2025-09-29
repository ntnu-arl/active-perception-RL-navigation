import os
ACTIVE_CAMERA_NAVIGATION_DIRECTORY = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

from aerial_gym.control.controllers.velocity_control import LeeVelocityController

from aerial_gym.registry.controller_registry import controller_registry

from src.config.controller.controller_config import (
    control as controller_config,
)

controller_registry.register_controller(
    "velocity_control", LeeVelocityController, controller_config
)