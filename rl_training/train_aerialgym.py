# this is here just to guarantee that isaacgym is imported before PyTorch
# isort: off
# noinspection PyUnresolvedReferences

# isort: on

from aerial_gym.registry.task_registry import task_registry
import sys
from typing import Dict, Optional, Tuple, List


import gymnasium as gym
import torch
import numpy as np


from torch import Tensor

from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.model.encoder import Encoder
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.model.model_utils import nonlinearity
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import *
from sample_factory.train import run_rl
from sample_factory.utils.typing import Config, Env
from sample_factory.utils.utils import str2bool

from rl_training.utils.encoder import make_map_encoder

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

class AerialGymVecEnv(gym.Env):
    """
    Wrapper for isaacgym environments to make them compatible with the sample factory.
    """

    def __init__(self, aerialgym_env, obs_key):
        self.env = aerialgym_env
        self.num_agents = self.env.num_envs
        self.action_space = convert_space(self.env.action_space)

        # isaacgym_examples environments actually return dicts
        if obs_key == "obs" or obs_key == "observations":
            new_spaces = {}
            new_spaces.update(gym.spaces.Dict(dict(observations=convert_space(self.env.observation_space["observations"]))))
            new_spaces.update(gym.spaces.Dict(dict(observations_map=convert_space(self.env.observation_space["observations_map"]))))
            self.observation_space = gym.spaces.Dict(new_spaces)
        else:
            raise ValueError(f"Unknown observation key: {obs_key}")

        self._truncated: Tensor = torch.zeros(self.num_agents, dtype=torch.bool)

    def reset(self, *args, **kwargs) -> Tuple[Dict[str, Tensor], Dict]:
        # some IGE envs return all zeros on the first timestep, but this is probably okay
        obs, rew, terminated, truncated, infos = self.env.reset()
        return obs, infos

    def step(self, action) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict]:
        obs, rew, terminated, truncated, infos = self.env.step(action)
        return obs, rew, terminated, truncated, infos

    def render(self):
        pass


def make_aerialgym_env(
    full_task_name: str,
    cfg: Config,
    _env_config=None,
    render_mode: Optional[str] = None,
) -> Env:

    return AerialGymVecEnv(task_registry.make_task(task_name=full_task_name), "obs")


def add_extra_params_func(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument(
        "--env_agents",
        default=-1,
        type=int,
        help="Num agents in each env (default: -1, means use default value from isaacgymenvs env yaml config file)",
    )
    p.add_argument(
        "--obs_key",
        default="obs",
        type=str,
        help='IsaacGym envs return dicts, some envs return just "obs", and some return "obs" and "states".'
        "States key denotes the full state of the environment, and obs key corresponds to limited observations "
        'available in real world deployment. If we use "states" here we can train will full information '
        "(although the original idea was to use asymmetric training - critic sees full state and policy only sees obs).",
    )
    p.add_argument(
        "--subtask",
        default=None,
        type=str,
        help="Subtask for envs that support it (i.e. AllegroKuka regrasping or manipulation or throw).",
    )
    p.add_argument(
        "--ige_api_version",
        default="preview4",
        type=str,
        choices=["preview3", "preview4"],
        help="We can switch between different versions of IsaacGymEnvs API using this parameter.",
    )
    p.add_argument(
        "--eval_stats",
        default=False,
        type=str2bool,
        help="Whether to collect env stats during evaluation.",
    )


def override_default_params_func(env, parser):
    """Most of these parameters are taken from IsaacGymEnvs default config files."""

    parser.set_defaults(
        # we're using a single very vectorized env, no need to parallelize it further
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        actor_worker_gpus=[0],  # obviously need a GPU
        train_for_env_steps=10000000,
        use_rnn=False,
        adaptive_stddev=True,
        policy_initialization="torch_default",
        env_gpu_actions=True,
        reward_scale=0.1,
        rollout=24,
        max_grad_norm=0.0,
        batch_size=2048,
        num_batches_per_epoch=2,
        num_epochs=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=2.0,
        exploration_loss_coeff=0.0,
        nonlinearity="elu",
        learning_rate=3e-4,
        lr_schedule="kl_adaptive_epoch",
        lr_schedule_kl_threshold=0.016,
        shuffle_minibatches=True,
        gamma=0.98,
        gae_lambda=0.95,
        with_vtrace=False,
        value_bootstrap=True,  # assuming reward from the last step in the episode can generally be ignored
        normalize_input=True,
        normalize_returns=True,  # does not improve results on all envs, but with return normalization we don't need to tune reward scale
        save_best_after=int(5e6),
        serial_mode=True,  # it makes sense to run isaacgym envs in serial mode since most of the parallelism comes from the env itself (although async mode works!)
        async_rl=True,
        use_env_info_cache=False,  # speeds up startup
        kl_loss_coeff=0.1,
        restart_behavior="overwrite",
    )

    # override default config parameters for specific envs
    if env in env_configs:
        parser.set_defaults(**env_configs[env])


# custom default configuration parameters for specific envs
# add more envs here analogously (env names should match config file names in IGE)
env_configs = dict(
    navigation_active_camera_task=dict(
        train_for_env_steps=131000000000,
        encoder_mlp_layers=[256, 128, 64],
        encoder_mlp_layers_custom=[256, 128, 64],
        encoder_conv_map_occupancy_architecture="resnet_impala_occupancy",
        encoder_conv_map_occupancy_mlp_layers=[],
        use_rnn=True,
        rnn_num_layers=1,
        rnn_size=64,
        rnn_type="gru",
        gamma=0.98,
        rollout=32,
        learning_rate=1e-4,
        lr_schedule_kl_threshold=0.016,
        batch_size=2048,
        num_epochs=4,
        max_grad_norm=1.0,
        num_batches_per_epoch=4,
        exploration_loss_coeff=0.0,
        with_wandb=False,
        # normalize_input_keys=["observations"],
    ),
)

class CustomEncoder(Encoder):
    """Just an example of how to use a custom model component."""

    def __init__(self, cfg, obs_space):
        super().__init__(cfg)

        out_size = 0
        self.encoders = torch.nn.ModuleDict()
        
        # robot state
        out_size += obs_space["observations"].shape[0]

        # 3D map observation        
        encoder_fn_map = make_map_encoder
        self.encoders["observations_map"] = encoder_fn_map(cfg, obs_space["observations_map"], "occupancy")
        out_size += self.encoders["observations_map"].get_out_size()

        # MLP for latent space encoding of all observations
        obs_space_custom = spaces.Box(np.ones(out_size) * -np.Inf, np.ones(out_size) * np.Inf)
        mlp_layers: List[int] = cfg.encoder_mlp_layers_custom
        self.mlp_head_custom = create_mlp(mlp_layers, obs_space_custom.shape[0], nonlinearity(cfg))
        if len(mlp_layers) > 0:
            self.mlp_head_custom = torch.jit.script(self.mlp_head_custom)
        self.encoder_out_size = calc_num_elements(self.mlp_head_custom, obs_space_custom.shape)
        
    def forward(self, obs_dict):
        x_map_encoding = self.encoders["observations_map"](obs_dict["observations_map"])
        encoding = self.mlp_head_custom(torch.cat((obs_dict["observations"], x_map_encoding), 1))
        return encoding

    def get_out_size(self) -> int:
        return self.encoder_out_size

def make_custom_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function as required by the API."""
    return CustomEncoder(cfg, obs_space)

def register_aerialgym_custom_components():
    for env_name in env_configs:
        register_env(env_name, make_aerialgym_env)
    
    global_model_factory().register_encoder_factory(make_custom_encoder)


def parse_aerialgym_cfg(evaluation=False):
    parser, partial_cfg = parse_sf_args(evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params_func(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser)
    return final_cfg


def main():
    """Script entry point."""
    register_aerialgym_custom_components()
    cfg = parse_aerialgym_cfg()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
