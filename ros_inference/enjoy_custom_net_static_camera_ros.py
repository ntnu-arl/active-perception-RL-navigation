from typing import List
import time

import numpy as np
import torch

from gymnasium import spaces

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import Decoder
# from sample_factory.model.core import ModelCore, ModelCoreRNN
from sample_factory.model.model_utils import nonlinearity
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import *

from sample_factory.algo.utils.gymnasium_utils import convert_space
from sample_factory.utils.typing import Config
from sample_factory.cfg.arguments import load_from_checkpoint, parse_full_cfg, parse_sf_args
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size

from sample_factory.algo.learning.learner import Learner

from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor


from rl_training.utils.encoder import make_map_encoder

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

class NN_Inference_ROS(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        global_model_factory().register_encoder_factory(make_custom_encoder)
        self.cfg = load_from_checkpoint(cfg)
        self.cfg.num_envs = 1
        self.num_actions = 6
        self.num_obs = 85
        self.num_agents = 1

        self.observation_space = spaces.Box(
            np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.observation_space_map = spaces.Box(np.ones(
            (1, 21, 21, 21), dtype=np.float32) * -np.Inf, np.ones((1, 21, 21, 21), dtype=np.float32) * np.Inf)
        new_spaces = {}
        new_spaces.update(spaces.Dict(dict(observations=convert_space(self.observation_space))))
        new_spaces.update(spaces.Dict(dict(observations_map=convert_space(self.observation_space_map))))
        self.observation_space = spaces.Dict(new_spaces)
        
        self.action_space = convert_space(spaces.Box(
            np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.))
        
        self.init_env_info()
        
        self.actor_critic = create_actor_critic(self.cfg, self.env_info.obs_space, self.env_info.action_space)

        self.actor_critic.eval()
        device = torch.device("cpu" if self.cfg.device == "cpu" else "cuda")
        self.actor_critic.model_to_device(device)
        print("Model:\n\n", self.actor_critic)
        
        
        # Load policy into model
        policy_id = self.cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(self.cfg, policy_id), f"{name_prefix}_*")
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        self.actor_critic.load_state_dict(checkpoint_dict["model"])
        self.rnn_states = torch.zeros([self.num_agents, get_rnn_size(self.cfg)], dtype=torch.float32, device=device)
    
    def init_env_info(self):
        self.env_info = EnvInfo(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_agents=self.num_agents,
            gpu_actions=self.cfg.env_gpu_actions,
            gpu_observations=self.cfg.env_gpu_observations,
            action_splits=None,
            all_discrete=None,
            frameskip=self.cfg.env_frameskip
        )
    
    def reset(self):
        self.rnn_states[:] = 0.0
    
    def get_action(self, obs):
        start_time = time.time()
        with torch.no_grad():
            # put obs to device
            processed_obs = prepare_and_normalize_obs(self.actor_critic, obs)
            policy_outputs = self.actor_critic(processed_obs, self.rnn_states)
            # sample actions from the distribution by default
            actions = policy_outputs["actions"]
            # if self.cfg.eval_deterministic:
            action_distribution = self.actor_critic.action_distribution()
            actions = argmax_actions(action_distribution)
            
            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(self.env_info, actions)

            self.rnn_states = policy_outputs["new_rnn_states"]

        actions_np = actions[0].cpu().numpy()
        # print("Time to get action:", time.time() - start_time)
        return actions_np

def make_custom_encoder(cfg: Config, obs_space: ObsSpace) -> Encoder:
    """Factory function as required by the API."""
    return CustomEncoder(cfg, obs_space)

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
        shuffle_minibatches=False,
        gamma=0.98,
        gae_lambda=0.95,
        with_vtrace=False,
        value_bootstrap=True,  # assuming reward from the last step in the episode can generally be ignored
        normalize_input=True,
        normalize_returns=True,  # does not improve results on all envs, but with return normalization we don't need to tune reward scale
        save_best_after=int(5e6),
        serial_mode=True,  # it makes sense to run isaacgym envs in serial mode since most of the parallelism comes from the env itself (although async mode works!)
        async_rl=False,
        use_env_info_cache=True, # speeds up startup
        kl_loss_coeff=0.1,
    )

    # override default config parameters for specific envs
    if env in env_configs:
        parser.set_defaults(**env_configs[env])


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

def parse_aerialgym_cfg(evaluation=False):
    parser, partial_cfg = parse_sf_args(evaluation=evaluation)
    # add_extra_params_func(parser)
    override_default_params_func(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser)
    return final_cfg

@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c

@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)

@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c