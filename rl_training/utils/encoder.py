from typing import Dict, List, Optional

import torch
from gymnasium import spaces
from torch import Tensor, nn

from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.model_utils import ModelModule, create_mlp, model_device, nonlinearity
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class Encoder(ModelModule):
    def __init__(self, cfg: Config):
        super().__init__(cfg)

    def get_out_size(self) -> int:
        raise NotImplementedError()

    def model_to_device(self, device):
        """Default implementation, can be overridden in derived classes."""
        self.to(device)

    def device_for_input_tensor(self, input_tensor_name: str) -> Optional[torch.device]:
        return model_device(self)

    def type_for_input_tensor(self, input_tensor_name: str) -> torch.dtype:
        return torch.float32

class ResBlock3D(nn.Module):
    def __init__(self, cfg, input_ch, output_ch):
        super().__init__()

        layers = [
            nonlinearity(cfg),
            nn.Conv3d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
            nonlinearity(cfg),
            nn.Conv3d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        identity = x
        out = self.res_block_core(x)
        out = out + identity
        return out

class Resnet3DEncoder(Encoder):
    def __init__(self, cfg, obs_space, type):
        super().__init__(cfg)

        input_ch = obs_space.shape[0]
        log.debug("Num input channels: %d", input_ch)

        if type == "occupancy":
            # configuration from the IMPALA paper
            resnet_conf = [[8, 2], [16, 2], [16, 2]]
        else:
            raise NotImplementedError(f"Unknown resnet architecture {cfg.encode_conv_map_architecture}")

        curr_input_channels = input_ch
        layers = []
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            layers.extend(
                [
                    nn.Conv3d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1),  # padding SAME
                    nn.MaxPool3d(kernel_size=3, stride=2, padding=1),  # padding SAME
                ]
            )

            for j in range(res_blocks):
                layers.append(ResBlock3D(cfg, out_channels, out_channels))

            curr_input_channels = out_channels

        activation = nonlinearity(cfg)
        layers.append(activation)

        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_space.shape)
        log.debug(f"Convolutional layer output size: {self.conv_head_out_size}")

        # should we do torch.jit here?
        if type == "occupancy":
            self.mlp_layers = create_mlp(cfg.encoder_conv_map_occupancy_mlp_layers, self.conv_head_out_size, activation)
        else:
            raise NotImplementedError(f"Unknown resnet architecture {cfg.encode_conv_map_mlp_architecture}")
        
        self.encoder_out_size = calc_num_elements(self.mlp_layers, (self.conv_head_out_size,))

    def forward(self, obs: Tensor):
        x = self.conv_head(obs)
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.mlp_layers(x)
        return x

    def get_out_size(self) -> int:
        return self.encoder_out_size

class OccupancyGridEncoder(nn.Module):
    def __init__(self, cfg, obs_space):
        super().__init__()
        
        # Configurable parameters
        latent_dim = 128
        embedding_dim = 8  # <-- fixed small embedding dimension (not grid size!)
        
        # Embed categorical values {0,1,2} into vectors
        self.embedding = nn.Embedding(num_embeddings=3, embedding_dim=embedding_dim)
        
        # 3D CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv3d(embedding_dim, 32, kernel_size=3, stride=2, padding=1),  # (21 → 11)
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),             # (11 → 6)
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),            # (6 → 3)
            nn.ReLU(inplace=True),
        )
        self.conv_head_out_size = calc_num_elements(self.conv_layers, (8, *obs_space.shape[1:]))        # Flatten + linear projection
        self.fc = nn.Linear(self.conv_head_out_size, latent_dim)
        self.encoder_out_size = latent_dim  # final embedding size
    
    def forward(self, x):
        """
        x: (batch, D, H, W) occupancy grid with values {0,1,2}
           or (batch, 1, D, H, W) if channel dim exists
        """
        # # DEBUG check
        # min_val, max_val = x.min().item(), x.max().item()
        # if min_val < 0 or max_val > 2:
        #     print(f"⚠️ Invalid occupancy values: min={min_val}, max={max_val}, uniques={x.unique()[:20]}")
        #     raise ValueError("Input to embedding has invalid indices!")
        
        if x.dim() == 5 and x.shape[1] == 1:
            # remove channel dimension
            x = x.squeeze(1)  # (B, D, H, W)

        x = self.embedding(x.long())       # (B, D, H, W, emb)
        x = x.permute(0, 4, 1, 2, 3)       # (B, emb, D, H, W)
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        latent = self.fc(x)
        return latent

    def get_out_size(self) -> int:
        """Return latent vector size (after FC)"""
        return self.encoder_out_size
    
def make_map_encoder(cfg: Config, obs_space: ObsSpace, type) -> Encoder:
    """Make (most likely convolutional) encoder for 3Dmap-based observations."""
    if type == "occupancy":
        if cfg.encoder_conv_map_occupancy_architecture.startswith("resnet"):
            # return OccupancyGridEncoder(cfg, obs_space)
            return Resnet3DEncoder(cfg, obs_space, type)
        else:
            raise NotImplementedError(f"Unknown convolutional architecture {cfg.encoder_conv_map_occupancy_architecture}")