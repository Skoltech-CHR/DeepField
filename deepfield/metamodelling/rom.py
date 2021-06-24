"""ROM utils."""
#pylint: disable=protected-access,too-many-locals,wrong-import-order,wrong-import-position
import sys

import torch
from torch import nn

sys.path.append('..')

from .custom_blocks import norm, nonlinearity, VoxelShuffle
from .custom_blocks.wrappers import (TimeInvariantWrapper, MultiInputSequential,
                                     CheckpointWrapper)
from .autoencoding import SpatialAutoencoder
from .dynamics import NeuralDifferentialEquation, LatentSpaceDynamics
from .factories import sequential_factory


def init_metamodel(autoenc_path=None, lsd_path=None, device=None, use_norm_state=True,
                   use_norm_params=True, use_norm_control=False,
                   s_ch=5, params_ch=4, control_ch=1, z_ch=32,
                   max_integration_timestep=1000, use_checkpointing=True,
                   use_inner_checkpointing=None, max_in_batch=1, max_seq_len=15, atol=1e-3):
    """Build and load metamodel."""
    n_layers = 4
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if use_inner_checkpointing is None:
        use_inner_checkpointing = use_checkpointing

    state_enc = sequential_factory(
        n_layers, nn.Conv3d, s_ch, [8, 16, 32, 32], kernel_size=[3, 2, 3, 2],
        padding=[1, 0, 1, 0], stride=[1, 2, 1, 2], residual_connections=(),
        wrappers=(MultiInputSequential, CheckpointWrapper, TimeInvariantWrapper),
        use_norm=use_norm_state
    )

    if use_norm_state:
        state_dec = [
            nn.Conv3d(32, 16 * 8, kernel_size=3, stride=1, padding=1),
            VoxelShuffle(2), norm(16), nonlinearity(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            norm(16), nonlinearity(),
            nn.Conv3d(16, 8 * 8, kernel_size=3, stride=1, padding=1),
            VoxelShuffle(2), norm(8), nonlinearity(),
            nn.Conv3d(8, s_ch, kernel_size=3, stride=1, padding=1)
        ]
    else:
        state_dec = [
            nn.Conv3d(32, 16 * 8, kernel_size=3, stride=1, padding=1),
            VoxelShuffle(2), nonlinearity(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nonlinearity(),
            nn.Conv3d(16, 8 * 8, kernel_size=3, stride=1, padding=1),
            VoxelShuffle(2), nonlinearity(),
            nn.Conv3d(8, s_ch, kernel_size=3, stride=1, padding=1)
        ]

    state_dec = TimeInvariantWrapper(CheckpointWrapper(MultiInputSequential(*state_dec)))

    autoencoder = SpatialAutoencoder(encoder=state_enc,
                                     decoder=state_dec,
                                     attr='states').to(device)
    if autoenc_path is not None:
        autoencoder.load(autoenc_path)

    params_enc = sequential_factory(
        n_layers, nn.Conv3d, params_ch, [8, 8, 8, 8], kernel_size=[3, 2, 3, 2],
        padding=[1, 0, 1, 0], stride=[1, 2, 1, 2], residual_connections=(),
        wrappers=(MultiInputSequential, CheckpointWrapper, TimeInvariantWrapper),
        use_norm=use_norm_params
    )

    control_enc = sequential_factory(
        n_layers, nn.Conv3d, control_ch, [8, 8, 8, 8], kernel_size=[3, 2, 3, 2],
        padding=[1, 0, 1, 0], stride=[1, 2, 1, 2], residual_connections=(),
        wrappers=(MultiInputSequential, CheckpointWrapper, TimeInvariantWrapper),
        use_norm=use_norm_control
    )

    time_derivative_module = sequential_factory(
        2, nn.Conv3d, z_ch + 8 + 8, [32, z_ch], kernel_size=[3, 3],
        padding=[1, 1], stride=[1, 1], residual_connections=(),
        wrappers=(MultiInputSequential, CheckpointWrapper, TimeInvariantWrapper)
    )

    nde = NeuralDifferentialEquation(time_derivative_module).to(device)
    lsd = LatentSpaceDynamics(state_enc, state_dec, params_enc, control_enc, nde).to(device)

    if lsd_path is not None:
        lsd.load(lsd_path)

    kwargs = dict(
        max_integration_timestep=max_integration_timestep,
        use_checkpointing=use_checkpointing,
        use_inner_checkpointing=use_inner_checkpointing,
        max_in_batch=max_in_batch,
        max_seq_len=max_seq_len,
        atol=atol)
    ae_kwargs = lsd._parse_ae_kwargs(kwargs)

    return lsd, ae_kwargs, kwargs
