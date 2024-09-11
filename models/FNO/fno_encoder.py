import torch.nn as nn
import torch

from typing import List, Tuple

from neuralop.models import FNO

class FNOEncoder(nn.Module):
    def __init__(
        self,
        n_modes: List[int],
        in_dim: int,
        out_dim: int,
        rank: float=0.3,
        n_layers: int=4,
        embed_dim: int=128,
        prj_ff_mult: float=0.5,
        lift_ff_mult: float=4.0,
        apply_norm: bool=True,
        spatiotemporal_size: Tuple[int, int, int] = (16, 128, 128),
        use_projection: bool=False,
        masking_method: str='freq', # 'freq' or 'spatial'
        # ssl_type: str = 'jepa', # 'jepa' or 'mae'
        factorization: str = 'tucker',
    ):
        super(FNOEncoder, self).__init__()

        # Map input to predictor dimension
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.in_channels = in_dim + out_dim
        self.out_channels = self.embed_dim = embed_dim
        self.hidden_channels = embed_dim 
        self.projection_channels = int(self.hidden_channels * prj_ff_mult) if use_projection else None
        self.lifting_channels = int(self.hidden_channels * lift_ff_mult)

        self.use_projection = use_projection
        self.masking_method = masking_method

        self.encoder = FNO(
            n_modes = n_modes,
            in_channels = self.in_channels,
            hidden_channels = self.hidden_channels,
            out_channels = self.out_channels,
            n_layers = n_layers,
            lifting_channels = self.lifting_channels,
            projection_channels = self.projection_channels,
            rank=rank,
            factorization=factorization,
            positional_embedding=None,
            use_projection = self.use_projection,
        )

        self.apply_norm = apply_norm
        if apply_norm:
            self.encoder_norm = nn.InstanceNorm3d(embed_dim)
            
    def forward(self, x: torch.Tensor, pos_enc=None, embedding=None, normalizer=None, embed_resolution=None):
        if pos_enc and embed_resolution is not None:
            x = torch.cat([x, pos_enc], dim=1)
            x = self.encoder(x, output_shape=embed_resolution)
        else:
            x = x.squeeze()
            x = self.encoder(x)
        if self.apply_norm:
            x = self.encoder_norm(x)
        return x