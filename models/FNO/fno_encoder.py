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
        n_layers: int=4,
        embed_dim: int=128,
        use_projection: bool=False,
    ):
        super(FNOEncoder, self).__init__()

        # Map input to predictor dimension

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.in_channels = in_dim
        self.out_channels = out_dim
        self.hidden_channels = embed_dim 
        self.projection_channels = int(self.hidden_channels) if use_projection else None

        self.encoder = FNO(
            n_modes = n_modes,
            in_channels = self.in_channels,
            hidden_channels = self.hidden_channels,
            out_channels = self.out_channels,
            n_layers = n_layers,
            lifting_channels = self.hidden_channels,
            projection_channels = self.hidden_channels,
        )
        
    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        return x