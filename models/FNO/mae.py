import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from models.FNO.fno_encoder import FNOEncoder

class FNOMAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_modes,
        decoder_width,
        masking_ratio,
        decoder_layers
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = masking_ratio

        # Encoder (FNO)
        self.encoder = encoder
        self.input_dim = encoder.in_dim
        self.output_dim = encoder.out_dim
        
        # Decoder (FNO)
        self.decoder = FNOEncoder(
            n_modes=decoder_modes,
            in_dim=self.output_dim,
            out_dim=self.input_dim,
            n_layers=decoder_layers,
            embed_dim=decoder_width,
        )
        
        #print(self.decoder)
        # Mask token
        self.mask_token = nn.Parameter(torch.randn(decoder_width))

    def forward(self, x, embedding=None, normalizer=None, features=None):
        # x shape: [batch_size, input_dim, *spatial_dims]
        device = x.device
        batch_size = x.shape[0]
        spatial_dims = x.shape[2:]
        num_elements = torch.prod(torch.tensor(spatial_dims))
        
        #print("Original shape", x.shape, num_elements)
        # Calculate number of elements to mask
        num_masked = int(self.masking_ratio * num_elements)
        #print("Num masked", num_masked)
        # Create mask
        mask = torch.zeros(batch_size, x.shape[1], num_elements, device=device)
        #print("Original mask", mask.shape)
        mask[:, :, :num_masked] = 1
        #mask = mask[torch.randperm(num_elements)]
        #print(mask.shape, *spatial_dims)
        mask = mask.reshape(batch_size, x.shape[1], *spatial_dims)
        
        #print("New shape mask", mask.shape)
        # Apply mask to input
        masked_input = x * (1 - mask)
        masked_input = masked_input.unsqueeze(1) # to undo channel dimension
        # Encode
        #print("new shape", masked_input.shape)
        encoded = self.encoder(masked_input)
        
        #print("encode shape", encoded.shape)

        # Replace masked tokens
        #mask_tokens = repeat(self.mask_token, 'd -> b n d', b=batch_size, n=num_masked)
        #print(mask_tokens.shape, mask.shape, encoded.shape)
        #print((encoded * (1 - mask.unsqueeze(1))).shape)
        #print((mask_tokens * mask).shape)
        #encoded_masked = encoded * (1 - mask.unsqueeze(1)) + mask_tokens * mask
        
        encoded_masked = encoded
        # Decode
        decoded = self.decoder(encoded_masked)
            
        #print(decoded.shape)
        # Calculate loss only for masked elements - not now
        #loss = F.mse_loss(decoded * mask.unsqueeze(1), x * mask.unsqueeze(1))
        loss = F.mse_loss(decoded.squeeze(), x)
        if not features:
            return loss
        else:
            return x, decoded.squeeze(), mask

    def encode(self, x, mask_ratio=0):
        # Optional: allow different mask ratio during encoding
        device = x.device
        batch_size = x.shape[0]
        spatial_dims = x.shape[2:]
        num_elements = torch.prod(torch.tensor(spatial_dims))

        num_masked = int(mask_ratio * num_elements)

        mask = torch.zeros(batch_size, num_elements, device=device)
        mask[:, :num_masked] = 1
        mask = mask[torch.randperm(num_elements)]
        mask = mask.reshape(batch_size, *spatial_dims)

        masked_input = x * (1 - mask.unsqueeze(1))

        return self.encoder(masked_input)