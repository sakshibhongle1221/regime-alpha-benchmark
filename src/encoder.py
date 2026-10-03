import torch
import torch.nn as nn
import torch.nn.functional as F

from config import N_FEATURES, LATENT_DIM


class TemporalEncoder(nn.Module):
    """
    Temporal Fusion Transformer encoder for contrastive regime learning.

    Input:  (batch, window, n_features)
    Output: (batch, latent_dim) — L2-normalized regime embedding

    Architecture:
      1. Linear projection: n_features → d_model
      2. Learnable positional encoding
      3. 2-layer Transformer encoder (batch_first=True)
      4. Mean pooling over time axis
      5. MLP projection head → latent_dim
      6. L2 normalization
    """
    def __init__(
        self,
        n_features: int = N_FEATURES,
        d_model:    int = 64,
        n_heads:    int = 4,
        n_layers:   int = 2,
        latent_dim: int = LATENT_DIM,
        dropout:    float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = nn.Parameter(torch.randn(1, 512, d_model) * 0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.proj_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = self.input_proj(x)            # (B, T, d_model)
        x = x + self.pos_enc[:, :T, :]   # add positional encoding
        x = self.transformer(x)           # (B, T, d_model)
        x = x.mean(dim=1)                 # mean pool → (B, d_model)
        z = self.proj_head(x)             # (B, latent_dim)
        z = F.normalize(z, dim=-1)        # L2 normalize
        return z


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR).

    For each anchor, the adjacent window is the positive.
    All other items in the batch are negatives.
    Lower temperature = harder negatives = better separation.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_anchor: torch.Tensor, z_positive: torch.Tensor) -> torch.Tensor:
        B = z_anchor.shape[0]
        z = torch.cat([z_anchor, z_positive], dim=0)   # (2B, latent_dim)

        sim  = torch.mm(z, z.T) / self.temperature     # (2B, 2B)
        mask = torch.eye(2 * B, device=z.device).bool()
        sim.masked_fill_(mask, float("-inf"))

        labels = torch.cat([
            torch.arange(B, 2 * B),
            torch.arange(0, B),
        ]).to(z.device)

        return F.cross_entropy(sim, labels)
