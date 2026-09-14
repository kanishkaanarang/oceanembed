"""Joint temperature/salinity reconstruction with depth-wise residual SE blocks."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .physics_losses import eos80_potential_density

STANDARD_DEPTHS = (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000)


class ResidualSE1D(nn.Module):
    def __init__(self, channels=128, dilation=1, dropout=.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels), nn.SiLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
        )
        self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, channels//8, 1),
                                  nn.SiLU(), nn.Conv1d(channels//8, channels, 1), nn.Sigmoid())

    def forward(self, x):
        residual = self.body(x)
        return x + residual * self.gate(residual)


class OceanEmbedNet_v3(nn.Module):
    """Inputs [batch, features]; outputs standardized T/S [batch, 15].

    Continuous depth coordinates account for irregular layer spacing. SE gates
    feature channels using the full column; dilated convolutions couple depths.
    Output de-normalization belongs to the saved training/inference pipeline.
    """
    def __init__(self, input_dim, latent_dim=128, depths=STANDARD_DEPTHS, dropout=.1):
        super().__init__()
        z = torch.as_tensor(depths, dtype=torch.float32)
        if z.ndim != 1 or len(z) != 15 or z[0] != 0 or not torch.all(z.diff() > 0):
            raise ValueError("Exactly 15 strictly increasing depths starting at zero are required.")
        if latent_dim % 8 or latent_dim < 8 or input_dim < 1:
            raise ValueError("latent_dim must be a positive multiple of 8; input_dim must be positive.")
        self.register_buffer("depths", z)
        self.encoder = nn.Sequential(nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.SiLU(),
                                     nn.Dropout(dropout), nn.Linear(256, latent_dim), nn.LayerNorm(latent_dim), nn.SiLU())
        self.depth_encoder = nn.Sequential(nn.Linear(2, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))
        self.depth_embedding = nn.Parameter(torch.randn(1, latent_dim, len(z)) * .02)
        self.column = nn.Sequential(*(ResidualSE1D(latent_dim, d, dropout) for d in (1, 2, 4)))
        self.temperature_head = nn.Sequential(nn.Conv1d(latent_dim, 64, 1), nn.SiLU(), nn.Conv1d(64, 1, 1))
        self.salinity_head = nn.Sequential(nn.Conv1d(latent_dim, 64, 1), nn.SiLU(), nn.Conv1d(64, 1, 1))

    def forward(self, x):
        latent = self.encoder(x)
        coords = torch.stack((self.depths / 1000, torch.log1p(self.depths) / 7), dim=-1)
        tokens = latent.unsqueeze(-1) + self.depth_encoder(coords).T.unsqueeze(0) + self.depth_embedding
        tokens = self.column(tokens)
        return {"temperature": self.temperature_head(tokens).squeeze(1),
                "salinity": self.salinity_head(tokens).squeeze(1), "latent": latent}


class OceanEmbedV3Loss(nn.Module):
    """MSE_T + alpha*MSE_S + beta*stability + gamma*thermocline.

    Supply physical potential temperature (deg C, referenced to 0 dbar) and
    practical salinity, NOT standardized predictions. Physics runs in float32.
    Monotonic cooling is a weak prior, not a universal oceanographic law.
    """
    def __init__(self, depths=STANDARD_DEPTHS, alpha=4., beta=.01, gamma=.1, monotonic_weight=.05):
        super().__init__()
        z = torch.as_tensor(depths, dtype=torch.float32)
        if z.ndim != 1 or len(z) < 2 or not torch.all(z.diff() > 0) or not (z == 10).any():
            raise ValueError("Increasing depth levels including 10 m are required.")
        if min(alpha, beta, gamma, monotonic_weight) < 0:
            raise ValueError("Loss weights must be nonnegative.")
        self.register_buffer("depths", z)
        self.alpha, self.beta, self.gamma, self.monotonic_weight = alpha, beta, gamma, monotonic_weight
        self.reference_index = int((z == 10).nonzero()[0])

    def forward(self, pred_t, pred_s, true_t, true_s, physics_weight=1.):
        with torch.autocast(device_type=pred_t.device.type, enabled=False):
            pt, ps, tt, ts = (v.float() for v in (pred_t, pred_s, true_t, true_s))
            if any(v.shape != tt.shape for v in (pt, ps, ts)) or tt.ndim != 2 or tt.shape[-1] != len(self.depths):
                raise ValueError("All profiles must have matching [batch, depth] shapes.")
            dz = self.depths.diff()
            mse_t, mse_s = F.mse_loss(pt, tt), F.mse_loss(ps, ts)
            rho = eos80_potential_density(ps, pt)
            # Gradient expressed per 100 m to keep beta numerically interpretable.
            hydro = F.relu(-rho.diff(dim=-1) / dz * 100).square().mean()
            pg, tg = pt.diff(dim=-1) / dz, tt.diff(dim=-1) / dz
            mid = (self.depths[1:] + self.depths[:-1]) * .5
            weight = 1 + 3 * ((mid >= 50) & (mid <= 150)).float()
            gradient = ((pg - tg).square() * weight).mean() / .05**2
            # Target-defined MLD prevents the network moving MLD to evade loss.
            with torch.no_grad():
                target_rho = eos80_potential_density(ts, tt)
                crossed = (target_rho - target_rho[:, self.reference_index, None] >= .03)
                crossed &= self.depths[None] > 10
                mld = torch.where(crossed, self.depths[None], torch.inf).amin(dim=-1)
                below = self.depths[:-1][None] >= mld[:, None]
                # Permit observed salinity-compensated warm inversions.
                allowed_warming = F.relu(tg)
            excess_warming = F.relu(pg - allowed_warming - .001)
            monotonic = ((excess_warming / .05).square() * below).sum() / below.sum().clamp_min(1)
            thermocline = gradient + self.monotonic_weight * monotonic
            total = mse_t + self.alpha*mse_s + physics_weight*(self.beta*hydro + self.gamma*thermocline)
            return {"total": total, "mse_temperature": mse_t, "mse_salinity": mse_s,
                    "hydrostatic": hydro, "thermocline": thermocline, "monotonic": monotonic}
