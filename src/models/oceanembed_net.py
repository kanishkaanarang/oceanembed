"""OceanEmbedNet PyTorch CNN Architecture and Inference Engine.

Predicts full 3D subsurface temperature fields (15 standard depths)
from 7 surface satellite observation channels for the North Indian Ocean.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn

STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
LAT_MIN, LAT_MAX, LAT_STEP = 5.0, 30.0, 0.25
LON_MIN, LON_MAX, LON_STEP = 45.0, 105.0, 0.25

LAT_POINTS = int(round((LAT_MAX - LAT_MIN) / LAT_STEP))  # 100
LON_POINTS = int(round((LON_MAX - LON_MIN) / LON_STEP))  # 240

COMMON_LATS = np.linspace(LAT_MIN, LAT_MAX - LAT_STEP, LAT_POINTS)
COMMON_LONS = np.linspace(LON_MIN, LON_MAX - LON_STEP, LON_POINTS)

CHANNEL_NAMES = [
    "SST (Sea Surface Temp)",
    "SSS (Sea Surface Salinity)",
    "SSH (Sea Level Anomaly)",
    "uwnd (Zonal Wind)",
    "vwnd (Meridional Wind)",
    "ucurr (Zonal Current)",
    "vcurr (Meridional Current)",
]


class OceanEmbedNet(nn.Module):
    """Deep CNN embedding network for vertical ocean reconstruction."""

    def __init__(self, in_channels: int = 7, embed_channels: int = 32, out_depths: int = 15):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, embed_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_depths, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        out = self.decoder(embedding)
        return out, embedding


class OceanEmbedEngine:
    """Manages weights, normalization statistics, dataset tensors, and caching."""

    _instance: OceanEmbedEngine | None = None

    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parent.parent.parent
        self.model_path = self._find_file([
            self.root / "results" / "oceanembed_cnn_best_v2.pt",
            self.root / "notebooks" / "results" / "oceanembed_cnn_best_v2.pt",
        ])
        self.norm_path = self._find_file([
            self.root / "training_arrays_v2_normalized" / "training_arrays_v2_normalized.npz",
            self.root / "data" / "processed" / "training_arrays_v2_normalized.npz",
        ])
        self.raw_path = self._find_file([
            self.root / "training_arrays_v2_normalized" / "training_arrays_v2.npz",
            self.root / "data" / "processed" / "training_arrays_v2.npz",
        ])

        # Load model
        self.model = OceanEmbedNet(in_channels=7, embed_channels=32, out_depths=15)
        if self.model_path and os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model checkpoint not found in {self.model_path}")

        # Load normalization arrays
        if not self.norm_path or not os.path.exists(self.norm_path):
            raise FileNotFoundError(f"Normalization data not found in {self.norm_path}")

        norm_data = np.load(self.norm_path)
        self.norm_inputs = norm_data["inputs"]  # (150, 7, 100, 240)
        self.norm_targets = norm_data["targets"]  # (150, 15, 100, 240)
        self.ocean_mask = norm_data["ocean_mask"].astype(bool)  # (100, 240)
        self.target_mean = norm_data["target_mean"].astype(np.float32)  # (1, 15, 1, 1)
        self.target_std = norm_data["target_std"].astype(np.float32)  # (1, 15, 1, 1)
        self.input_mean = norm_data["input_mean"].astype(np.float32)  # (1, 7, 1, 1)
        self.input_std = norm_data["input_std"].astype(np.float32)  # (1, 7, 1, 1)

        # Load raw data if present
        if self.raw_path and os.path.exists(self.raw_path):
            raw_data = np.load(self.raw_path)
            self.raw_inputs = raw_data["inputs"]  # (150, 7, 100, 240)
            self.raw_targets = raw_data["targets"]  # (150, 15, 100, 240)
        else:
            self.raw_inputs = self.norm_inputs * self.input_std + self.input_mean
            self.raw_targets = self.norm_targets * self.target_std + self.target_mean

        self.num_days = self.norm_inputs.shape[0]
        # Memory cache for day-level inference results
        self._inference_cache: dict[int, dict[str, Any]] = {}

    def _find_file(self, candidates: list[Path]) -> str | None:
        for p in candidates:
            if p.exists():
                return str(p)
        return str(candidates[0]) if candidates else None

    @classmethod
    def get_instance(cls) -> OceanEmbedEngine:
        if cls._instance is None:
            cls._instance = OceanEmbedEngine()
        return cls._instance

    def predict_day(self, day_index: int) -> dict[str, Any]:
        """Run inference on the full 0.25 deg grid for a given day index (0..149)."""
        day_idx = max(0, min(int(day_index), self.num_days - 1))
        if day_idx in self._inference_cache:
            return self._inference_cache[day_idx]

        x_norm = torch.tensor(self.norm_inputs[day_idx : day_idx + 1], dtype=torch.float32)

        with torch.no_grad():
            pred_norm, embedding = self.model(x_norm)

        pred_celsius = pred_norm.numpy()[0] * self.target_std[0] + self.target_mean[0]  # (15, 100, 240)
        actual_celsius = self.raw_targets[day_idx]  # (15, 100, 240)
        embedding_arr = embedding.numpy()[0]  # (32, 100, 240)
        surface_inputs = self.raw_inputs[day_idx]  # (7, 100, 240)

        result = {
            "day_index": day_idx,
            "pred_celsius": pred_celsius,
            "actual_celsius": actual_celsius,
            "embedding": embedding_arr,
            "surface_inputs": surface_inputs,
            "ocean_mask": self.ocean_mask,
        }
        self._inference_cache[day_idx] = result
        return result


def get_engine() -> OceanEmbedEngine:
    return OceanEmbedEngine.get_instance()
