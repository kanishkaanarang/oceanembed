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
            self.root / "notebooks" / "results" / "oceanembed_cnn_best_v2.pt",
            self.root / "results" / "oceanembed_cnn_best_v2.pt",
            self.root / "notebooks" / "results" / "oceanembed_cnn_best.pt",
            self.root / "results" / "oceanembed_cnn_best.pt",
        ])
        self.norm_path = self._find_file([
            self.root / "training_arrays_v2_normalized" / "training_arrays_v2_normalized.npz",
            self.root / "data" / "processed" / "training_arrays_v2_normalized.npz",
        ])
        self.compact_path = self._find_file([
            self.root / "data" / "processed" / "oceanembed_reference_compact.npz",
            self.root / "training_arrays_v2_normalized" / "oceanembed_reference_compact.npz",
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
            self.model_loaded = True
        else:
            self.model.eval()
            self.model_loaded = False

        # Load normalization arrays
        if self.norm_path and os.path.exists(self.norm_path):
            norm_data = np.load(self.norm_path)
            self.norm_inputs = norm_data["inputs"]  # (150, 7, 100, 240)
            self.norm_targets = norm_data["targets"]  # (150, 15, 100, 240)
            self.ocean_mask = norm_data["ocean_mask"].astype(bool)  # (100, 240)
            self.target_mean = norm_data["target_mean"].astype(np.float32)  # (1, 15, 1, 1)
            self.target_std = norm_data["target_std"].astype(np.float32)  # (1, 15, 1, 1)
            self.input_mean = norm_data["input_mean"].astype(np.float32)  # (1, 7, 1, 1)
            self.input_std = norm_data["input_std"].astype(np.float32)  # (1, 7, 1, 1)

            if self.raw_path and os.path.exists(self.raw_path):
                raw_data = np.load(self.raw_path)
                self.raw_inputs = raw_data["inputs"]  # (150, 7, 100, 240)
                self.raw_targets = raw_data["targets"]  # (150, 15, 100, 240)
            else:
                self.raw_inputs = self.norm_inputs * self.input_std + self.input_mean
                self.raw_targets = self.norm_targets * self.target_std + self.target_mean

            self.num_days = self.norm_inputs.shape[0]
            self.dataset_mode = "full_satellite_dataset"
            self.is_compact = False

        elif self.compact_path and os.path.exists(self.compact_path):
            compact_data = np.load(self.compact_path)
            self.ocean_mask = compact_data["ocean_mask"].astype(bool)
            self.target_mean = compact_data["target_mean"].astype(np.float32)
            self.target_std = compact_data["target_std"].astype(np.float32)
            self.input_mean = compact_data["input_mean"].astype(np.float32)
            self.input_std = compact_data["input_std"].astype(np.float32)
            sample_inputs = compact_data["sample_inputs"].astype(np.float32)
            sample_targets = compact_data["sample_targets"].astype(np.float32)
            k = sample_inputs.shape[0]

            self.num_days = 150
            self.norm_inputs = np.zeros((150, 7, 100, 240), dtype=np.float32)
            self.norm_targets = np.zeros((150, 15, 100, 240), dtype=np.float32)
            for i in range(150):
                base_idx = i % k
                mod = float(np.sin((i / 150.0) * np.pi) * 0.12)
                self.norm_inputs[i] = sample_inputs[base_idx] + mod
                self.norm_targets[i] = sample_targets[base_idx] + mod

            self.raw_inputs = self.norm_inputs * self.input_std + self.input_mean
            self.raw_targets = self.norm_targets * self.target_std + self.target_mean
            self.dataset_mode = "compact_reference"
            self.is_compact = True

        else:
            self.dataset_mode = "synthetic_fallback"
            self.is_compact = True
            self.num_days = 150
            self.ocean_mask = np.ones((LAT_POINTS, LON_POINTS), dtype=bool)
            for lat_i, lat in enumerate(COMMON_LATS):
                for lon_i, lon in enumerate(COMMON_LONS):
                    if 72 <= lon <= 88 and 10 <= lat <= 26:
                        if (lat - 10) * 1.2 > (lon - 72) or (lat - 10) * 1.2 > (88 - lon):
                            self.ocean_mask[lat_i, lon_i] = False
            self.input_mean = np.array([14.2, 16.5, 0.05, 0.39, 0.13, 0.002, 0.006], dtype=np.float32).reshape(1, 7, 1, 1)
            self.input_std = np.array([14.5, 17.2, 0.09, 3.13, 2.86, 0.15, 0.13], dtype=np.float32).reshape(1, 7, 1, 1)
            self.target_mean = np.array([14.27, 14.20, 13.76, 13.38, 12.86, 12.0, 10.5, 9.2, 8.1, 7.2, 6.0, 4.8, 3.9, 3.2, 2.8], dtype=np.float32).reshape(1, 15, 1, 1)
            self.target_std = np.array([14.54, 14.47, 14.44, 14.37, 14.19, 13.8, 13.2, 12.5, 11.9, 11.2, 10.1, 8.8, 7.5, 6.2, 5.0], dtype=np.float32).reshape(1, 15, 1, 1)
            self.norm_inputs = np.zeros((150, 7, 100, 240), dtype=np.float32)
            self.norm_targets = np.zeros((150, 15, 100, 240), dtype=np.float32)
            sst_norm = (29.0 - self.input_mean[0, 0, 0, 0]) / self.input_std[0, 0, 0, 0]
            self.norm_inputs[:, 0, :, :] = sst_norm
            sss_norm = (34.5 - self.input_mean[0, 1, 0, 0]) / self.input_std[0, 1, 0, 0]
            self.norm_inputs[:, 1, :, :] = sss_norm
            self.raw_inputs = self.norm_inputs * self.input_std + self.input_mean
            self.raw_targets = self.norm_targets * self.target_std + self.target_mean

        # Memory cache for day-level inference results
        self._inference_cache: dict[int, dict[str, Any]] = {}

    def _find_file(self, candidates: list[Path]) -> str | None:
        for p in candidates:
            if p.exists():
                return str(p)
        return None

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
