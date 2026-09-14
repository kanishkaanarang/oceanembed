"""Validation boundary for real OceanEmbed model and local-file integrations.

Replace the dummy generators with an adapter that returns the validated tables
below. Keeping this contract separate prevents model-specific code from leaking
into the Streamlit presentation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd


class ModelOutputError(ValueError):
    """Raised when an uploaded or model-generated result misses the UI contract."""


@dataclass(frozen=True)
class ReconstructionRequest:
    latitude: float
    longitude: float
    analysis_date: date


class OceanModel(Protocol):
    """Interface a trained model adapter should satisfy."""

    def predict_profile(self, request: ReconstructionRequest) -> pd.DataFrame: ...

    def predict_grid(self, request: ReconstructionRequest, depth_m: int, variable: str) -> pd.DataFrame: ...


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise ModelOutputError(f"'{column}' must contain numeric values only.")
    return values


def validate_profile(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a model-output CSV into the vertical-profile UI contract.

    Required columns are ``depth_m``, ``temperature_c``, and ``salinity_psu``.
    Optional uncertainty/reference columns receive transparent local defaults so
    basic model exports can be explored immediately.
    """
    required = {"depth_m", "temperature_c", "salinity_psu"}
    missing = required.difference(frame.columns)
    if missing:
        raise ModelOutputError("Profile CSV is missing: " + ", ".join(sorted(missing)))
    profile = frame.copy()
    for column in required:
        profile[column] = _numeric(profile, column)
    if (profile["depth_m"] < 0).any() or profile["depth_m"].duplicated().any():
        raise ModelOutputError("Depth values must be unique non-negative numbers.")
    profile = profile.sort_values("depth_m", ignore_index=True)
    temperature_width = 0.13 + 0.00035 * profile["depth_m"]
    salinity_width = 0.025 + 0.000055 * profile["depth_m"]
    defaults = {
        "temperature_lower_c": profile["temperature_c"] - temperature_width,
        "temperature_upper_c": profile["temperature_c"] + temperature_width,
        "argo_temperature_c": profile["temperature_c"],
        "salinity_lower_psu": profile["salinity_psu"] - salinity_width,
        "salinity_upper_psu": profile["salinity_psu"] + salinity_width,
        "argo_salinity_psu": profile["salinity_psu"],
        "confidence_pct": np.clip(97 - profile["depth_m"] * 0.024, 71, 97),
    }
    for column, default in defaults.items():
        profile[column] = _numeric(profile, column) if column in profile else default
    return profile[["depth_m", "temperature_c", "temperature_lower_c", "temperature_upper_c", "argo_temperature_c", "salinity_psu", "salinity_lower_psu", "salinity_upper_psu", "argo_salinity_psu", "confidence_pct"]]


def validate_grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the map-grid table expected by the map renderer."""
    required = {"longitude", "latitude", "value", "unit"}
    missing = required.difference(frame.columns)
    if missing:
        raise ModelOutputError("Grid output is missing: " + ", ".join(sorted(missing)))
    grid = frame.copy()
    for column in ("longitude", "latitude", "value"):
        grid[column] = _numeric(grid, column)
    if grid["unit"].isna().any():
        raise ModelOutputError("'unit' must be present for every grid cell.")
    return grid[["longitude", "latitude", "value", "unit"]]
