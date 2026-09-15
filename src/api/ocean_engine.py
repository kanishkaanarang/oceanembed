"""
Oceanographic Calculation & Prediction Engine for OceanEmbed (SIH26066).
Extracts live multi-modal satellite data (SST, SSS, SSH, Winds, Currents),
runs 3D subsurface profile estimations using the trained OceanEmbedNet v2 CNN,
and computes seawater density, thermocline depth, and naval sound velocity profiles (SVP).
"""
import os
import math
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
if __package__ in (None, ""):
    # Preserve the existing `python src/api/ocean_engine.py` entry point.
    import sys
    sys.path.insert(0, str(ROOT))

from src.models.mission_metrics import (
    compute_acoustic_profile_diagnostics,
    compute_cyclone_heat_potential,
    compute_mission_metrics,
    mackenzie_sound_speed,
)

STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

LAT_MIN, LAT_MAX, LAT_STEP = 5.0, 30.0, 0.25
LON_MIN, LON_MAX, LON_STEP = 45.0, 105.0, 0.25
LAT_POINTS = int(round((LAT_MAX - LAT_MIN) / LAT_STEP))  # 100
LON_POINTS = int(round((LON_MAX - LON_MIN) / LON_STEP))  # 240
COMMON_LATS = np.linspace(LAT_MIN, LAT_MAX - LAT_STEP, LAT_POINTS)
COMMON_LONS = np.linspace(LON_MIN, LON_MAX - LON_STEP, LON_POINTS)

_cached_arrays: Optional[Dict[str, np.ndarray]] = None
_cached_weights: Optional[Dict[str, np.ndarray]] = None


def _date_to_day_index(value: str, day_count: int = 150) -> int:
    """Use the same 2023-02-01 origin and clipping as the Streamlit backend."""
    try:
        observation = pd.Timestamp(value)
        if pd.isna(observation):
            return 0
        offset = (observation.date() - pd.Timestamp("2023-02-01").date()).days
        return int(np.clip(offset, 0, max(0, day_count - 1)))
    except (ValueError, TypeError, OverflowError):
        return 0


def load_training_arrays() -> Optional[Dict[str, np.ndarray]]:
    """Loads and caches the 150-day Copernicus normalized satellite and reanalysis arrays."""
    global _cached_arrays
    if _cached_arrays is not None:
        return _cached_arrays

    candidates = [
        ROOT / "data" / "processed" / "training_arrays_v2_normalized.npz",
        ROOT / "training_arrays_v2_normalized" / "training_arrays_v2_normalized.npz",
        ROOT / "data" / "processed" / "oceanembed_reference_compact.npz",
    ]

    for path in candidates:
        if path.exists():
            try:
                npz = np.load(path)
                # Check whether full arrays or compact arrays
                if "inputs" in npz and "targets" in npz:
                    _cached_arrays = {
                        "inputs": npz["inputs"],
                        "targets": npz["targets"],
                        "ocean_mask": npz["ocean_mask"].astype(bool),
                        "input_mean": npz["input_mean"].astype(np.float32),
                        "input_std": npz["input_std"].astype(np.float32),
                        "target_mean": npz["target_mean"].astype(np.float32),
                        "target_std": npz["target_std"].astype(np.float32),
                    }
                    return _cached_arrays
            except Exception:
                continue

    return None

def get_oceanembed_net_weights() -> Optional[Dict[str, np.ndarray]]:
    """Loads OceanEmbedNet v2 CNN weights directly from the checkpoint archive."""
    global _cached_weights
    if _cached_weights is not None:
        return _cached_weights

    candidates = [
        ROOT / "results" / "oceanembed_cnn_best_v2.pt",
        ROOT / "notebooks" / "results" / "oceanembed_cnn_best_v2.pt",
        ROOT / "results" / "oceanembed_cnn_best.pt",
    ]

    checkpoint_file = None
    for p in candidates:
        if p.exists():
            checkpoint_file = p
            break

    if checkpoint_file is None:
        return None

    try:
        with zipfile.ZipFile(checkpoint_file, "r") as z:
            # Locate prefix inside zip
            prefix = ""
            for name in z.namelist():
                if name.endswith("data/0"):
                    prefix = name.split("data/0")[0]
                    break

            w0 = np.frombuffer(z.read(f"{prefix}data/0"), dtype=np.float32).reshape(16, 7, 3, 3)
            b0 = np.frombuffer(z.read(f"{prefix}data/1"), dtype=np.float32)
            w1 = np.frombuffer(z.read(f"{prefix}data/2"), dtype=np.float32).reshape(32, 16, 3, 3)
            b1 = np.frombuffer(z.read(f"{prefix}data/3"), dtype=np.float32)
            w2 = np.frombuffer(z.read(f"{prefix}data/4"), dtype=np.float32).reshape(16, 32, 3, 3)
            b2 = np.frombuffer(z.read(f"{prefix}data/5"), dtype=np.float32)
            w3 = np.frombuffer(z.read(f"{prefix}data/6"), dtype=np.float32).reshape(15, 16, 3, 3)
            b3 = np.frombuffer(z.read(f"{prefix}data/7"), dtype=np.float32)

            _cached_weights = {
                "w0": w0, "b0": b0,
                "w1": w1, "b1": b1,
                "w2": w2, "b2": b2,
                "w3": w3, "b3": b3,
            }
            return _cached_weights
    except Exception:
        return None

def _conv2d_fast(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized 2D Convolution in NumPy with 3x3 kernel and unit padding."""
    C_in, H, W = x.shape
    C_out, _, kH, kW = w.shape
    pad_h, pad_w = kH // 2, kW // 2
    x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")

    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(x_padded, (C_in, kH, kW))[0]  # Shape: (H, W, C_in, kH, kW)
    out = np.tensordot(windows, w, axes=((2, 3, 4), (1, 2, 3))) + b  # Shape: (H, W, C_out)
    return np.transpose(out, (2, 0, 1))  # Shape: (C_out, H, W)

def run_numpy_cnn_forward(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executes full OceanEmbedNet v2 forward pass:
    Input: (7, 100, 240) normalized satellite surface tensor
    Returns: (pred_norm (15, 100, 240), embedding (32, 100, 240))
    """
    weights = get_oceanembed_net_weights()
    if weights is None:
        raise RuntimeError("Model checkpoint weights could not be loaded.")

    # Layer 0: Conv 7 -> 16 + ReLU
    h0 = np.maximum(0.0, _conv2d_fast(x, weights["w0"], weights["b0"]))
    # Layer 1: Conv 16 -> 32 + ReLU (Latent Embedding)
    embed = np.maximum(0.0, _conv2d_fast(h0, weights["w1"], weights["b1"]))
    # Layer 2: Conv 32 -> 16 + ReLU
    h2 = np.maximum(0.0, _conv2d_fast(embed, weights["w2"], weights["b2"]))
    # Layer 3: Conv 16 -> 15 (Linear output)
    pred_norm = _conv2d_fast(h2, weights["w3"], weights["b3"])

    return pred_norm, embed

def compute_seawater_density(temp_c: float, sal_psu: float, depth_m: float) -> float:
    """
    Approximation of UNESCO 1980 Equation of State for Seawater Density (rho) in kg/m^3.
    Incorporates temperature, salinity, and hydrostatic pressure compression.
    """
    p = depth_m * 0.1007  # Pressure in dbar
    # Density of standard pure water
    rho_pure = (999.842594 + 6.793952e-2 * temp_c - 9.095290e-3 * temp_c**2 +
                1.001685e-4 * temp_c**3 - 1.120083e-6 * temp_c**4 + 6.536332e-9 * temp_c**5)

    A = 8.24493e-1 - 4.0899e-3 * temp_c + 7.6438e-5 * temp_c**2 - 8.2467e-7 * temp_c**3 + 5.3875e-9 * temp_c**4
    B = -5.72466e-3 + 1.0227e-4 * temp_c - 1.6546e-6 * temp_c**2
    C = 4.8314e-4
    rho_0 = rho_pure + A * sal_psu + B * (sal_psu**1.5) + C * (sal_psu**2)

    K0 = (19652.21 + 148.4206 * temp_c - 2.327105 * temp_c**2 + 1.360477e-2 * temp_c**3 - 5.155288e-5 * temp_c**4)
    secant_bulk = K0 + (54.6746 - 0.603459 * temp_c + 1.09987e-2 * temp_c**2) * sal_psu
    rho = rho_0 / max(0.8, (1.0 - p / max(secant_bulk, 1e-4)))
    return round(float(rho), 3)

def compute_sound_velocity(temp_c: float, sal_psu: float, depth_m: float) -> float:
    """
    Mackenzie (1981) formula for Sound Velocity in seawater (m/s).
    Crucial for naval underwater acoustics, sonar detection range, and acoustic propagation.
    """
    return round(float(mackenzie_sound_speed(temp_c, sal_psu, depth_m)), 2)

def detect_thermocline(depths: List[float], temps: List[float]) -> float:
    """Locates the depth of maximum vertical temperature gradient (-dT/dz)."""
    if len(depths) < 2:
        return 75.0

    max_gradient = 0.0
    thermocline_d = 75.0

    for i in range(len(depths) - 1):
        dz = depths[i + 1] - depths[i]
        if dz > 0:
            dt = temps[i] - temps[i + 1]  # positive when cooling with depth
            gradient = dt / dz
            if gradient > max_gradient:
                max_gradient = gradient
                thermocline_d = (depths[i] + depths[i + 1]) / 2.0

    return round(float(thermocline_d), 1)

def detect_sofar_channel(depths: List[float], sound_speeds: List[float]) -> float:
    """Locates the minimum sound speed axis (SOFAR channel / acoustic waveguide)."""
    if not sound_speeds:
        return 800.0
    min_idx = int(np.argmin(sound_speeds))
    return round(float(depths[min_idx]), 1)

def _find_nearest_grid_indices(lat: float, lon: float) -> Tuple[int, int]:
    """Maps continuous lat/lon coordinates to grid array index (0..99, 0..239)."""
    lat_idx = int(np.clip(np.round((lat - LAT_MIN) / LAT_STEP), 0, LAT_POINTS - 1))
    lon_idx = int(np.clip(np.round((lon - LON_MIN) / LON_STEP), 0, LON_POINTS - 1))
    return lat_idx, lon_idx

def extract_satellite_surface(lat: float, lon: float, date: str = "2023-01-05") -> Dict[str, Any]:
    """
    Extracts multi-modal surface observations for a location from the preprocessed
    historical reanalysis arrays, or provides physical fallback.
    """
    data = load_training_arrays()
    lat_idx, lon_idx = _find_nearest_grid_indices(lat, lon)

    day_idx = _date_to_day_index(date, len(data["inputs"]) if data is not None else 150)

    if data is not None and "inputs" in data:
        # Array shape: (150, 7, 100, 240)
        day_inputs = data["inputs"][day_idx]
        in_std = data["input_std"][0]
        in_mean = data["input_mean"][0]
        unnorm_inputs = day_inputs * in_std + in_mean

        sst = float(unnorm_inputs[0, lat_idx, lon_idx])
        sss = float(unnorm_inputs[1, lat_idx, lon_idx])
        ssh = float(unnorm_inputs[2, lat_idx, lon_idx])
        uwnd = float(unnorm_inputs[3, lat_idx, lon_idx])
        vwnd = float(unnorm_inputs[4, lat_idx, lon_idx])
        ucurr = float(unnorm_inputs[5, lat_idx, lon_idx])
        vcurr = float(unnorm_inputs[6, lat_idx, lon_idx])

        return {
            "sst": round(sst, 2),
            "sss": round(sss, 2),
            "ssh": round(ssh, 3),
            "wind_u": round(uwnd, 2),
            "wind_v": round(vwnd, 2),
            "current_u": round(ucurr, 3),
            "current_v": round(vcurr, 3),
            "grid_lat": round(float(COMMON_LATS[lat_idx]), 2),
            "grid_lon": round(float(COMMON_LONS[lon_idx]), 2),
            "source": "copernicus_reanalysis_arrays"
        }

    # Physical fallback for North Indian Ocean
    return {
        "sst": 28.6,
        "sss": 33.2,
        "ssh": 0.08,
        "wind_u": 3.2,
        "wind_v": -1.5,
        "current_u": 0.12,
        "current_v": -0.06,
        "grid_lat": lat,
        "grid_lon": lon,
        "source": "climatology_fallback"
    }

def predict_full_column_v2(lat: float, lon: float, date: str = "2023-01-05", depths: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Computes full 3D vertical ocean reconstruction (Temperature, Salinity, Density, Sound Speed)
    using the trained OceanEmbedNet v2 CNN model.
    """
    if depths is None:
        depths = STANDARD_DEPTHS
    requested_depths = np.asarray(depths, dtype=float)
    if (requested_depths.ndim != 1 or requested_depths.size == 0
            or not np.all(np.isfinite(requested_depths))
            or np.any(requested_depths < 0) or np.any(requested_depths > STANDARD_DEPTHS[-1])):
        raise ValueError("Depths must be a nonempty list of finite values between 0 and 1000 m.")
    depths = requested_depths.tolist()

    lat_idx, lon_idx = _find_nearest_grid_indices(lat, lon)
    sat = extract_satellite_surface(lat, lon, date)
    sst = sat["sst"]
    sss = sat["sss"]

    data = load_training_arrays()
    weights = get_oceanembed_net_weights()

    temps: List[float] = []
    sals: List[float] = []

    model_used = "oceanembed_cnn_v2"

    day_idx = _date_to_day_index(date, len(data["inputs"]) if data is not None else 150)

    if data is not None and weights is not None:
        # Run CNN inference on the chosen day
        x_norm = data["inputs"][day_idx] # (7, 100, 240)
        pred_norm, _ = run_numpy_cnn_forward(x_norm)
        # Un-normalize
        pred_c = pred_norm * data["target_std"][0] + data["target_mean"][0] # (15, 100, 240)
        temp_profile = pred_c[:, lat_idx, lon_idx] # (15,)

        for d in depths:
            t_val = round(float(np.interp(d, STANDARD_DEPTHS, temp_profile)), 2)
            # Salinity profile derivation from standard T-S relation
            s_val = round(sss + (34.90 - sss) * (1.0 - math.exp(-d / 150.0)), 2)
            temps.append(t_val)
            sals.append(s_val)
    else:
        # Physics-informed analytical transition
        model_used = "analytical_ocean_physics"
        thermocline_depth = 75.0
        deep_ocean_temp = 4.2
        for d in depths:
            sig = 1.0 / (1.0 + math.exp((d - thermocline_depth) / 50.0))
            t_val = deep_ocean_temp + (sst - deep_ocean_temp) * sig
            s_val = sss + (34.90 - sss) * (1.0 - math.exp(-d / 150.0))
            temps.append(round(max(2.5, min(32.0, t_val)), 2))
            sals.append(round(max(28.0, min(36.5, s_val)), 2))

    # Compute physical derived properties
    profile_tiers = []
    densities: List[float] = []
    sound_speeds: List[float] = []
    inversion_detected = False

    for i, d in enumerate(depths):
        t = temps[i]
        s = sals[i]
        rho = compute_seawater_density(t, s, d)
        c = compute_sound_velocity(t, s, d)

        if densities and rho < densities[-1]:
            inversion_detected = True

        densities.append(rho)
        sound_speeds.append(c)

        profile_tiers.append({
            "depth": d,
            "temperature": t,
            "salinity": s,
            "density": rho,
            "sound_speed": c
        })

    thermocline_d = detect_thermocline(depths, temps)
    sofar_axis_d = detect_sofar_channel(depths, sound_speeds)

    return {
        "location": {"latitude": lat, "longitude": lon},
        "date": date,
        "model_provenance": model_used,
        "surface_satellite_inputs": sat,
        "thermocline_depth_estimate_m": thermocline_d,
        "sofar_axis_depth_m": sofar_axis_d,
        "hydrostatic_stability": "inversion_warning" if inversion_detected else "stable",
        "mission_intelligence": compute_mission_metrics(depths, temps, sound_speeds),
        "profile": profile_tiers
    }

def predict_full_column(lat: float, lon: float, date: str = "2023-02-01", depths=None):
    """Serve the same v3 T/S checkpoint and physical column as Streamlit."""
    from datetime import date as Date
    from backend import reconstruct, get_surface_conditions
    from src.models.serving_engine import STANDARD_DEPTHS as v3_depths
    selected = np.asarray(v3_depths if depths is None else depths, dtype=float)
    if (selected.ndim != 1 or not selected.size or not np.isfinite(selected).all()
            or (selected < 0).any() or (selected > v3_depths[-1]).any()):
        raise ValueError("V3 supports finite depths between 0 and 900 m; no extrapolation is performed.")
    day = Date.fromisoformat(date)
    frame = reconstruct(lat, lon, day)
    temperature = np.interp(selected, frame.depth_m, frame.temperature_c)
    salinity = np.interp(selected, frame.depth_m, frame.salinity_psu)
    from backend import calc_density, calc_sound_speed
    rho = calc_density(temperature, salinity)
    sound = calc_sound_speed(temperature, salinity, selected)
    sat = get_surface_conditions(lat, lon, day)
    sat.update(ssh=sat['sla'], wind_u=sat['uwnd'], wind_v=sat['vwnd'],
               current_u=sat['ucurr'], current_v=sat['vcurr'],
               grid_lat=sat['latitude'], grid_lon=sat['longitude'])
    ordered = np.argsort(selected)
    mission = compute_mission_metrics(frame.depth_m, frame.temperature_c, frame.sound_speed_m_s,
                                      salinities=frame.salinity_psu, densities=frame.density_kg_m3)
    return dict(location=dict(latitude=lat, longitude=lon), date=date,
        model_provenance="oceanembed_v3_se_resnet", surface_satellite_inputs=sat,
        thermocline_depth_estimate_m=detect_thermocline(frame.depth_m.to_list(), frame.temperature_c.to_list()),
        sofar_axis_depth_m=detect_sofar_channel(frame.depth_m.to_list(), frame.sound_speed_m_s.to_list()),
        hydrostatic_stability="inversion_warning" if (np.diff(rho[ordered]) < 0).any() else "stable",
        mission_intelligence=mission,
        profile=[dict(depth=float(d), temperature=float(ti), salinity=float(si),
                      density=float(ri), sound_speed=float(ci))
                 for d, ti, si, ri, ci in zip(selected, temperature, salinity, rho, sound)])


if __name__ == "__main__":
    test_res = predict_full_column(15.0, 88.0, "2023-01-05")
    print(f"Model: {test_res['model_provenance']}")
    print(f"Thermocline Depth: {test_res['thermocline_depth_estimate_m']} m")
    print(f"SOFAR Sound Channel Axis: {test_res['sofar_axis_depth_m']} m")
    print("Top 3 Tiers:")
    for tier in test_res["profile"][:3]:
        print(f"  {tier['depth']:g}m: Temp={tier['temperature']}°C, Sal={tier['salinity']}PSU, Density={tier['density']}kg/m³, Sound={tier['sound_speed']}m/s")
