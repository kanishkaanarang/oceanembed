"""Pointwise features only: spatial derivatives require neighboring ocean cells."""
import numpy as np
import pandas as pd


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["lat", "lon", "doy_sin", "doy_cos", "sst", "sss", "ssh", "current_u", "current_v"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing surface inputs: {sorted(missing)}")
    x = frame[required].astype(float).copy()
    u, v = x.current_u, x.current_v
    x["current_speed"] = np.hypot(u, v)
    x["current_ke"] = .5*(u*u + v*v)
    x["current_uv"] = u*v
    x["sst_ssh"] = x.sst*x.ssh
    x["sss_ssh"] = x.sss*x.ssh
    x["sst_sss"] = (x.sst-26)*(x.sss-35)
    x["warm_excess"] = np.maximum(x.sst-26, 0)
    x["f_coriolis"] = 2*7.292115e-5*np.sin(np.deg2rad(x.lat))
    x["latitude_sin"] = np.sin(np.deg2rad(x.lat))
    x["longitude_sin"] = np.sin(np.deg2rad(x.lon))
    x["longitude_cos"] = np.cos(np.deg2rad(x.lon))
    for name in ("sst", "sss", "ssh"):
        x[f"{name}_season_sin"] = x[name]*x.doy_sin
        x[f"{name}_season_cos"] = x[name]*x.doy_cos
    if ("wind_u" in frame) != ("wind_v" in frame):
        raise ValueError("Supply both wind_u and wind_v, or neither.")
    if "wind_u" in frame:
        wu, wv = frame.wind_u.astype(float), frame.wind_v.astype(float)
        speed = np.hypot(wu, wv)
        x["wind_u"], x["wind_v"], x["wind_speed"] = wu, wv, speed
        # Constant-drag bulk proxy, not a measured stress field.
        x["tau_x_proxy"] = 1.225*1.3e-3*speed*wu
        x["tau_y_proxy"] = 1.225*1.3e-3*speed*wv
        x["wind_current_dot"] = wu*u + wv*v
        x["wind_current_cross"] = wu*v - wv*u
        x["wind_mixing_proxy"] = speed**3
    # Derived upstream on a collocated grid, never estimated from unordered rows.
    for name in ("wind_stress_curl", "ekman_pumping", "geostrophic_vorticity",
                 "sst_gradient_magnitude", "sss_gradient_magnitude", "thermal_advection",
                 "sla", "mld_climatology", "mixed_layer_heat_proxy"):
        if name in frame:
            x[name] = frame[name].astype(float)
    return x.replace([np.inf, -np.inf], np.nan)


def fit_preprocessor(x):
    values = x.to_numpy(dtype=np.float64)
    if np.any(np.all(~np.isfinite(values), axis=0)):
        raise ValueError("A training feature has no finite observations.")
    median = np.nanmedian(values, axis=0)
    filled = np.where(np.isfinite(values), values, median)
    return {"columns": list(x.columns), "median": median.tolist(),
            "mean": filled.mean(0).tolist(), "scale": np.maximum(filled.std(0), 1e-6).tolist()}


def transform_features(x, state):
    missing = set(state["columns"]) - set(x.columns)
    if missing:
        raise ValueError(f"Checkpoint requires missing features: {sorted(missing)}")
    values = x[state["columns"]].to_numpy(dtype=np.float64)
    mask = ~np.isfinite(values)
    values = np.where(mask, np.asarray(state["median"]), values)
    values = (values - np.asarray(state["mean"])) / np.asarray(state["scale"])
    return np.concatenate([values, mask.astype(float)], axis=1).astype(np.float32)
