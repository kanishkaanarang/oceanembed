"""
Oceanographic Calculation & Prediction Engine for OceanEmbed (SIH26066).
Extracts live multi-modal satellite data (SST, SSS, SSH, Currents),
runs profile predictions, and computes seawater density and sound velocity (SVP).
"""
import os
import math
import numpy as np
import pandas as pd
import xarray as xr

STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
CMEMS_PATH = os.path.join(os.path.dirname(__file__), "../../data/raw/cmems_multimodal.nc")

_cmems_ds = None

def get_cmems_dataset():
    """Load and cache the CMEMS satellite NetCDF dataset."""
    global _cmems_ds
    if _cmems_ds is None and os.path.exists(CMEMS_PATH):
        _cmems_ds = xr.open_dataset(CMEMS_PATH)
    return _cmems_ds

def extract_satellite_surface(lat: float, lon: float, date: str = "2023-01-05"):
    """
    Query the nearest multi-modal satellite observations from CMEMS.
    Returns: dict with sst (°C), sss (PSU), ssh (m), u (m/s), v (m/s)
    """
    ds = get_cmems_dataset()
    if ds is None:
        # Fallback values for Bay of Bengal if NetCDF is not loaded
        return {
            "sst": 28.5,
            "sss": 32.8,
            "ssh": 0.12,
            "current_u": 0.15,
            "current_v": -0.08,
            "source": "climatology_fallback"
        }
    
    try:
        # Select nearest point
        point = ds.sel(latitude=lat, longitude=lon, method="nearest").isel(depth=0)
        # Select nearest time if available
        if "time" in point.dims and point.sizes["time"] > 1:
            point = point.sel(time=date, method="nearest")
        elif "time" in point.dims:
            point = point.isel(time=0)
            
        sst = float(point["thetao"].values) if "thetao" in point else 28.5
        sss = float(point["so"].values) if "so" in point else 32.8
        ssh = float(point["zos"].values) if "zos" in point else 0.12
        uo = float(point["uo"].values) if "uo" in point else 0.1
        vo = float(point["vo"].values) if "vo" in point else -0.05

        # Handle NaN values near land
        if np.isnan(sst): sst = 28.5
        if np.isnan(sss): sss = 32.8
        if np.isnan(ssh): ssh = 0.12
        if np.isnan(uo): uo = 0.1
        if np.isnan(vo): vo = -0.05

        return {
            "sst": round(sst, 3),
            "sss": round(sss, 3),
            "ssh": round(ssh, 3),
            "current_u": round(uo, 3),
            "current_v": round(vo, 3),
            "source": "cmems_satellite_netcdf"
        }
    except Exception as e:
        return {
            "sst": 28.5,
            "sss": 32.8,
            "ssh": 0.12,
            "current_u": 0.1,
            "current_v": -0.05,
            "source": f"fallback_due_to_{str(e)}"
        }

def compute_seawater_density(temp_c: float, sal_psu: float, depth_m: float) -> float:
    """
    Approximation of UNESCO seawater density rho(T, S, P) in kg/m^3.
    Standard equation of state for physical stability analysis.
    """
    # Pressure approximation from depth (1 dbar ~ 1 meter)
    p = depth_m * 0.1007
    
    # Pure water density at atmospheric pressure
    rho_pure = (999.842594 + 6.793952e-2 * temp_c - 9.095290e-3 * temp_c**2 +
                1.001685e-4 * temp_c**3 - 1.120083e-6 * temp_c**4 + 6.536332e-9 * temp_c**5)
    
    # Salinity influence
    A = 8.24493e-1 - 4.0899e-3 * temp_c + 7.6438e-5 * temp_c**2 - 8.2467e-7 * temp_c**3 + 5.3875e-9 * temp_c**4
    B = -5.72466e-3 + 1.0227e-4 * temp_c - 1.6546e-6 * temp_c**2
    C = 4.8314e-4
    rho_0 = rho_pure + A * sal_psu + B * (sal_psu**1.5) + C * (sal_psu**2)
    
    # Pressure compressibility factor
    K0 = (19652.21 + 148.4206 * temp_c - 2.327105 * temp_c**2 + 1.360477e-2 * temp_c**3 - 5.155288e-5 * temp_c**4)
    secant_bulk = K0 + (54.6746 - 0.603459 * temp_c + 1.09987e-2 * temp_c**2) * sal_psu
    rho = rho_0 / (1.0 - p / max(secant_bulk, 1e-6))
    return round(float(rho), 3)

def compute_sound_velocity(temp_c: float, sal_psu: float, depth_m: float) -> float:
    """
    Mackenzie (1981) formula for Sound Velocity in seawater (m/s).
    Crucial for naval acoustics and sonar propagation modeling.
    c = 1448.96 + 4.591*T - 5.304e-2*T^2 + 2.374e-4*T^3 + 1.340*(S-35) + 1.630e-2*D + 1.675e-7*D^2 - 1.025e-2*T*(S-35) - 7.139e-13*T*D^3
    """
    T = temp_c
    S = sal_psu
    D = depth_m
    c = (1448.96 + 4.591 * T - 5.304e-2 * (T**2) + 2.374e-4 * (T**3) +
         1.340 * (S - 35.0) + 1.630e-2 * D + 1.675e-7 * (D**2) -
         1.025e-2 * T * (S - 35.0) - 7.139e-13 * T * (D**3))
    return round(float(c), 2)

def predict_full_column(lat: float, lon: float, date: str = "2023-01-05", depths=None):
    """
    Generates a full subsurface column estimation combining surface satellite inputs
    with vertical ocean physics (Temperature, Salinity, Density, and Sound Speed).
    """
    if depths is None:
        depths = STANDARD_DEPTHS
        
    sat = extract_satellite_surface(lat, lon, date)
    sst = sat["sst"]
    sss = sat["sss"]
    ssh = sat["ssh"]
    
    # Calculate day of year
    doy = pd.Timestamp(date).dayofyear if date else 15
    
    # Load trained Multi-Task Model if available
    model_path = os.path.join(os.path.dirname(__file__), "../../results/multitask_ocean_model.pkl")
    ml_predicted = False
    
    if os.path.exists(model_path):
        try:
            import joblib
            model = joblib.load(model_path)
            doy_sin = math.sin(2 * math.pi * doy / 365.25)
            doy_cos = math.cos(2 * math.pi * doy / 365.25)
            u = sat.get("current_u", 0.0)
            v = sat.get("current_v", 0.0)
            spd = math.hypot(u, v)
            
            feat_df = pd.DataFrame([{
                "lat": lat, "lon": lon,
                "doy_sin": doy_sin, "doy_cos": doy_cos,
                "sst": sst, "sss": sss, "ssh": ssh,
                "current_u": u, "current_v": v, "current_speed": spd
            }])
            raw_pred = model.predict(feat_df)[0]
            # First 15 outputs are temp, next 15 are salinity
            model_target_depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 900]
            temp_dict = {d: raw_pred[idx] for idx, d in enumerate(model_target_depths)}
            sal_dict = {d: raw_pred[15 + idx] for idx, d in enumerate(model_target_depths)}
            ml_predicted = True
        except Exception:
            ml_predicted = False

    points = []
    previous_density = None
    inversion_detected = False
    
    for d in depths:
        if ml_predicted and d in temp_dict and d in sal_dict:
            temp = round(float(temp_dict[d]), 2)
            sal = round(float(sal_dict[d]), 2)
        else:
            # Physics-informed fallback transition
            sigmoid = 1.0 / (1.0 + math.exp((d - thermocline_depth) / 45.0))
            temp = deep_ocean_temp + (sst - deep_ocean_temp) * sigmoid
            temp = round(max(3.0, min(32.0, temp)), 2)
            sal = sss + (34.85 - sss) * (1.0 - math.exp(-d / 120.0))
            sal = round(max(28.0, min(36.5, sal)), 2)
        
        # Calculate derived physics
        rho = compute_seawater_density(temp, sal, d)
        sound_speed = compute_sound_velocity(temp, sal, d)
        
        if previous_density is not None and rho < previous_density:
            inversion_detected = True
        previous_density = rho
        
        points.append({
            "depth": d,
            "temperature": temp,
            "salinity": sal,
            "density": rho,
            "sound_speed": sound_speed
        })
        
    return {
        "location": {"latitude": lat, "longitude": lon},
        "date": date,
        "surface_satellite_inputs": sat,
        "thermocline_depth_estimate_m": round(thermocline_depth, 1),
        "hydrostatic_stability": "stable" if not inversion_detected else "inversion_warning",
        "profile": points
    }

if __name__ == "__main__":
    res = predict_full_column(15.0, 88.0, "2023-01-05")
    print("Test Prediction for Lat 15N, Lon 88E:")
    print("Surface inputs:", res["surface_satellite_inputs"])
    print("Top 3 depth tiers:", res["profile"][:3])
    print("Deepest tier (1000m):", res["profile"][-1])
