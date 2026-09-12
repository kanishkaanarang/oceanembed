"""
Dataset Preprocessing Pipeline for OceanEmbed (SIH26066).
Extracts 3D profiles from Copernicus GLORYS reanalysis,
interpolates to official standard oceanic depth tiers,
and creates the enriched multi-modal training dataset.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d

RAW_NETCDF = os.path.join(os.path.dirname(__file__), "../../data/raw/glorys_3d_profiles_2023.nc")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "../../data/processed/multimodal_training_dataset.parquet")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "../../data/processed/multimodal_training_dataset.csv")

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 900]

def build_training_table():
    print(f"Loading 3D NetCDF dataset from {RAW_NETCDF}...")
    ds = xr.open_dataset(RAW_NETCDF)
    
    depths = ds.depth.values
    times = ds.time.values
    lats = ds.latitude.values
    lons = ds.longitude.values
    
    print(f"Dataset covers: {len(times)} days, {len(depths)} depths, {len(lats)} lats, {len(lons)} lons")
    
    records = []
    
    for t_idx, t_val in enumerate(times):
        doy = pd.Timestamp(t_val).dayofyear
        ds_t = ds.isel(time=t_idx)
        
        # Surface variables (depth index 0 ~ 0.5m)
        sst_map = ds_t["thetao"].isel(depth=0).values
        sss_map = ds_t["so"].isel(depth=0).values
        ssh_map = ds_t["zos"].values
        uo_map = ds_t["uo"].isel(depth=0).values
        vo_map = ds_t["vo"].isel(depth=0).values
        
        # 3D temperature and salinity
        temp_3d = ds_t["thetao"].values
        sal_3d = ds_t["so"].values
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                sst = sst_map[i, j]
                # Skip land cells (NaN)
                if np.isnan(sst):
                    continue
                    
                sss = sss_map[i, j]
                ssh = ssh_map[i, j]
                uo = uo_map[i, j]
                vo = vo_map[i, j]
                
                # Full vertical columns
                col_temp = temp_3d[:, i, j]
                col_sal = sal_3d[:, i, j]
                
                # Check for valid column data
                valid_mask = ~np.isnan(col_temp) & ~np.isnan(col_sal)
                if np.sum(valid_mask) < 10:
                    continue
                    
                valid_depths = depths[valid_mask]
                valid_temps = col_temp[valid_mask]
                valid_sals = col_sal[valid_mask]
                
                # Interpolate to official 15 depth tiers
                f_temp = interp1d(valid_depths, valid_temps, bounds_error=False, fill_value="extrapolate")
                f_sal = interp1d(valid_depths, valid_sals, bounds_error=False, fill_value="extrapolate")
                
                interp_temps = f_temp(TARGET_DEPTHS)
                interp_sals = f_sal(TARGET_DEPTHS)
                
                rec = {
                    "lat": round(float(lat), 3),
                    "lon": round(float(lon), 3),
                    "day_of_year": int(doy),
                    "doy_sin": round(float(np.sin(2 * np.pi * doy / 365.25)), 4),
                    "doy_cos": round(float(np.cos(2 * np.pi * doy / 365.25)), 4),
                    "sst": round(float(sst), 3),
                    "sss": round(float(sss), 3),
                    "ssh": round(float(ssh), 3),
                    "current_u": round(float(uo), 4),
                    "current_v": round(float(vo), 4),
                    "current_speed": round(float(np.hypot(uo, vo)), 4),
                }
                
                # Add 15 temperature targets
                for d, val in zip(TARGET_DEPTHS, interp_temps):
                    rec[f"temp_{d}m"] = round(float(val), 3)
                    
                # Add 15 salinity targets
                for d, val in zip(TARGET_DEPTHS, interp_sals):
                    rec[f"sal_{d}m"] = round(float(val), 3)
                    
                records.append(rec)
                
    df = pd.DataFrame(records)
    print(f"Generated {len(df)} complete 3D ocean profiles!")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    # Also save first 5000 as CSV for easy inspection
    df.head(5000).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved parquet to {OUTPUT_PATH} and preview CSV to {OUTPUT_CSV}")
    return df

if __name__ == "__main__":
    df = build_training_table()
    print("Columns:", list(df.columns)[:15], "...")
    print("Sample row:")
    print(df.iloc[0])
