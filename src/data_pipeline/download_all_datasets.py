"""
Automated Data Pipeline for OceanEmbed (SIH26066).
Fetches ground truth Argo profiles (TEMP & PSAL) and multi-modal Copernicus Marine satellite data
(SST, SSS, SSH/SLA, and surface currents u/v).
"""
import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime

# Ensure data directories exist
RAW_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def fetch_argo_data(lon_min=82.0, lon_max=92.0, lat_min=8.0, lat_max=18.0,
                    depth_min=0, depth_max=1000,
                    date_min="2023-01-01", date_max="2023-03-31",
                    output_file="argo_profiles_bengal.nc"):
    """
    Fetch in-situ Argo profiling floats with both TEMP and PSAL (Salinity).
    """
    import argopy
    print(f"[Argo] Fetching profiles in region [{lon_min}, {lon_max}, {lat_min}, {lat_max}] from {date_min} to {date_max}...")
    
    fetcher = argopy.DataFetcher(src='erddap')
    ds = fetcher.region([lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, date_min, date_max]).to_xarray()
    
    out_path = os.path.join(RAW_DIR, output_file)
    ds.to_netcdf(out_path)
    print(f"[Argo] Successfully saved {ds.dims.get('N_POINTS', 'unknown')} profile points to {out_path}")
    return ds

def fetch_cmems_multimodal(lon_min=80.0, lon_max=95.0, lat_min=8.0, lat_max=20.0,
                           date_min="2023-01-01", date_max="2023-01-31",
                           output_file="cmems_multimodal.nc"):
    """
    Download SST (thetao), SSS (so), SSH (zos), and surface currents (uo, vo) from Copernicus Marine.
    """
    import copernicusmarine
    out_path = os.path.join(RAW_DIR, output_file)
    print(f"[CMEMS] Downloading SST, SSS, SSH, and currents for {date_min} to {date_max}...")
    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao", "so", "zos", "uo", "vo"],
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        start_datetime=date_min,
        end_datetime=date_max,
        minimum_depth=0.0,
        maximum_depth=1.0,
        output_directory=RAW_DIR,
        output_filename=output_file,
        overwrite=True
    )
    print(f"[CMEMS] Successfully downloaded satellite dataset to {out_path}")
    return xr.open_dataset(out_path)

if __name__ == "__main__":
    print("=== OceanEmbed Unified Data Downloader ===")
    # 1. Download CMEMS satellite data
    cmems_ds = fetch_cmems_multimodal()
    print("CMEMS variables available:", list(cmems_ds.data_vars))
    print("Download script ready for pipeline execution!")
