def fetch_incois_gridded_argo(date_str, lat_min=5, lat_max=30, lon_min=45, lon_max=105, depth_min=5, depth_max=1000):
    """Fetch INCOIS gridded Argo TEMP for a given date (monthly product, use the 15th of the month)."""
    import requests, xarray as xr
    url = (
        f"https://erddap.incois.gov.in/erddap/griddap/incois_argo_mnt_VAM.nc?"
        f"TEMP[({date_str}T00:00:00Z)][({depth_min}):({depth_max})]"
        f"[({lat_min}):({lat_max})][({lon_min}):({lon_max})]"
    )
    r = requests.get(url, timeout=60, verify=False)
    path = f"data/raw/incois_{date_str}.nc"
    with open(path, "wb") as f:
        f.write(r.content)
    ds = xr.open_dataset(path).rename({"ZAX": "depth"})
    return ds