from argopy import DataFetcher

def fetch_argo_region(lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, date_min, date_max):
    """Fetch Argo float profiles for a given region and time range."""
    ds = DataFetcher().region(
        [lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, date_min, date_max]
    ).to_xarray()
    return ds

if __name__ == "__main__":
    ds = fetch_argo_region(80, 100, 5, 20, 0, 1000, '2020-01-01', '2024-12-31')
    ds.to_netcdf("data/raw/argo_bay_of_bengal_2020_2024.nc")
    print(f"Saved {ds.dims['N_POINTS']} points")