import copernicusmarine

def fetch_cmems_variable(dataset_id, variable, lon_min, lon_max, lat_min, lat_max,
                          date_min, date_max, depth_min, depth_max, output_filename):
    """Fetch a single CMEMS variable for a region and time range."""
    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=[variable],
        minimum_longitude=lon_min, maximum_longitude=lon_max,
        minimum_latitude=lat_min, maximum_latitude=lat_max,
        start_datetime=date_min, end_datetime=date_max,
        minimum_depth=depth_min, maximum_depth=depth_max,
        output_filename=output_filename,
        output_directory="data/raw"
    )

if __name__ == "__main__":
    fetch_cmems_variable(
        "cmems_mod_glo_phy_my_0.083deg_P1D-m", "thetao",
        80, 100, 5, 20, "2020-01-01", "2024-12-31", 0, 1,
        "sst_bay_of_bengal_2020_2024.nc"
    )