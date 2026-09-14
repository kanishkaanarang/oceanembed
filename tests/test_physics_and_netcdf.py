"""Tests for CF-1.8 NetCDF export and physics integrity diagnostics."""
import os
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

from src.models.mission_metrics import (
    compute_physics_integrity,
    export_profile_to_netcdf,
    compute_mission_metrics,
    compute_marine_heatwave_metrics,
    compute_naval_sonar_tactics,
    generate_mission_briefing,
)


class PhysicsIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
        self.temps = [29.5, 29.5, 29.4, 29.3, 29.0, 26.5, 22.0, 18.0, 15.0, 13.0, 11.0, 9.0, 6.5, 5.2, 4.3]
        self.sals = [32.0, 32.1, 32.2, 33.5, 34.2, 34.6, 34.8, 34.9, 35.0, 35.0, 35.0, 34.9, 34.8, 34.8, 34.7]
        self.rhos = [1019.5, 1019.6, 1019.7, 1020.9, 1022.0, 1024.1, 1025.8, 1026.9, 1027.8, 1028.3, 1028.8, 1029.5, 1030.5, 1031.4, 1032.1]

    def test_stable_water_column(self):
        res = compute_physics_integrity(self.depths, self.temps, self.sals, self.rhos)
        self.assertTrue(res["is_stable"])
        self.assertEqual(res["inversion_count"], 0)
        self.assertEqual(res["stability_status"], "100% Hydrostatically Stable")
        self.assertGreater(res["min_n2_s2"], 0)

    def test_density_inversion_detected(self):
        inverted_rhos = list(self.rhos)
        # Introduce an inversion at depth index 5 (50m): density drops below depth 4 (30m)
        inverted_rhos[5] = inverted_rhos[4] - 0.5
        res = compute_physics_integrity(self.depths, self.temps, self.sals, inverted_rhos)
        self.assertFalse(res["is_stable"])
        self.assertGreater(res["inversion_count"], 0)
        self.assertIn("Inversion Layer", res["stability_status"])

    def test_barrier_layer_computation(self):
        res = compute_physics_integrity(self.depths, self.temps, self.sals, self.rhos)
        self.assertIsNotNone(res["mld_m"])
        self.assertIsNotNone(res["ild_m"])
        self.assertGreaterEqual(res["blt_m"], 0.0)
        self.assertTrue(res["has_barrier_layer"])

    def test_compute_mission_metrics_includes_physics(self):
        sound_speeds = [1540.0] * len(self.depths)
        res = compute_mission_metrics(
            self.depths, self.temps, sound_speeds,
            salinities=self.sals, densities=self.rhos,
        )
        self.assertIn("physics", res)
        self.assertIsNotNone(res["physics"])
        self.assertTrue(res["physics"]["is_stable"])


class NetCDFExportTests(unittest.TestCase):
    def setUp(self):
        self.profile = pd.DataFrame({
            "depth_m": [0, 10, 50, 100, 200, 500, 1000],
            "temperature_c": [29.0, 28.8, 25.5, 20.0, 15.0, 8.0, 4.5],
            "salinity_psu": [33.5, 33.6, 34.2, 34.8, 35.0, 34.9, 34.8],
            "density_kg_m3": [1020.5, 1020.8, 1023.5, 1026.0, 1027.5, 1030.0, 1032.0],
            "sound_speed_m_s": [1541.0, 1541.2, 1533.0, 1520.0, 1508.0, 1495.0, 1494.0],
            "confidence_pct": [96, 95, 92, 88, 85, 80, 78],
            "uncertainty_c": [0.45, 0.48, 0.65, 0.95, 0.70, 0.40, 0.35],
        })
        self.metadata = {
            "title": "OceanEmbed 3D Test Sounding",
            "date": "2023-04-15",
            "latitude": 15.0,
            "longitude": 88.0,
        }

    def test_export_netcdf_bytes_header(self):
        nc_bytes = export_profile_to_netcdf(self.profile, self.metadata)
        self.assertIsInstance(nc_bytes, bytes)
        self.assertTrue(len(nc_bytes) > 200)
        # NetCDF classical binary format magic header
        self.assertEqual(nc_bytes[:3], b"CDF")

    def test_export_netcdf_readable(self):
        nc_bytes = export_profile_to_netcdf(self.profile, self.metadata)
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with open(tmp_path, "wb") as f:
                f.write(nc_bytes)
            with netcdf_file(tmp_path, "r") as nc:
                self.assertIn("depth", nc.variables)
                self.assertIn("temperature", nc.variables)
                self.assertIn("salinity", nc.variables)
                self.assertIn("density", nc.variables)
                self.assertIn("sound_speed", nc.variables)
                self.assertEqual(len(nc.variables["depth"][:]), len(self.profile))
                self.assertEqual(nc.Conventions, b"CF-1.8")
                np.testing.assert_allclose(nc.variables["depth"][:], self.profile["depth_m"])
                np.testing.assert_allclose(nc.variables["temperature"][:], self.profile["temperature_c"], rtol=1e-5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class AcousticRayTests(unittest.TestCase):
    def test_ray_tracing_computation(self):
        from src.models.mission_metrics import compute_acoustic_ray_paths
        depths = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 700, 1000]
        speeds = [1542, 1542, 1542, 1541, 1540, 1535, 1528, 1519, 1505, 1501, 1496, 1494, 1493, 1495]
        rays = compute_acoustic_ray_paths(depths, speeds, source_depth_m=15.0, max_range_km=20.0)
        self.assertEqual(len(rays), 7)
        for r in rays:
            self.assertIn("launch_angle_deg", r)
            self.assertIn("range_km", r)
            self.assertIn("depth_m", r)
            self.assertEqual(len(r["range_km"]), len(r["depth_m"]))
            self.assertGreater(len(r["range_km"]), 10)


class TransectCurtainTests(unittest.TestCase):
    def test_zonal_and_meridional_transects(self):
        from backend import transect
        zonal = transect(15.0, axis="lat", variable="Temperature")
        self.assertEqual(zonal["axis_label"], "Longitude (°E)")
        self.assertEqual(len(zonal["depths"]), 15)
        self.assertEqual(len(zonal["coords"]), 240)
        self.assertEqual(zonal["values"].shape, (15, 240))

        meridional = transect(88.0, axis="lon", variable="Temperature")
        self.assertEqual(meridional["axis_label"], "Latitude (°N)")
        self.assertEqual(len(meridional["coords"]), 100)
        self.assertEqual(meridional["values"].shape, (15, 100))


class MarineHeatwaveTests(unittest.TestCase):
    def test_normal_conditions(self):
        res = compute_marine_heatwave_metrics(sst_c=27.5, climatology_sst_c=28.0)
        self.assertFalse(res["is_heatwave"])
        self.assertEqual(res["category_level"], 0)
        self.assertEqual(res["category"], "No Heatwave")
        self.assertEqual(res["bleaching_status"], "No Thermal Stress")

    def test_moderate_category_i_heatwave(self):
        # delta_t = 29.5 - 28.0 = 1.5 C (with threshold=1.0 C, multiple=1.5 -> Cat I)
        res = compute_marine_heatwave_metrics(sst_c=29.5, climatology_sst_c=28.0, baseline_threshold_c=29.0)
        self.assertTrue(res["is_heatwave"])
        self.assertEqual(res["category_level"], 1)
        self.assertEqual(res["category"], "Category I")
        self.assertIn("Bleaching Likely", res["bleaching_status"])

    def test_severe_category_iii_heatwave(self):
        # delta_t = 31.5 - 28.0 = 3.5 C -> Cat III
        res = compute_marine_heatwave_metrics(sst_c=31.5, climatology_sst_c=28.0, baseline_threshold_c=29.0)
        self.assertTrue(res["is_heatwave"])
        self.assertEqual(res["category_level"], 3)
        self.assertIn("Alert Level 2", res["bleaching_status"])


class NavalTacticsTests(unittest.TestCase):
    def setUp(self):
        self.depths = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 700, 1000]
        # Sound speed profile with maximum at 30m (SLD = 30m), minimum at 700m (SOFAR axis)
        self.speeds = [1540.0, 1540.5, 1541.0, 1541.5, 1542.0, 1535.0, 1525.0, 1515.0, 1502.0, 1498.0, 1495.0, 1492.0, 1491.0, 1494.0]

    def test_sonic_layer_depth_and_cutoff_frequency(self):
        res = compute_naval_sonar_tactics(self.depths, self.speeds)
        self.assertEqual(res["sld_m"], 30.0)
        self.assertEqual(res["sld_sound_speed_m_s"], 1542.0)
        self.assertIsNotNone(res["f_cutoff_hz"])
        # f_cutoff = 1420 / sqrt(30^3) = 1420 / 164.316 = ~8.6 Hz
        self.assertAlmostEqual(res["f_cutoff_hz"], 8.6, delta=0.5)
        self.assertTrue(res["has_surface_duct"])
        self.assertTrue(res["shadow_zone_active"])
        self.assertEqual(res["sofar_axis_m"], 700.0)

    def test_shallow_mixed_layer_cutoff(self):
        # Linear decreasing sound speed: no duct, SLD at surface (0m)
        speeds = [1540.0 - 0.1 * z for z in self.depths]
        res = compute_naval_sonar_tactics(self.depths, speeds)
        self.assertEqual(res["sld_m"], 0.0)
        self.assertIsNone(res["f_cutoff_hz"])
        self.assertFalse(res["has_surface_duct"])


class ExecutiveBriefingTests(unittest.TestCase):
    def test_briefing_generation(self):
        context = {
            "date": "2023-04-15",
            "sampled_location": {"latitude": 15.0, "longitude": 88.0},
        }
        depths = [0, 10, 50, 100, 200, 500, 1000]
        temps = [29.5, 29.2, 25.0, 19.0, 14.0, 8.0, 4.5]
        speeds = [1542.0, 1542.5, 1530.0, 1515.0, 1500.0, 1492.0, 1494.0]
        sals = [33.0, 33.2, 34.0, 34.8, 35.0, 34.8, 34.7]
        rhos = [1020.0, 1020.5, 1023.0, 1026.0, 1028.0, 1030.0, 1032.0]
        metrics = compute_mission_metrics(depths, temps, speeds, salinities=sals, densities=rhos)
        briefing = generate_mission_briefing(context, metrics)
        self.assertIsInstance(briefing, str)
        self.assertIn("EXECUTIVE OPERATIONAL MISSION BRIEFING", briefing)
        self.assertIn("Bay of Bengal", briefing)
        self.assertIn("TCHP", briefing)
        self.assertIn("Sonic Layer Depth", briefing)
        self.assertIn("Mixed Layer Depth", briefing)


if __name__ == "__main__":
    unittest.main()
