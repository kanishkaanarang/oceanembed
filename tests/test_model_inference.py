"""
Unit and physical consistency tests for OceanEmbed model inference and oceanographic equations.
Tests:
1. Weight unpacking and tensor architecture verification for OceanEmbedNet v2.
2. Direct 2D CNN inference pipeline (PyTorch / NumPy).
3. Physical oceanography equations:
   - UNESCO 1980 Equation of State (seawater density).
   - Mackenzie (1981) formula (naval sound velocity / SVP).
   - Pycnocline stability & acoustic sound channel (SOFAR) axis detection.
4. Input and output bounds verification.
"""
import os
import unittest
import numpy as np

from src.api.ocean_engine import (
    STANDARD_DEPTHS,
    compute_seawater_density,
    compute_sound_velocity,
    predict_full_column,
    get_oceanembed_net_weights,
    run_numpy_cnn_forward,
)

class TestOceanEmbedModel(unittest.TestCase):

    def test_standard_depths(self):
        """Verify the 15 standard oceanographic depth tiers."""
        self.assertEqual(len(STANDARD_DEPTHS), 15)
        self.assertEqual(STANDARD_DEPTHS[0], 0)
        self.assertEqual(STANDARD_DEPTHS[-1], 1000)
        self.assertEqual(STANDARD_DEPTHS, [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000])

    def test_checkpoint_weights_exist_and_match_shapes(self):
        """Verify CNN weights can be loaded and match expected layer shapes."""
        weights = get_oceanembed_net_weights()
        self.assertIsNotNone(weights, "Failed to load OceanEmbedNet weights from checkpoint.")
        self.assertEqual(weights["w0"].shape, (16, 7, 3, 3))
        self.assertEqual(weights["b0"].shape, (16,))
        self.assertEqual(weights["w1"].shape, (32, 16, 3, 3))
        self.assertEqual(weights["b1"].shape, (32,))
        self.assertEqual(weights["w2"].shape, (16, 32, 3, 3))
        self.assertEqual(weights["b2"].shape, (16,))
        self.assertEqual(weights["w3"].shape, (15, 16, 3, 3))
        self.assertEqual(weights["b3"].shape, (15,))

    def test_cnn_forward_pass(self):
        """Verify the CNN forward pass produces the correct shape and finite values."""
        # Simulated 7-channel surface observation tensor: (7, 100, 240)
        dummy_input = np.random.randn(7, 100, 240).astype(np.float32)
        pred, embed = run_numpy_cnn_forward(dummy_input)

        self.assertEqual(pred.shape, (15, 100, 240))
        self.assertEqual(embed.shape, (32, 100, 240))
        self.assertTrue(np.all(np.isfinite(pred)), "Predictions contain NaN or Inf values")
        self.assertTrue(np.all(np.isfinite(embed)), "Embeddings contain NaN or Inf values")

    def test_unesco_seawater_density(self):
        """Verify seawater density matches expected oceanographic range (1020 - 1035 kg/m^3)."""
        # Surface warm tropical water (29 deg C, 34 PSU, 0m)
        rho_surface = compute_seawater_density(temp_c=29.0, sal_psu=34.0, depth_m=0.0)
        self.assertTrue(1018.0 <= rho_surface <= 1026.0, f"Unexpected surface density: {rho_surface}")

        # Deep cold water (4 deg C, 35 PSU, 1000m)
        rho_deep = compute_seawater_density(temp_c=4.0, sal_psu=35.0, depth_m=1000.0)
        self.assertTrue(1028.0 <= rho_deep <= 1035.0, f"Unexpected deep density: {rho_deep}")
        self.assertGreater(rho_deep, rho_surface, "Deep ocean must be denser than surface water for stability")

    def test_mackenzie_sound_velocity(self):
        """Verify sound velocity formula against standard oceanic sound speeds (1480 - 1545 m/s)."""
        # Surface sound speed: approx 1540 m/s for warm tropical water (28°C, 34 PSU)
        c_surface = compute_sound_velocity(temp_c=28.0, sal_psu=34.0, depth_m=0.0)
        self.assertTrue(1530.0 <= c_surface <= 1550.0, f"Unexpected surface sound speed: {c_surface}")

        # SOFAR channel minimum: cold water at 800-1000m (~1485 m/s)
        c_deep = compute_sound_velocity(temp_c=5.0, sal_psu=34.9, depth_m=1000.0)
        self.assertTrue(1480.0 <= c_deep <= 1505.0, f"Unexpected deep sound speed: {c_deep}")

    def test_full_column_prediction(self):
        """Verify the end-to-end full column prediction returns all variables and physics metrics."""
        result = predict_full_column(lat=15.0, lon=88.0, date="2023-01-05")

        self.assertIn("location", result)
        self.assertIn("date", result)
        self.assertIn("surface_satellite_inputs", result)
        self.assertIn("thermocline_depth_estimate_m", result)
        self.assertIn("sofar_axis_depth_m", result)
        self.assertIn("hydrostatic_stability", result)
        self.assertIn("profile", result)

        profile = result["profile"]
        self.assertEqual(len(profile), 15)

        for tier in profile:
            self.assertIn("depth", tier)
            self.assertIn("temperature", tier)
            self.assertIn("salinity", tier)
            self.assertIn("density", tier)
            self.assertIn("sound_speed", tier)

            # Check physical realism
            self.assertTrue(0.0 <= tier["temperature"] <= 35.0)
            self.assertTrue(25.0 <= tier["salinity"] <= 40.0)
            self.assertTrue(1015.0 <= tier["density"] <= 1040.0)
            self.assertTrue(1450.0 <= tier["sound_speed"] <= 1560.0)

if __name__ == "__main__":
    unittest.main()
