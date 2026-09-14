"""Analytic physics fixtures and API compatibility checks; no model assets needed."""
import json
import unittest
from unittest.mock import patch

import numpy as np

from src.models.mission_metrics import (
    HEAT_FACTOR, compute_acoustic_profile_diagnostics,
    compute_cyclone_heat_potential, compute_mission_metrics, mackenzie_sound_speed,
)


class HeatPotentialTests(unittest.TestCase):
    def test_interpolated_triangle_and_units(self):
        # 28 -> 24 C over 100 m crosses at 50 m; 50 K*m = 20.464125 kJ/cm2.
        result = compute_cyclone_heat_potential([0, 100], [28, 24])
        self.assertEqual(result["d26_m"], 50)
        self.assertAlmostEqual(result["tchp_kj_cm2"], 20.464125)
        self.assertEqual(result["tchp_kj_cm2"], result["ohc_kj_cm2"])

    def test_irregular_depths_and_exact_crossing(self):
        result = compute_cyclone_heat_potential([0, 10, 60], [29, 28, 26])
        self.assertEqual(result["d26_m"], 60)
        self.assertAlmostEqual(result["tchp_kj_cm2"], 75 * HEAT_FACTOR)

    def test_sorting_preserves_pairing_and_input(self):
        z, t = [100, 0, 25], [24, 28, 27]
        result = compute_cyclone_heat_potential(z, t)
        self.assertEqual(result["d26_m"], 50)
        self.assertEqual(z, [100, 0, 25])
        self.assertEqual(t, [24, 28, 27])

    def test_cold_and_threshold_surface(self):
        for surface in [25, 26]:
            result = compute_cyclone_heat_potential([0, 100], [surface, 29])
            self.assertEqual(result["d26_m"], 0)
            self.assertEqual(result["tchp_kj_cm2"], 0)

    def test_inversion_stops_at_first_crossing(self):
        result = compute_cyclone_heat_potential([0, 50, 100, 200], [28, 26, 29, 20])
        self.assertEqual(result["d26_m"], 50)
        self.assertAlmostEqual(result["tchp_kj_cm2"], 50 * HEAT_FACTOR)

    def test_risk_boundaries(self):
        for heat, expected in [(49.99, "low"), (50, "moderate"), (80, "moderate"), (80.01, "high")]:
            with self.subTest(heat=heat):
                result = compute_cyclone_heat_potential([0, heat / HEAT_FACTOR], [28, 26])
                self.assertEqual(result["risk_level"], expected)

    def test_unresolved_isotherm_is_lower_bound(self):
        for depth, expected in [(10, "undetermined"), (70, "undetermined"), (100, "high")]:
            result = compute_cyclone_heat_potential([0, depth], [28, 28])
            self.assertIsNone(result["d26_m"])
            self.assertTrue(result["is_lower_bound"])
            self.assertEqual(result["risk_level"], expected)
            self.assertAlmostEqual(result["tchp_kj_cm2"], 2 * depth * HEAT_FACTOR)

    def test_invalid_profiles(self):
        cases = [([], []), ([0], [28]), ([0, 1], [28]), ([0, 0], [28, 27]),
                 ([-1, 0], [28, 27]), ([5, 10], [28, 27]),
                 ([0, 10], [28, np.nan]), ([0, np.inf], [28, 27]),
                 ([[0, 1]], [[28, 27]])]
        for z, t in cases:
            with self.subTest(z=z, t=t), self.assertRaises(ValueError):
                compute_cyclone_heat_potential(z, t)


class AcousticDiagnosticsTests(unittest.TestCase):
    def test_mackenzie_reference_and_cross_terms(self):
        self.assertAlmostEqual(mackenzie_sound_speed(10, 35, 1000), 1506.263761)
        self.assertAlmostEqual(mackenzie_sound_speed(10, 30, 1000), 1500.076261)

    def test_scalar_array_broadcast(self):
        result = mackenzie_sound_speed([10, 10], 35, [0, 1000])
        np.testing.assert_allclose(result, [1489.8034, 1506.263761])
        self.assertIsInstance(mackenzie_sound_speed(10, 35, 0), float)

    def test_cooling_interval_and_range(self):
        result = compute_acoustic_profile_diagnostics([100, 0, 25], [20, 28, 27], [1490, 1520, 1510])
        self.assertEqual(result["strongest_cooling_interval_m"], [25, 100])
        self.assertEqual(result["minimum_sound_speed_m_s"], 1490)
        self.assertEqual(result["maximum_sound_speed_m_s"], 1520)
        self.assertAlmostEqual(result["maximum_cooling_gradient_c_m"], 7/75)

    def test_uniform_and_warming_profiles(self):
        for t in [[20, 20], [20, 21]]:
            result = compute_acoustic_profile_diagnostics([0, 100], t, [1500, 1501])
            self.assertIsNone(result["strongest_cooling_interval_m"])

    def test_invalid_speed(self):
        for speeds in [[1500, 0], [1500, np.nan], [1500]]:
            with self.assertRaises(ValueError):
                compute_acoustic_profile_diagnostics([0, 100], [28, 20], speeds)

    def test_independent_failures_and_strict_json(self):
        result = compute_mission_metrics([0, 100], [28, 24], [1500, np.nan])
        self.assertIsNotNone(result["cyclone"])
        self.assertIsNone(result["acoustics"])
        json.dumps(result, allow_nan=False)
        result = compute_mission_metrics([5, 100], [28, 24], [1500, 1490])
        self.assertIsNone(result["cyclone"])
        self.assertIsNotNone(result["acoustics"])
        json.dumps(result, allow_nan=False)


class ApiIntegrationTests(unittest.TestCase):
    def test_response_preserves_new_metrics_and_legacy_fields(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        with patch("src.api.ocean_engine.load_training_arrays", return_value=None), \
             patch("src.api.ocean_engine.get_oceanembed_net_weights", return_value=None):
            response = TestClient(app).get("/api/predict/profile", params={"lat": 15, "lon": 88})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["profile"]), 15)
        self.assertIn("sofar_axis_depth_m", payload)
        self.assertIn("thermocline_depth_estimate_m", payload)
        self.assertEqual(set(payload["profile"][0]), {"depth", "temperature", "salinity", "density", "sound_speed"})
        self.assertEqual(payload["mission_intelligence"]["errors"], {})
        self.assertIsNotNone(payload["mission_intelligence"]["cyclone"])
        json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
