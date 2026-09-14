"""Scientific plotting and export checks independent of model inference."""
import json
import unittest
import numpy as np
import pandas as pd

from src.ui.mission_panel import analysis_export, heat_content_figure, profile_metrics, sound_profile_figure


class MissionPresentationTests(unittest.TestCase):
    def setUp(self):
        self.profile = pd.DataFrame({"depth_m": [0, 100, 200], "temperature_c": [28, 24, 20],
                                     "sound_speed_m_s": [1530, 1510, 1490], "salinity_psu": [35, 35, 35]})
        self.metrics = profile_metrics(self.profile)
        self.colors = dict(paper="#fff", panel="#fff", text="#123", grid="#ddd", model="#087f83")

    def test_heat_polygon_ends_at_interpolated_d26(self):
        figure = heat_content_figure(self.profile, self.metrics["cyclone"], self.colors)
        self.assertEqual(max(figure.data[0].y), 50)
        self.assertTrue(np.all(np.asarray(figure.data[0].x) >= 26))
        self.assertEqual(figure.layout.yaxis.autorange, "reversed")

    def test_cold_profile_has_no_shaded_reservoir(self):
        self.profile["temperature_c"] = [25, 24, 20]
        heat = profile_metrics(self.profile)["cyclone"]
        figure = heat_content_figure(self.profile, heat, self.colors)
        self.assertFalse(any(trace.fill == "toself" for trace in figure.data))

    def test_truncated_warm_profile_shades_only_sampled_depths(self):
        self.profile["temperature_c"] = [29, 28, 27]
        heat = profile_metrics(self.profile)["cyclone"]
        figure = heat_content_figure(self.profile, heat, self.colors)
        self.assertEqual(max(figure.data[0].y), 200)
        self.assertTrue(heat["is_lower_bound"])

    def test_comparison_and_shading(self):
        figure = sound_profile_figure(self.profile, self.metrics["acoustics"], self.colors, self.profile)
        self.assertEqual(len(figure.data), 2)
        self.assertEqual(figure.layout.shapes[0].y0, 0)
        self.assertEqual(figure.layout.shapes[0].y1, 100)

    def test_export_retains_context_units_and_comparison(self):
        context = {"date": "2023-03-01", "dataset_mode": "test_fixture"}
        report = json.loads(analysis_export(self.profile, self.metrics, context, self.profile, context))
        self.assertEqual(report["context"], context)
        self.assertEqual(report["comparison"]["context"], context)
        self.assertEqual(report["units"]["heat_content"], "kJ/cm²")
        self.assertEqual(len(report["profile"]), 3)
        self.assertEqual(report["mission_intelligence"]["cyclone"]["d26_m"], 50)


if __name__ == "__main__":
    unittest.main()
