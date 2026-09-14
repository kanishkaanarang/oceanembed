"""Regressions for preset navigation, API indexing, and export metadata."""
from datetime import date
from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.io import netcdf_file
from streamlit.testing.v1 import AppTest

from src.api import ocean_engine
from src.models.mission_metrics import export_profile_to_netcdf


class PresetRegressionTests(unittest.TestCase):
    def test_every_event_preset_and_return_to_custom_coordinates(self):
        from backend import CYCLONE_EVENT_PRESETS
        page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
        for preset in CYCLONE_EVENT_PRESETS:
            with self.subTest(preset=preset["id"]):
                page.button(key=f"btn_p_{preset['id']}").click().run()
                self.assertEqual(len(page.exception), 0, str(page.exception))
                self.assertEqual(page.session_state["latitude"], preset["latitude"])
                self.assertEqual(page.session_state["longitude"], preset["longitude"])
                self.assertEqual(page.session_state["analysis_date"], preset["date"])
                self.assertEqual(page.selectbox(key="region_preset").value, preset["id"])
        page.selectbox(key="region_preset").select("central_as").run()
        self.assertEqual(len(page.exception), 0, str(page.exception))
        self.assertEqual(page.session_state["longitude"], 66.0)
        page.slider(key="latitude").set_value(17.0).run()
        self.assertEqual(len(page.exception), 0, str(page.exception))
        self.assertEqual(page.session_state["active_preset"], "custom")
        self.assertEqual(page.session_state["region_preset"], "custom")
        page.toggle(key="dark_mode").set_value(True).run()
        self.assertEqual(page.session_state["latitude"], 17.0)
        self.assertEqual(len(page.exception), 0, str(page.exception))


class ApiIndexRegressionTests(unittest.TestCase):
    def test_date_origin_and_clipping_match_streamlit(self):
        from backend import _date_to_day_index
        for text in ["2023-01-05", "2023-02-01", "2023-03-01", "2023-06-01", "2023-06-30", "2026-01-01"]:
            self.assertEqual(ocean_engine._date_to_day_index(text), _date_to_day_index(date.fromisoformat(text)))
        self.assertEqual(ocean_engine._date_to_day_index("invalid"), 0)
        self.assertEqual(ocean_engine._date_to_day_index("NaT"), 0)

    def test_satellite_and_custom_depth_prediction_use_same_day(self):
        inputs = np.zeros((150, 7, 1, 1), dtype=np.float32)
        inputs[:, 0, 0, 0] = np.arange(150)
        inputs[:, 1, 0, 0] = 35
        arrays = {"inputs": inputs, "input_mean": np.zeros((1, 7, 1, 1)),
                  "input_std": np.ones((1, 7, 1, 1)),
                  "target_mean": np.zeros((1, 15, 1, 1)), "target_std": np.ones((1, 15, 1, 1))}
        temperature = (30 - np.asarray(ocean_engine.STANDARD_DEPTHS) / 50).reshape(15, 1, 1)
        with patch.object(ocean_engine, "load_training_arrays", return_value=arrays), \
             patch.object(ocean_engine, "get_oceanembed_net_weights", return_value={}), \
             patch.object(ocean_engine, "run_numpy_cnn_forward", return_value=(temperature, None)) as forward:
            result = ocean_engine.predict_full_column(5, 45, "2023-03-01", [0, 37, 1000])
        self.assertEqual(result["surface_satellite_inputs"]["sst"], 28)
        self.assertEqual(forward.call_args.args[0][0, 0, 0], 28)
        np.testing.assert_allclose([row["temperature"] for row in result["profile"]], [30, 29.26, 10])
        self.assertEqual([row["depth"] for row in result["profile"]], [0, 37, 1000])

    def test_invalid_custom_depths_fail_before_loading_model(self):
        with patch.object(ocean_engine, "load_training_arrays") as loader:
            for depths in [[], [-1, 10], [0, 1001], [0, np.nan], [[0, 10]]]:
                with self.subTest(depths=depths), self.assertRaises(ValueError):
                    ocean_engine.predict_full_column(5, 45, depths=depths)
            loader.assert_not_called()


class ExportLocationRegressionTests(unittest.TestCase):
    def test_nested_ui_context_exports_sampled_not_requested_coordinates(self):
        profile = pd.DataFrame({"depth_m": [0, 10], "temperature_c": [28, 27]})
        data = export_profile_to_netcdf(profile, {
            "date": "2023-03-01", "requested_location": {"latitude": 14.6, "longitude": 88.1},
            "sampled_location": {"latitude": 14.5, "longitude": 88.0},
        })
        with netcdf_file(BytesIO(data), "r", mmap=False) as dataset:
            self.assertEqual(dataset.latitude, 14.5)
            self.assertEqual(dataset.longitude, 88.0)
            self.assertEqual(dataset.analysis_date, b"2023-03-01")


if __name__ == "__main__":
    unittest.main()
