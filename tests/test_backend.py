"""Local checks for the synthetic backend and the future model contract."""

from datetime import date
import unittest

import pandas as pd

from backend import field, reconstruct
from model_adapter import ModelOutputError, validate_profile


class BackendTests(unittest.TestCase):
    def test_profile_changes_by_location_date_and_depth(self) -> None:
        first = reconstruct(13.5, 88.5, date(2023, 3, 1))
        second = reconstruct(15.0, 91.0, date(2023, 4, 1))
        self.assertFalse(first["temperature_c"].equals(second["temperature_c"]))
        self.assertNotEqual(first.loc[first.depth_m == 0, "temperature_c"].iat[0], first.loc[first.depth_m == 500, "temperature_c"].iat[0])

    def test_grid_changes_with_depth(self) -> None:
        surface = field(13.5, 88.5, 0, "Temperature", date(2023, 3, 1))
        deep = field(13.5, 88.5, 500, "Temperature", date(2023, 3, 1))
        self.assertFalse(surface["value"].equals(deep["value"]))

    def test_profile_contract_normalizes_minimal_model_output(self) -> None:
        result = validate_profile(pd.DataFrame({"depth_m": [100, 0], "temperature_c": [22.0, 28.0], "salinity_psu": [34.6, 34.1]}))
        self.assertEqual(list(result["depth_m"]), [0, 100])
        self.assertIn("confidence_pct", result.columns)

    def test_profile_contract_rejects_missing_columns(self) -> None:
        with self.assertRaises(ModelOutputError):
            validate_profile(pd.DataFrame({"depth_m": [0], "temperature_c": [28.0]}))


if __name__ == "__main__":
    unittest.main()
