"""Exercise the real Streamlit page, existing controls, and additive diagnostics."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class MissionDashboardTests(unittest.TestCase):
    def test_cards_preview_themes_and_existing_controls(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        page = AppTest.from_file(str(app_path), default_timeout=60).run()
        self.assertEqual(len(page.exception), 0, str(page.exception))
        self.assertIn("TCHP / OHC above 26°C", [m.label for m in page.metric])
        self.assertEqual(len(page.tabs), 4)
        table_columns = [list(df.value.columns) for df in page.dataframe]
        chart_count = len(page.get("plotly_chart"))
        page.checkbox(key="mission-svp-preview").check().run()
        self.assertEqual(len(page.exception), 0, str(page.exception))
        self.assertEqual(len(page.get("plotly_chart")), chart_count + 1)
        page.toggle(key="dark_mode").set_value(True).run()
        self.assertEqual(len(page.exception), 0, str(page.exception))
        self.assertEqual([list(df.value.columns) for df in page.dataframe], table_columns)
        page.selectbox[0].select("central_as").run()
        self.assertEqual(len(page.exception), 0, str(page.exception))
        self.assertEqual(page.session_state["latitude"], 16.0)
        self.assertEqual(page.session_state["longitude"], 66.0)
        page.toggle(key="dark_mode").set_value(False).run()
        self.assertEqual(len(page.exception), 0, str(page.exception))
        page.checkbox(key="mission-svp-preview").uncheck().run()
        self.assertEqual(len(page.exception), 0, str(page.exception))
        self.assertEqual(len(page.get("plotly_chart")), chart_count)
        page.toggle(key="comparison_enabled").set_value(True).run()
        self.assertEqual(len(page.exception), 0, str(page.exception))
        self.assertTrue(any("vs comparison" in (m.delta or "") for m in page.metric))
        self.assertTrue(any("Download mission analysis" in b.label for b in page.get("download_button")))
        page.checkbox(key="mission-svp-preview").check().run()
        self.assertEqual(len(page.exception), 0, str(page.exception))


if __name__ == "__main__":
    unittest.main()
