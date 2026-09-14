from pathlib import Path
from datetime import date

import numpy as np
from streamlit.testing.v1 import AppTest
from backend import transect, field, COMMON_LATS, COMMON_LONS

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def test_all_pages_preserve_shared_selection_and_theme():
    page = AppTest.from_file(APP, default_timeout=60).run()
    assert not page.exception
    assert len(page.get('plotly_chart')) == 0
    page.slider(key='latitude').set_value(16.).run()
    page.toggle(key='dark_mode').set_value(True).run()
    for destination in page.radio(key='workspace_page').options:
        page.radio(key='workspace_page').set_value(destination).run()
        assert not page.exception, str(page.exception)
        assert page.session_state['latitude'] == 16.
        assert page.session_state['dark_mode'] is True
    page.radio(key='workspace_page').set_value('Profiles & Data').run()
    assert any(len(df.value) == 15 for df in page.dataframe)
    labels = [item.label for item in page.get('download_button')]
    assert any('CSV' in label for label in labels)
    assert any('NetCDF' in label for label in labels)


def test_transect_every_layer_both_axes_and_return_navigation():
    page = AppTest.from_file(APP, default_timeout=60).run()
    page.radio(key='workspace_page').set_value('Ocean Explorer').run()
    for layer in page.selectbox(key='variable').options:
        page.selectbox(key='variable').select(layer).run()
        assert not page.exception, str(page.exception)
        page.radio(key='explore_view_mode').set_value('Vertical Transect Curtain (0–1000m)').run()
        for orientation in page.radio(key='transect_axis_choice').options:
            page.radio(key='transect_axis_choice').set_value(orientation).run()
            assert not page.exception, str(page.exception)
        page.radio(key='explore_view_mode').set_value('Horizontal Map (Depth Slice)').run()
    page.radio(key='explore_view_mode').set_value('Vertical Transect Curtain (0–1000m)').run()
    page.radio(key='workspace_page').set_value('Overview').run()
    page.radio(key='workspace_page').set_value('Ocean Explorer').run()
    assert page.radio(key='explore_view_mode').value == 'Vertical Transect Curtain (0–1000m)'
    assert page.selectbox(key='variable').value == 'Confidence'
    assert not page.exception


def test_transect_matches_map_values_and_units():
    for layer in ['Temperature', 'Salinity', 'Density', 'Sound Speed', 'Confidence', 'Embedding']:
        for axis, coordinate in [('lat', 14.5), ('lon', 88.)]:
            curtain = transect(coordinate, axis, layer, date(2023, 3, 1))
            grid = field(14.5, 88., 100, layer, date(2023, 3, 1))
            values = grid.value.to_numpy().reshape(len(COMMON_LATS), len(COMMON_LONS))
            fixed = COMMON_LATS if axis == 'lat' else COMMON_LONS
            idx = np.argmin(abs(fixed-coordinate))
            expected = values[idx, :] if axis == 'lat' else values[:, idx]
            np.testing.assert_allclose(curtain['values'][7], expected, equal_nan=True)
            assert curtain['unit'] == grid.unit.iat[0]
