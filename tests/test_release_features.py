"""Exercise release-critical controls in both themes without touching user data."""
import json
import base64
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import backend
from src.ui.theme import themed_table
from src.ui.mission_panel import multi_variable_figure

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def assert_clean(page):
    assert not page.exception, str(page.exception)


@pytest.mark.parametrize('dark', [False, True])
def test_every_page_theme_round_trip_preserves_data_and_controls(dark):
    page = AppTest.from_file(APP, default_timeout=60).run()
    page.toggle(key='comparison_enabled').set_value(True).run()
    page.slider(key='comparison_latitude').set_value(17.).run()
    for destination in page.radio(key='workspace_page').options:
        page.radio(key='workspace_page').set_value(destination).run()
        page.toggle(key='dark_mode').set_value(dark).run()
        assert_clean(page)
        before = [df.value.copy() for df in page.dataframe]
        metrics = [(m.label, m.value) for m in page.metric]
        for theme in [not dark, dark]:
            page.toggle(key='dark_mode').set_value(theme).run()
            assert_clean(page)
            assert page.session_state['comparison_latitude'] == 17.
            assert [(m.label, m.value) for m in page.metric] == metrics
            assert len(page.dataframe) == len(before)
            for a, b in zip(before, page.dataframe):
                pd.testing.assert_frame_equal(a, b.value)
            for chart in page.get('plotly_chart'):
                layout = json.loads(chart.proto.spec)['layout']
                assert layout['paper_bgcolor'] == ('#071b2a' if theme else '#ffffff')
                assert layout['legend']['font']['color'] == ('#e7f4f8' if theme else '#0f172a')
                assert chart.proto.theme == ''


@pytest.mark.parametrize('dark', [False, True])
def test_map_layers_dates_depths_and_preview_persistence(dark):
    page = AppTest.from_file(APP, default_timeout=60).run()
    page.toggle(key='dark_mode').set_value(dark).run()
    page.radio(key='workspace_page').set_value('Ocean Explorer').run()
    for layer in page.selectbox(key='variable').options:
        page.selectbox(key='variable').select(layer).run()
        assert_clean(page)
        plot = json.loads(page.get('plotly_chart')[0].proto.spec)
        assert plot['layout']['clickmode'] == 'event+select'
        assert any(trace.get('name') == 'Ocean cells' for trace in plot['data'])
        if layer in ('Ground Truth', 'Residual', 'Density', 'Sound Speed', 'Confidence'):
            column = {'Ground Truth': 'argo_temperature_c', 'Residual': 'error_c',
                      'Density': 'density_kg_m3', 'Sound Speed': 'sound_speed_m_s',
                      'Confidence': 'confidence_pct'}[layer]
            profile_chart = json.loads(page.get('plotly_chart')[1].proto.spec)
            x = profile_chart['data'][0]['x']
            if isinstance(x, dict):
                x = np.frombuffer(base64.b64decode(x['bdata']), dtype=x['dtype'])
            np.testing.assert_allclose(x, page.dataframe[0].value[column])
    for day, depth in [(backend.ANALYSIS_START, 0), (backend.ANALYSIS_END, int(backend.DEPTHS[-1]))]:
        page.select_slider(key='analysis_date').set_value(day).run()
        page.select_slider(key='depth').set_value(depth).run()
        assert_clean(page)
    page.radio(key='workspace_page').set_value('Mission Intelligence').run()
    initial = len(page.get('plotly_chart'))
    page.checkbox(key='mission-svp-preview').check().run()
    assert_clean(page)
    assert len(page.get('plotly_chart')) == initial + 1
    page.radio(key='workspace_page').set_value('Profiles & Data').run()
    page.radio(key='workspace_page').set_value('Mission Intelligence').run()
    assert_clean(page)
    assert page.checkbox(key='mission-svp-preview').value
    assert len(page.get('download_button')) == 3
    page.toggle(key='dark_mode').set_value(not dark).run()
    assert_clean(page)


def test_saved_locations_add_duplicate_list_clear_in_isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, 'LOCATIONS_DB', tmp_path / 'locations.sqlite')
    page = AppTest.from_file(APP, default_timeout=60).run()
    for _ in range(2):
        page.button(key='save-location').click().run()
        assert_clean(page)
    page.radio(key='workspace_page').set_value('About & Saved Places').run()
    assert len(backend.list_saved_locations()) == 1
    next(b for b in page.button if b.label == 'Clear saved locations').click().run()
    assert_clean(page)
    assert backend.list_saved_locations().empty


def test_all_presets_remain_available_in_both_themes():
    page = AppTest.from_file(APP, default_timeout=60).run()
    for dark in (True, False):
        page.toggle(key='dark_mode').set_value(dark).run()
        for preset in backend.REGIONAL_PRESETS + backend.CYCLONE_EVENT_PRESETS:
            page.selectbox(key='region_preset').select(preset['id']).run()
            assert_clean(page)
            assert page.session_state['latitude'] == preset['latitude']
            assert page.session_state['longitude'] == preset['longitude']


@pytest.mark.parametrize('dark', [False, True])
def test_table_styling_changes_only_presentation(dark):
    frame = pd.DataFrame({'depth_m': [0, 1000], 'temperature_c': [28., 4.]})
    styled = themed_table(frame, dark)
    pd.testing.assert_frame_equal(styled.data, frame)
    html = styled.to_html()
    assert ('#0c2433' if dark else '#ffffff') in html
    assert ('#e7f4f8' if dark else '#0f172a') in html


def test_multivariable_axes_do_not_share_the_same_bottom_position():
    frame = pd.DataFrame({'depth_m': [0, 1000], 'temperature_c': [28, 4],
                          'salinity_psu': [34, 35], 'sound_speed_m_s': [1540, 1490]})
    colors = dict(paper='#fff', panel='#fff', text='#123', grid='#ddd')
    figure = multi_variable_figure(frame, colors)
    assert figure.layout.xaxis3.anchor == 'free'
    assert figure.layout.xaxis3.position < figure.layout.yaxis.domain[0]
