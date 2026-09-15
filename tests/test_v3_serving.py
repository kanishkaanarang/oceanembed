from datetime import date
import numpy as np
import pandas as pd
import pytest
from backend import reconstruct, field
from src.models.serving_engine import get_engine, CHECKPOINT, STANDARD_DEPTHS
from src.models.train_oceanembed_v3 import predict_from_checkpoint
from src.api.ocean_engine import predict_full_column


def test_serving_matches_checkpoint_and_both_map_heads():
    engine = get_engine()
    day = 28
    data = engine.predict_day(day)
    lat_idx, lon_idx = 38, 172  # 14.5 N, 88 E
    surface = data['surface_inputs'][:, lat_idx, lon_idx]
    doy = date(2023, 3, 1).timetuple().tm_yday
    frame = pd.DataFrame([dict(lat=14.5, lon=88., doy_sin=np.sin(2*np.pi*doy/365),
        doy_cos=np.cos(2*np.pi*doy/365), sst=surface[0], sss=surface[1], ssh=surface[2],
        current_u=surface[5], current_v=surface[6])])
    direct = predict_from_checkpoint(CHECKPOINT, frame)
    np.testing.assert_allclose(data['pred_celsius'][:, lat_idx, lon_idx], direct['temperature_c'][0], atol=1e-4)
    np.testing.assert_allclose(data['pred_salinity'][:, lat_idx, lon_idx], direct['salinity_psu'][0], atol=1e-4)
    profile = reconstruct(14.5, 88., date(2023, 3, 1))
    assert profile.depth_m.to_list() == direct['depths_m'] == STANDARD_DEPTHS
    assert profile.argo_salinity_psu.isna().all()
    for variable, column in [('Temperature', 'temperature_c'), ('Salinity', 'salinity_psu')]:
        grid = field(14.5, 88., 900, variable, date(2023, 3, 1))
        cell = grid[(grid.latitude == 14.5) & (grid.longitude == 88.)].value.iat[0]
        assert cell == pytest.approx(profile[column].iat[-1], abs=.011)


def test_api_v3_matches_dashboard_and_rejects_untrained_depth():
    result = predict_full_column(14.5, 88., '2023-03-01', [0, 37, 900])
    profile = reconstruct(14.5, 88., date(2023, 3, 1))
    assert result['model_provenance'] == 'oceanembed_v3_se_resnet'
    np.testing.assert_allclose([r['temperature'] for r in result['profile']],
                              np.interp([0, 37, 900], profile.depth_m, profile.temperature_c))
    with pytest.raises(ValueError, match='900'):
        predict_full_column(14.5, 88., '2023-03-01', [1000])


def test_api_metadata_identifies_loaded_checkpoint():
    from src.api.main import get_model_info, get_model_benchmark_metrics, health_check
    info = get_model_info()
    assert 'v3' in info['model_name']
    assert info['output_depths_m'] == STANDARD_DEPTHS
    assert info['performance'] == get_model_benchmark_metrics()['test']
    assert 'v3' in health_check()['model']


def test_old_browser_session_migrates_depth_and_transect_controls():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=60)
    page.session_state['depth'] = 1000
    page.session_state['explore_view_mode'] = 'Vertical Transect Curtain (0\u20131000m)'
    page.session_state['workspace_page'] = 'Ocean Explorer'
    page.run()
    assert not page.exception, str(page.exception)
    assert page.session_state['depth'] == 900
    assert page.radio(key='explore_view_mode').value == 'Vertical Transect Curtain (0\u2013900m)'
