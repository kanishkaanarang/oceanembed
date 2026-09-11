"""
Baseline XGBoost model for OceanEmbed.
Predicts subsurface temperature at standard depth levels from surface inputs.
"""
import pandas as pd
import joblib

STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

_model = None

def load_model(path="results/baseline_xgb_model.pkl"):
    """Load the trained model once and cache it."""
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model

def predict_profile(lat, lon, date, sst_val, sss_val, depths=None):
    """
    Predict subsurface temperature at each depth for a given location/date.
    sst_val and sss_val must be supplied by the caller (pulled from the
    satellite data pipeline) — this function does not fetch them itself.
    Returns a list of (depth, predicted_temperature) tuples.
    """
    model = load_model()
    depths = depths or STANDARD_DEPTHS
    doy = pd.Timestamp(date).dayofyear

    rows = pd.DataFrame({
        'lat': [lat] * len(depths),
        'lon': [lon] * len(depths),
        'day_of_year': [doy] * len(depths),
        'sst': [sst_val] * len(depths),
        'sss': [sss_val] * len(depths),
        'depth': depths,
    })
    preds = model.predict(rows)
    return list(zip(depths, preds))