import joblib
joblib.dump(model, '../results/baseline_xgb_model.pkl')

def predict_profile(lat, lon, date, depths=[0,10,25,50,75,100,150,200,300,500,700,1000]):
    doy = pd.Timestamp(date).dayofyear
    sst_val = float(ds_sst['thetao'].sel(latitude=lat, longitude=lon, time=str(date), method='nearest').isel(depth=0))
    rows = pd.DataFrame({
        'lat': [lat]*len(depths), 'lon': [lon]*len(depths),
        'day_of_year': [doy]*len(depths), 'sst': [sst_val]*len(depths), 'depth': depths
    })
    preds = model.predict(rows)
    return list(zip(depths, preds))