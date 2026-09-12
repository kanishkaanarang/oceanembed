"""
Multi-Task Subsurface Ocean Model Training (SIH26066).
Predicts full 3D Temperature and Salinity profiles across 15 standard oceanic depths
simultaneously using fused multi-modal satellite surface observations (SST, SSS, SSH, Currents).
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data/processed/multimodal_training_dataset.parquet")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "../../results/multitask_ocean_model.pkl")
METRICS_OUT = os.path.join(os.path.dirname(__file__), "../../results/model_benchmark_metrics.json")

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 900]
TEMP_COLS = [f"temp_{d}m" for d in TARGET_DEPTHS]
SAL_COLS = [f"sal_{d}m" for d in TARGET_DEPTHS]
FEATURE_COLS = ["lat", "lon", "doy_sin", "doy_cos", "sst", "sss", "ssh", "current_u", "current_v", "current_speed"]

def train_and_evaluate():
    print(f"Loading multi-modal dataset from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"Dataset shape: {df.shape}")
    
    X = df[FEATURE_COLS]
    y_temp = df[TEMP_COLS]
    y_sal = df[SAL_COLS]
    y_combined = pd.concat([y_temp, y_sal], axis=1)
    
    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_combined, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    print("Training Multi-Output Gradient Boosted Ensemble...")
    base_estimator = HistGradientBoostingRegressor(max_iter=150, max_depth=8, learning_rate=0.08, random_state=42)
    model = MultiOutputRegressor(base_estimator, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Predicting on test set...")
    preds = model.predict(X_test)
    preds_df = pd.DataFrame(preds, columns=y_combined.columns, index=y_test.index)
    
    # Evaluate Temperature
    temp_preds = preds_df[TEMP_COLS].values
    temp_actual = y_test[TEMP_COLS].values
    temp_rmse = np.sqrt(mean_squared_error(temp_actual, temp_preds))
    temp_mae = mean_absolute_error(temp_actual, temp_preds)
    temp_r2 = r2_score(temp_actual, temp_preds)
    
    # Evaluate Salinity
    sal_preds = preds_df[SAL_COLS].values
    sal_actual = y_test[SAL_COLS].values
    sal_rmse = np.sqrt(mean_squared_error(sal_actual, sal_preds))
    sal_mae = mean_absolute_error(sal_actual, sal_preds)
    sal_r2 = r2_score(sal_actual, sal_preds)
    
    print("\n" + "="*50)
    print("OCEANEMBED MULTI-TASK MODEL BENCHMARK RESULTS")
    print("="*50)
    print(f"Temperature Overall RMSE: {temp_rmse:.3f} deg C (Teammate was at 3.600 deg C)")
    print(f"Temperature Overall MAE:  {temp_mae:.3f} deg C")
    print(f"Temperature Overall R2:   {temp_r2:.4f}")
    print("-" * 50)
    print(f"Salinity Overall RMSE:    {sal_rmse:.3f} PSU")
    print(f"Salinity Overall MAE:     {sal_mae:.3f} PSU")
    print(f"Salinity Overall R2:      {sal_r2:.4f}")
    print("="*50)
    
    # Depth-resolved error breakdown
    depth_metrics = []
    print("\nDepth-by-Depth Error Breakdown:")
    print(f"{'Depth':<8} | {'Temp RMSE (°C)':<15} | {'Sal RMSE (PSU)':<15}")
    print("-" * 44)
    for d in TARGET_DEPTHS:
        t_col = f"temp_{d}m"
        s_col = f"sal_{d}m"
        d_t_rmse = np.sqrt(mean_squared_error(y_test[t_col], preds_df[t_col]))
        d_s_rmse = np.sqrt(mean_squared_error(y_test[s_col], preds_df[s_col]))
        print(f"{d:>5}m   | {d_t_rmse:>13.3f}   | {d_s_rmse:>13.3f}")
        depth_metrics.append({"depth": d, "temp_rmse": round(d_t_rmse, 3), "sal_rmse": round(d_s_rmse, 3)})
        
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nSaved trained model artifact to {MODEL_OUT}")
    
    import json
    metrics = {
        "temperature": {
            "overall_rmse": round(temp_rmse, 3),
            "overall_mae": round(temp_mae, 3),
            "overall_r2": round(temp_r2, 4)
        },
        "salinity": {
            "overall_rmse": round(sal_rmse, 3),
            "overall_mae": round(sal_mae, 3),
            "overall_r2": round(sal_r2, 4)
        },
        "depth_breakdown": depth_metrics
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved benchmark metrics to {METRICS_OUT}")
    return metrics

if __name__ == "__main__":
    train_and_evaluate()
