"""
FastAPI Backend Service for OceanEmbed (SIH26066).
Serves multi-modal satellite queries, subsurface vertical profile estimations,
physics stability metrics, acoustic sound velocity profiles (SVP), and AI model validation benchmarks.
"""
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .ocean_engine import (
    predict_full_column,
    extract_satellite_surface,
    load_training_arrays,
)

from src.models.serving_engine import STANDARD_DEPTHS, get_engine

ROOT = Path(__file__).resolve().parent.parent.parent

app = FastAPI(
    title="OceanEmbed Subsurface Estimation API",
    version="3.0.0",
    description="SIH26066: AI-driven 3D ocean temperature, salinity & sound velocity reconstruction from 2D satellite observations"
)

# Enable CORS for React frontend on Vite port (5173, 5174, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProfileTier(BaseModel):
    depth: float
    temperature: float
    salinity: float
    density: float
    sound_speed: float

class PredictionResponse(BaseModel):
    location: Dict[str, float]
    date: str
    model_provenance: str
    surface_satellite_inputs: Dict[str, Any]
    thermocline_depth_estimate_m: float
    sofar_axis_depth_m: float
    hydrostatic_stability: str
    profile: List[ProfileTier]
    mission_intelligence: Optional[Dict[str, Any]] = None

@app.get("/api/health")
def health_check():
    engine = get_engine()
    return {"status": "ok", "service": "OceanEmbed Subsurface AI Engine", "version": "3.0.0",
            "model": engine.model_name, "dataset_status": engine.dataset_mode}


@app.get("/api/model/info")
def get_model_info():
    engine = get_engine()
    return {"model_name": engine.model_name, "architecture": "Depth-wise residual SE network with dual T/S heads",
            "weights": engine.model_path.name, "latent_dim": 128,
            "input_channels": engine.processor['columns'], "output_depths_m": STANDARD_DEPTHS,
            "performance": engine.report['test'],
            "evaluation_scope": "Five GLORYS days; held-out day 5; 0-900 m. Not independent Argo validation."}


@app.get("/api/model/metrics")
def get_model_benchmark_metrics():
    return get_engine().report


@app.get("/api/satellite/live")
def get_live_satellite(
    lat: float = Query(15.0, description="Latitude in degrees North"),
    lon: float = Query(88.0, description="Longitude in degrees East"),
    date: str = Query("2023-03-01", description="Observation date (YYYY-MM-DD)")
):
    """Returns live SST, SSS, SSH, wind and ocean surface current for chosen coordinates."""
    return extract_satellite_surface(lat, lon, date)

@app.get("/api/predict/profile", response_model=PredictionResponse)
def get_predicted_profile(
    lat: float = Query(15.0, description="Latitude in degrees North"),
    lon: float = Query(88.0, description="Longitude in degrees East"),
    date: str = Query("2023-03-01", description="Observation date (YYYY-MM-DD)")
):
    """
    Computes full 3D subsurface ocean column reconstruction (Temperature, Salinity, Density, Sound Speed)
    from surface satellite observations using the OceanEmbedNet v3 SE-ResNet.
    """
    try:
        result = predict_full_column(lat, lon, date)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/presets")
def get_ocean_presets():
    """Returns curated oceanographic regions in the North Indian Ocean."""
    return [
        {
            "id": "central_bob",
            "name": "Central Bay of Bengal",
            "category": "Open Ocean",
            "lat": 15.0,
            "lon": 88.0,
            "description": "Stratified warm pool with distinct seasonal thermocline and internal waves."
        },
        {
            "id": "ganga_plume",
            "name": "Northern Bay (Ganges Plume)",
            "category": "River Discharge",
            "lat": 19.5,
            "lon": 89.0,
            "description": "Massive freshwater river discharge creating strong surface barrier layer."
        },
        {
            "id": "eicc_boundary",
            "name": "Western Bay / EICC Current",
            "category": "Boundary Current",
            "lat": 13.5,
            "lon": 83.5,
            "description": "East India Coastal Current domain with active coastal upwelling and eddy dynamics."
        },
        {
            "id": "andaman_sea",
            "name": "Andaman Sea Basin",
            "category": "Marginal Sea",
            "lat": 11.5,
            "lon": 94.5,
            "description": "Semi-enclosed basin with deep pycnocline and high internal wave activity."
        },
        {
            "id": "central_arabian_sea",
            "name": "Central Arabian Sea",
            "category": "High Salinity",
            "lat": 16.0,
            "lon": 66.0,
            "description": "Arabian Sea High Salinity Water (ASHSW) with intense winter convective cooling."
        },
        {
            "id": "oman_upwelling",
            "name": "Oman Upwelling Zone",
            "category": "Upwelling",
            "lat": 20.0,
            "lon": 59.0,
            "description": "Strong summer monsoon wind-driven upwelling bringing cold nutrient-rich waters to surface."
        },
        {
            "id": "equatorial_channel",
            "name": "Equatorial Indian Ocean",
            "category": "Equatorial",
            "lat": 7.0,
            "lon": 86.0,
            "description": "Deep ocean conduit with equatorial jets and Wyrtki jets during monsoon transitions."
        }
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8005, reload=True)
