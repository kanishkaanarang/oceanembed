"""
FastAPI Backend Service for OceanEmbed (SIH26066).
Serves multi-modal satellite queries, subsurface vertical profile estimations,
physics stability metrics, and naval sound velocity profiles.
"""
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from .ocean_engine import predict_full_column, extract_satellite_surface

app = FastAPI(
    title="OceanEmbed Subsurface Estimation API",
    version="1.0.0",
    description="SIH26066: AI-driven 3D ocean temperature & salinity profile reconstruction from 2D satellite observations"
)

# Enable CORS for React frontend on Port 5174
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
    surface_satellite_inputs: Dict[str, Any]
    thermocline_depth_estimate_m: float
    hydrostatic_stability: str
    profile: List[ProfileTier]

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "OceanEmbed Subsurface API", "version": "1.0.0"}

@app.get("/api/satellite/live")
def get_live_satellite(
    lat: float = Query(15.0, description="Latitude in degrees North"),
    lon: float = Query(88.0, description="Longitude in degrees East"),
    date: str = Query("2023-01-05", description="Observation date (YYYY-MM-DD)")
):
    """Returns live SST, SSS, SSH, and ocean surface velocity from CMEMS."""
    return extract_satellite_surface(lat, lon, date)

@app.get("/api/predict/profile", response_model=PredictionResponse)
def get_predicted_profile(
    lat: float = Query(15.0, description="Latitude in degrees North"),
    lon: float = Query(88.0, description="Longitude in degrees East"),
    date: str = Query("2023-01-05", description="Observation date (YYYY-MM-DD)")
):
    """
    Computes full 3D subsurface column estimation (Temperature, Salinity, Density, Sound Speed)
    using fused 2D satellite surface observations.
    """
    try:
        result = predict_full_column(lat, lon, date)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8005, reload=True)
