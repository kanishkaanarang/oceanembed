"""OceanEmbed: Deep Learning 3D Ocean State Reconstruction Platform."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.ui.mission_panel import render_mission_intelligence, multi_variable_figure
from src.ui.theme import themed_table, render_chart
from src.models.mission_metrics import export_profile_to_netcdf

from backend import (
    ANALYSIS_END,
    ANALYSIS_START,
    CYCLONE_EVENT_PRESETS,
    DEPTHS,
    LATITUDE_RANGE,
    LONGITUDE_RANGE,
    REGIONAL_PRESETS,
    clear_saved_locations,
    field,
    generate_profile_interpretation,
    get_depth_metrics,
    get_engine,
    get_surface_conditions,
    list_saved_locations,
    load_catalogue,
    nearby_observations,
    reconstruct,
    save_location,
    transect,
    validation_summary,
)

st.set_page_config(
    page_title="OceanEmbed | 3D Deep Ocean Reconstruction",
    page_icon=":material/waves:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.html(
    """
    <style>
    [data-testid="stMainBlockContainer"] { padding-top: 4rem; padding-bottom: 2rem; }
    [data-testid="stSidebarUserContent"] { padding-top: 0; }
    [data-testid="stSidebar"] hr { margin: .5rem 0; }
    .st-key-workspace_page label { border-radius: 8px; padding: 5px 8px; margin: 0; }
    .st-key-workspace_page label:has(input:checked) { background: rgba(2,132,199,.12); }
    [data-testid="stMetricValue"] { font-size: clamp(1.35rem, 2vw, 1.85rem); }
    [data-testid="stMetricValue"] > div, [data-testid="stMetricLabel"] p {
      white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
    }
    [data-testid="stMainBlockContainer"] h3 { font-size: 1.35rem; line-height: 1.4; }
    @media (max-width: 700px) {
      [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
      .st-key-hero { padding: 1rem !important; }
    }
    @media (prefers-reduced-motion: no-preference) {
      @keyframes oceanembed-rise {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @keyframes oceanembed-tide {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
      }
      @keyframes oceanembed-pulse {
        0%, 100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--primary-color) 22%, transparent); }
        50% { box-shadow: 0 0 0 9px color-mix(in srgb, var(--primary-color) 0%, transparent); }
      }
      .st-key-hero { animation: oceanembed-rise 560ms cubic-bezier(.2,.8,.2,1) both, oceanembed-tide 14s ease-in-out infinite; }
      .st-key-metrics [data-testid="stMetric"], .st-key-depth-cards [data-testid="stMetric"], .st-key-surface-cards [data-testid="stMetric"] {
        animation: oceanembed-rise 540ms cubic-bezier(.2,.8,.2,1) both;
      }
      [data-testid="stTabs"] {
        animation: oceanembed-rise 640ms 240ms cubic-bezier(.2,.8,.2,1) both;
      }
      .st-key-map-card, .st-key-profile-card, .st-key-depth-table-card,
      .st-key-casts-card, .st-key-validation-card, .st-key-catalogue-card,
      .st-key-data-card, .st-key-saved-card, .st-key-quality-notes, .st-key-explore-guide,
      .st-key-summary-card, .st-key-hotspot-card, .st-key-sounding-inspect {
        animation: oceanembed-rise 620ms cubic-bezier(.2,.8,.2,1) both;
        transition: transform 220ms ease, box-shadow 220ms ease, border-color 220ms ease;
      }
      .st-key-map-card:hover, .st-key-profile-card:hover, .st-key-depth-table-card:hover,
      .st-key-casts-card:hover, .st-key-validation-card:hover, .st-key-catalogue-card:hover,
      .st-key-data-card:hover, .st-key-saved-card:hover, .st-key-quality-notes:hover,
      .st-key-explore-guide:hover, .st-key-summary-card:hover, .st-key-sounding-inspect:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 32px color-mix(in srgb, var(--primary-color) 16%, transparent);
        border-color: color-mix(in srgb, var(--primary-color) 46%, var(--border-color));
      }
      .st-key-hero-status { animation: oceanembed-rise 680ms 120ms cubic-bezier(.2,.8,.2,1) both, oceanembed-pulse 3.2s 820ms ease-in-out infinite; }
    }
    .st-key-hero {
      background: radial-gradient(circle at 88% 18%, color-mix(in srgb, var(--primary-color) 25%, transparent), transparent 26%), linear-gradient(115deg, color-mix(in srgb, var(--primary-color) 14%, var(--background-color)), var(--background-color) 62%, transparent);
      background-size: 170% 170%;
      border: 1px solid color-mix(in srgb, var(--primary-color) 18%, var(--background-color));
      border-radius: 20px;
      padding: 1.4rem 1.6rem 1.1rem;
      margin-bottom: 0.6rem;
    }
    .st-key-hero h1 { letter-spacing: -0.045em; line-height: 1.05; font-size: 2.1rem; }
    .st-key-hero-status {
      background: color-mix(in srgb, var(--background-color) 72%, transparent);
      border-color: color-mix(in srgb, var(--primary-color) 26%, var(--border-color));
      border-radius: 16px;
    }
    .st-key-hero-meta { padding-top: 0.2rem; }
    .st-key-metrics [data-testid="stMetric"], .st-key-depth-cards [data-testid="stMetric"], .st-key-surface-cards [data-testid="stMetric"] {
      background: linear-gradient(145deg, var(--background-color), var(--secondary-background-color));
      border-radius: 14px;
      box-shadow: 0 8px 20px rgba(18, 38, 48, 0.06);
    }
    [data-testid="stTabs"] [role="tab"] {
      border-radius: 10px 10px 0 0;
      transition: background 180ms ease, color 180ms ease, transform 180ms ease;
    }
    [data-testid="stTabs"] [role="tab"]:hover {
      background: color-mix(in srgb, var(--primary-color) 10%, transparent);
      transform: translateY(-1px);
    }
    .st-key-data-card, .st-key-saved-card { background: color-mix(in srgb, var(--secondary-background-color) 72%, transparent); }
    .st-key-explore-guide, .st-key-summary-card { background: color-mix(in srgb, var(--primary-color) 5%, var(--background-color)); }
    .st-key-save-location button, .st-key-grid-download button, .st-key-profile-download button {
      border-radius: 999px;
      transition: transform 180ms ease, box-shadow 180ms ease;
    }
    .st-key-save-location button:hover, .st-key-grid-download button:hover, .st-key-profile-download button:hover {
      transform: translateY(-1px);
      box-shadow: 0 8px 18px color-mix(in srgb, var(--primary-color) 20%, transparent);
    }
    .sounding-row-active {
      background-color: color-mix(in srgb, var(--primary-color) 20%, transparent) !important;
      font-weight: 700;
    }
    </style>
    """
)

COASTAL_LANDMARKS = [
    ("Mumbai", 18.92, 72.83),
    ("Chennai", 13.08, 80.27),
    ("Kolkata", 22.57, 88.36),
    ("Kochi", 9.93, 76.26),
    ("Colombo", 6.92, 79.86),
    ("Karachi", 24.86, 67.00),
    ("Muscat", 23.58, 58.40),
    ("Malé", 4.17, 73.50),
    ("Port Blair", 11.62, 92.72),
    ("Phuket", 7.88, 98.39),
]


def initialize_state() -> None:
    defaults = {
        "workspace_page": "Overview",
        "dark_mode": False,  # Default to Light theme!
        "analysis_date": date(2023, 3, 1),
        "latitude": 14.5,
        "longitude": 88.0,
        "depth": 100,
        "variable": "Temperature",
        "run_count": 0,
        "comparison_enabled": False,
        "comparison_date": date(2023, 4, 15),
        "comparison_latitude": 16.0,
        "comparison_longitude": 66.0,
        "active_preset": "central_bob",
        "region_preset": "central_bob",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


initialize_state()
if st.session_state.get("explore_view_mode") == "Vertical Transect Curtain (0–1000m)":
    st.session_state.explore_view_mode = "Vertical Transect Curtain (0–900m)"
# Keep optional widget values when their page is not mounted.
for state_key in ("mission-svp-preview", "mission-ray-preview", "explore_view_mode", "transect_axis_choice",
                  "comparison_date", "comparison_latitude", "comparison_longitude"):
    if state_key in st.session_state:
        st.session_state[state_key] = st.session_state[state_key]

PAGES = ("Overview", "Ocean Explorer", "Profiles & Data", "Mission Intelligence", "Validation", "About & Saved Places")

def navigate_to(destination):
    st.session_state.workspace_page = destination

is_dark = st.session_state.get("dark_mode", False)
if is_dark:
    st.html(
        """
        <style>
        [data-testid="stAppViewContainer"], .stApp {
          background-color: #071b2a !important;
          color: #e7f4f8 !important;
        }
        [data-testid="stSidebar"] {
          background-color: #0c2433 !important;
          color: #e7f4f8 !important;
          border-right: 1px solid #1a425a !important;
        }
        [data-testid="stWidgetLabel"] p,
        [data-testid="stRadio"] label p,
        [data-testid="stCheckbox"] label p {
          color: #e7f4f8 !important;
        }
        [data-testid="stTooltipIcon"] svg { fill: #94a3b8 !important; }
        .stMarkdownBadge, [data-testid="stMetricDelta"] {
          color: #e7f4f8 !important;
        }
        [data-testid="stMarkdownContainer"] code {
          color: #a5f3fc !important;
          background-color: #113348 !important;
        }
        [data-testid="stSelectbox"] input,
        [data-testid="stSelectbox"] div:has(> input[role="combobox"]),
        [data-testid="stSelectboxVirtualDropdown"],
        [role="listbox"], [role="option"] {
          background-color: #113348 !important;
          color: #e7f4f8 !important;
          border-color: #356077 !important;
        }
        [role="option"][aria-selected="true"], [role="option"][data-focused="true"] {
          background-color: #1a4b65 !important;
        }
        [data-testid="stSelectbox"] button {
          color: #e7f4f8 !important;
          background: #113348 !important;
        }
        [data-testid="stSidebarCollapseButton"] button { color: #bce9fc !important; }
        [data-testid="stAlert"] p { color: #bce9fc !important; }
        [data-testid="stButton"] button,
        [data-testid="stDownloadButton"] button {
          background: #113348 !important;
          border-color: #356077 !important;
          color: #e7f4f8 !important;
        }
        [data-testid="stHeader"] {
          background-color: rgba(7, 27, 42, 0.85) !important;
        }
        .st-key-hero {
          background: radial-gradient(circle at 88% 18%, rgba(0, 210, 255, 0.22), transparent 26%), linear-gradient(115deg, #0d2a3e, #071b2a 62%) !important;
          border: 1px solid #1a425a !important;
          box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4) !important;
        }
        .st-key-hero h1, .st-key-hero p, .st-key-hero span {
          color: #e7f4f8 !important;
        }
        .st-key-metrics [data-testid="stMetric"], 
        .st-key-depth-cards [data-testid="stMetric"], 
        .st-key-surface-cards [data-testid="stMetric"], 
        .st-key-comparison-cards [data-testid="stMetric"] {
          background: linear-gradient(145deg, #0d2a3e, #092030) !important;
          border: 1px solid #1a425a !important;
          color: #e7f4f8 !important;
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25) !important;
          border-radius: 12px !important;
        }
        [data-testid="stMetricLabel"] p {
          color: #94a3b8 !important;
          font-weight: 500 !important;
        }
        [data-testid="stMetricValue"] {
          color: #f1f5f9 !important;
          font-weight: 700 !important;
        }
        .st-key-map-card, .st-key-profile-card, .st-key-depth-table-card,
        .st-key-casts-card, .st-key-validation-card, .st-key-catalogue-card,
        .st-key-data-card, .st-key-saved-card, .st-key-quality-notes, .st-key-explore-guide,
        .st-key-summary-card, .st-key-hotspot-card, .st-key-sounding-inspect,
        .st-key-mission-cyclone-card, .st-key-mission-acoustic-card, .st-key-mission-physics-card,
        .st-key-transect-card, .st-key-transect-meta-card, .st-key-hero-status {
          background: #0c2433 !important;
          border: 1px solid #1a425a !important;
          color: #e7f4f8 !important;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
          border-radius: 14px !important;
        }
        .st-key-summary-card p, .st-key-summary-card strong {
          color: #e7f4f8 !important;
        }
        [data-testid="stTabs"] [role="tab"] {
          color: #94a3b8 !important;
          font-weight: 500 !important;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
          color: #00d2ff !important;
          border-bottom: 2px solid #00d2ff !important;
          font-weight: 600 !important;
        }
        [data-testid="stExpander"] {
          background-color: #0c2433 !important;
          border: 1px solid #1a425a !important;
          border-radius: 12px !important;
        }
        [data-testid="stExpander"] details {
          background-color: #0c2433 !important;
        }
        [data-testid="stExpander"] summary {
          background-color: #0d2a3e !important;
          color: #e7f4f8 !important;
          border-radius: 12px 12px 0 0 !important;
          font-weight: 600 !important;
        }
        </style>
        """
    )
else:
    st.html(
        """
        <style>
        [data-testid="stAppViewContainer"], .stApp {
          background-color: #ffffff !important;
          color: #0f172a !important;
        }
        [data-testid="stSidebar"] {
          background-color: #ffffff !important;
          color: #0f172a !important;
          border-right: 1px solid #e2e8f0 !important;
        }
        [data-testid="stHeader"] {
          background-color: rgba(255, 255, 255, 0.9) !important;
        }
        .st-key-hero {
          background: linear-gradient(135deg, #f0f9ff 0%, #ffffff 100%) !important;
          border: 1px solid #bae6fd !important;
          box-shadow: 0 4px 20px rgba(2, 132, 199, 0.06) !important;
        }
        .st-key-hero h1 {
          color: #0369a1 !important;
        }
        .st-key-hero p, .st-key-hero span {
          color: #334155 !important;
        }
        .st-key-metrics [data-testid="stMetric"], 
        .st-key-depth-cards [data-testid="stMetric"], 
        .st-key-surface-cards [data-testid="stMetric"], 
        .st-key-comparison-cards [data-testid="stMetric"] {
          background: #ffffff !important;
          border: 1px solid #e2e8f0 !important;
          color: #0f172a !important;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
          border-radius: 12px !important;
        }
        [data-testid="stMetricLabel"] p {
          color: #64748b !important;
          font-weight: 500 !important;
        }
        [data-testid="stMetricValue"] {
          color: #0f172a !important;
          font-weight: 700 !important;
        }
        .st-key-map-card, .st-key-profile-card, .st-key-depth-table-card,
        .st-key-casts-card, .st-key-validation-card, .st-key-catalogue-card,
        .st-key-data-card, .st-key-saved-card, .st-key-quality-notes, .st-key-explore-guide,
        .st-key-summary-card, .st-key-hotspot-card, .st-key-sounding-inspect,
        .st-key-mission-cyclone-card, .st-key-mission-acoustic-card, .st-key-mission-physics-card,
        .st-key-transect-card, .st-key-transect-meta-card, .st-key-hero-status {
          background: #ffffff !important;
          border: 1px solid #e2e8f0 !important;
          color: #0f172a !important;
          box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03) !important;
          border-radius: 14px !important;
        }
        .st-key-summary-card p, .st-key-summary-card strong {
          color: #1e293b !important;
        }
        [data-testid="stTabs"] [role="tab"] {
          color: #64748b !important;
          font-weight: 500 !important;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
          color: #0284c7 !important;
          border-bottom: 2px solid #0284c7 !important;
          font-weight: 600 !important;
        }
        [data-testid="stExpander"] {
          background-color: #ffffff !important;
          border: 1px solid #e2e8f0 !important;
          border-radius: 12px !important;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02) !important;
        }
        [data-testid="stExpander"] details {
          background-color: #ffffff !important;
        }
        [data-testid="stExpander"] summary {
          background-color: #f8fafc !important;
          color: #0f172a !important;
          border-radius: 12px 12px 0 0 !important;
          font-weight: 600 !important;
        }
        </style>
        """
    )


def record_interaction() -> None:
    st.session_state.run_count += 1


def on_coordinate_change() -> None:
    st.session_state.active_preset = "custom"
    st.session_state.region_preset = "custom"
    record_interaction()


def apply_preset(preset_id: str) -> None:
    all_presets = list(REGIONAL_PRESETS) + list(CYCLONE_EVENT_PRESETS)
    for item in all_presets:
        if item["id"] == preset_id:
            st.session_state.latitude = float(item["latitude"])
            st.session_state.longitude = float(item["longitude"])
            if "date" in item:
                st.session_state.analysis_date = item["date"]
            st.session_state.active_preset = preset_id
            st.session_state.region_preset = preset_id
            record_interaction()
            break


def temperature_at(profile: pd.DataFrame, depth_m: int) -> float:
    ordered = profile.sort_values("depth_m")
    return float(np.interp(depth_m, ordered["depth_m"], ordered["temperature_c"]))


def calculate_mld_and_thermocline(profile: pd.DataFrame) -> tuple[float, float]:
    ordered = profile.sort_values("depth_m")
    depths = ordered["depth_m"].to_numpy()
    temps = ordered["temperature_c"].to_numpy()

    surface_temp = temps[0]
    mld = 35.0
    for d, t in zip(depths, temps):
        if (surface_temp - t) >= 0.5:
            mld = float(d)
            break

    dT = np.diff(temps)
    dz = np.diff(depths)
    dz = np.where(dz == 0, 1e-3, dz)
    gradients = np.abs(dT / dz)
    max_idx = int(np.argmax(gradients))
    thermocline_depth = float((depths[max_idx] + depths[max_idx + 1]) / 2.0)

    return mld, thermocline_depth


def update_location_from_map_selection() -> None:
    selection = st.session_state.get("map_selection", {})
    points = selection.get("selection", {}).get("points", [])
    if not points:
        return
    point = points[-1]
    try:
        longitude, latitude = float(point["x"]), float(point["y"])
    except (KeyError, TypeError, ValueError):
        return
    if LONGITUDE_RANGE[0] <= longitude <= LONGITUDE_RANGE[1] and LATITUDE_RANGE[0] <= latitude <= LATITUDE_RANGE[1]:
        st.session_state.longitude = round(longitude, 2)
        st.session_state.latitude = round(latitude, 2)
        st.session_state.active_preset = "custom"
        st.session_state.region_preset = "custom"
        record_interaction()


def plot_colors() -> dict[str, str]:
    is_dark = st.session_state.get("dark_mode", False)
    return {
        "paper": "#071b2a" if is_dark else "#ffffff",
        "panel": "#0c2433" if is_dark else "#ffffff",
        "text": "#e7f4f8" if is_dark else "#0f172a",
        "grid": "#1a425a" if is_dark else "#f1f5f9",
        "model": "#00d2ff" if is_dark else "#0284c7",
        "reference": "#ff9e00" if is_dark else "#d97706",
        "temperature": "#fdba74" if is_dark else "#c2410c",
        "salinity": "#5eead4" if is_dark else "#047857",
        "acoustic": "#7dd3fc" if is_dark else "#0369a1",
    }


def map_figure(grid: pd.DataFrame, latitude: float, longitude: float, variable: str, colors: dict[str, str]) -> go.Figure:
    matrix = grid.pivot(index="latitude", columns="longitude", values="value")
    unit = grid["unit"].iat[0]

    if variable == "Salinity":
        scale = "Tealgrn"
    elif variable == "Confidence":
        scale = "Viridis"
    elif "Error" in variable or "Residual" in variable:
        scale = "Reds"
    elif "Embedding" in variable:
        scale = "Plasma"
    elif "Density" in variable:
        scale = "Blues"
    elif "Sound" in variable:
        scale = "Magma"
    else:
        scale = "Thermal"

    figure = go.Figure()

    figure.add_trace(go.Heatmap(
        x=matrix.columns,
        y=matrix.index,
        z=matrix.values,
        colorscale=scale,
        colorbar=dict(title=unit, thickness=14, len=0.82, tickfont=dict(color=colors["text"])),
        hovertemplate=f"Longitude %{{x:.2f}}°E<br>Latitude %{{y:.2f}}°N<br>{variable}: %{{z:.2f}} {unit}<extra></extra>",
        connectgaps=False,
    ))

    # Heatmap cells do not emit Streamlit's Plotly point-selection events.
    # A transparent scatter layer makes every valid ocean cell selectable.
    selectable = grid.loc[np.isfinite(grid["value"])]
    figure.add_trace(go.Scattergl(
        x=selectable["longitude"], y=selectable["latitude"],
        customdata=selectable["value"], mode="markers", name="Ocean cells",
        marker=dict(size=7, color="rgba(2,132,199,0.01)"),
        selected=dict(marker=dict(opacity=0.8, color=colors["model"])),
        hovertemplate=f"Longitude %{{x:.2f}}°E<br>Latitude %{{y:.2f}}°N<br>{variable}: %{{customdata:.2f}} {unit}<extra></extra>",
        showlegend=False,
    ))

    lats = [item[1] for item in COASTAL_LANDMARKS]
    lons = [item[2] for item in COASTAL_LANDMARKS]
    names = [item[0] for item in COASTAL_LANDMARKS]

    figure.add_trace(go.Scatter(
        x=lons,
        y=lats,
        mode="markers+text",
        name="Coastal landmarks",
        text=names,
        textposition="top center",
        textfont=dict(size=10, color=colors["text"]),
        marker=dict(size=6, color="#ffaa00", symbol="circle", line=dict(color="#111", width=1)),
        hoverinfo="text",
        showlegend=False,
    ))

    figure.add_trace(go.Scatter(
        x=[longitude],
        y=[latitude],
        mode="markers",
        name="Target sounding",
        marker=dict(size=15, color="#ffffff", symbol="cross", line=dict(color=colors["model"], width=3)),
        hovertemplate="Target sounding location<br>Lat: %{y:.2f}°N, Lon: %{x:.2f}°E<extra></extra>",
    ))

    figure.update_layout(
        height=480,
        margin=dict(l=5, r=10, t=10, b=5),
        paper_bgcolor=colors["paper"],
        plot_bgcolor=colors["panel"],
        font=dict(color=colors["text"]),
        xaxis=dict(title="Longitude (°E)", range=LONGITUDE_RANGE, gridcolor=colors["grid"], zeroline=False),
        yaxis=dict(title="Latitude (°N)", range=LATITUDE_RANGE, gridcolor=colors["grid"], zeroline=False),
        showlegend=False,
        clickmode="event+select",
    )
    return figure


def profile_figure(profile: pd.DataFrame, variable: str, colors: dict[str, str], comparison_profile: pd.DataFrame | None = None) -> go.Figure:
    single_variables = {
        "Confidence": ("confidence_pct", "Model confidence", "%"),
        "Ground Truth": ("argo_temperature_c", "GLORYS temperature", "°C"),
        "Residual": ("error_c", "Absolute temperature residual", "°C"),
        "Density": ("density_kg_m3", "Seawater density", "kg/m³"),
        "Sound Speed": ("sound_speed_m_s", "Sound speed", "m/s"),
    }
    if variable in single_variables:
        column, title, unit = single_variables[variable]
        figure = go.Figure()
        for frame, label, color, dash in [
            (profile, title, colors["model"], "solid"),
            (comparison_profile, "Comparison sounding", colors["reference"], "dash"),
        ]:
            if frame is not None:
                figure.add_trace(go.Scatter(
                    x=frame[column], y=frame["depth_m"], mode="lines+markers", name=label,
                    line=dict(color=color, width=3, dash=dash), marker=dict(size=5),
                    hovertemplate=f"Depth: %{{y}} m<br>{title}: %{{x:.2f}} {unit}<extra></extra>",
                ))
        figure.update_layout(
            height=480, margin=dict(l=5, r=5, t=35, b=5),
            paper_bgcolor=colors["paper"], plot_bgcolor=colors["panel"],
            font=dict(color=colors["text"]), legend=dict(orientation="h", y=1.14, x=0),
            xaxis=dict(title=f"{title} ({unit})", gridcolor=colors["grid"],
                       range=[50, 100] if variable == "Confidence" else None),
            yaxis=dict(title="Depth (m)", autorange="reversed", gridcolor=colors["grid"]),
        )
        return figure
    if variable == "Salinity":
        estimate, lower, upper, reference, unit = "salinity_psu", "salinity_lower_psu", "salinity_upper_psu", "argo_salinity_psu", "PSU"
    else:
        estimate, lower, upper, reference, unit = "temperature_c", "temperature_lower_c", "temperature_upper_c", "argo_temperature_c", "°C"

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=profile[upper], y=profile["depth_m"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
    figure.add_trace(go.Scatter(
        x=profile[lower],
        y=profile["depth_m"],
        mode="lines",
        line=dict(width=0),
        fill="tonextx",
        fillcolor="rgba(0, 180, 216, 0.18)",
        name="Recorded test RMSE band",
        hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        x=profile[estimate],
        y=profile["depth_m"],
        mode="lines+markers",
        name="OceanEmbedNet v3",
        line=dict(color=colors["model"], width=3.5),
        marker=dict(size=6),
        hovertemplate=f"Depth: %{{y}}m<br>Estimate: %{{x:.2f}} {unit}<extra></extra>",
    ))
    if profile[reference].notna().any():
        figure.add_trace(go.Scatter(
            x=profile[reference],
            y=profile["depth_m"],
            mode="lines+markers",
            name="GLORYS reference (900 m interpolated)",
            line=dict(color=colors["reference"], width=2.2, dash="dot"),
            marker=dict(size=5),
            hovertemplate=f"Depth: %{{y}}m<br>Ground truth: %{{x:.2f}} {unit}<extra></extra>",
        ))
    if comparison_profile is not None:
        figure.add_trace(go.Scatter(
            x=comparison_profile[estimate],
            y=comparison_profile["depth_m"],
            mode="lines",
            name="Comparison sounding",
            line=dict(color="#a855f7", width=2.5, dash="dash"),
            hovertemplate=f"Depth: %{{y}}m<br>Compare: %{{x:.2f}} {unit}<extra></extra>",
        ))

    figure.update_layout(
        height=480,
        margin=dict(l=5, r=5, t=35, b=5),
        paper_bgcolor=colors["paper"],
        plot_bgcolor=colors["panel"],
        font=dict(color=colors["text"]),
        legend=dict(orientation="h", y=1.14, x=0),
        xaxis=dict(title=f"{variable} ({unit})", gridcolor=colors["grid"]),
        yaxis=dict(title="Depth (m)", autorange="reversed", gridcolor=colors["grid"]),
    )
    return figure

def transect_figure(transect_data: dict, variable: str, colors: dict[str, str]) -> go.Figure:
    fig = go.Figure()
    unit = transect_data.get("unit", "°C")
    scale = "Thermal"
    if "Salinity" in variable:
        scale = "Tealgrn"
    elif "Density" in variable:
        scale = "Blues"
    elif "Sound" in variable:
        scale = "Magma"
    elif "Residual" in variable or "Error" in variable:
        scale = "Reds"

    fig.add_trace(go.Contour(
        x=transect_data["coords"],
        y=transect_data["depths"],
        z=transect_data["values"],
        colorscale=scale,
        colorbar=dict(title=unit, thickness=14, len=0.82, tickfont=dict(color=colors["text"])),
        contours=dict(coloring="heatmap", showlabels=True, labelfont=dict(size=10, color=colors["text"])),
        hovertemplate=f"{transect_data['axis_label']}: %{{x:.2f}}<br>Depth: %{{y}}m<br>{variable}: %{{z:.2f}} {unit}<extra></extra>",
        connectgaps=False,
    ))
    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor=colors["paper"],
        plot_bgcolor=colors["panel"],
        font=dict(color=colors["text"]),
        xaxis=dict(title=transect_data["axis_label"], gridcolor=colors["grid"], zeroline=False),
        yaxis=dict(title="Depth (m)", autorange="reversed", gridcolor=colors["grid"]),
    )
    return fig



def validation_figure(depth_metrics: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Bar(
        x=[f"{d}m" for d in depth_metrics["depth_m"]],
        y=depth_metrics["rmse_c"],
        name="RMSE (°C)",
        marker_color=colors["model"],
    ))
    figure.add_trace(go.Bar(
        x=[f"{d}m" for d in depth_metrics["depth_m"]],
        y=depth_metrics["mae_c"],
        name="MAE (°C)",
        marker_color=colors["reference"],
    ))
    figure.update_layout(
        height=340,
        margin=dict(l=5, r=5, t=25, b=5),
        paper_bgcolor=colors["paper"],
        plot_bgcolor=colors["panel"],
        font=dict(color=colors["text"]),
        barmode="group",
        legend=dict(orientation="h", y=1.12, x=0),
        xaxis=dict(title="Standard Depth (m)", gridcolor=colors["grid"]),
        yaxis=dict(title="Validation Error (°C)", range=[0, 1.6], gridcolor=colors["grid"]),
    )
    return figure


def save_current_location() -> None:
    label = f"Target {len(list_saved_locations()) + 1:02d}"
    is_new = save_location(label, st.session_state.latitude, st.session_state.longitude)
    if is_new:
        st.toast("Location saved to local database.", icon=":material/bookmark_added:")
    else:
        st.toast("That location is already saved.", icon=":material/bookmark:")


def clear_locations() -> None:
    clear_saved_locations()
    st.toast("Saved locations cleared.", icon=":material/delete_sweep:")


catalogue = load_catalogue()
depth_metrics = get_depth_metrics()
analysis_dates = list(pd.date_range(ANALYSIS_START, ANALYSIS_END, freq="D").date)

with st.sidebar:
    st.title("OceanEmbed", icon=":material/waves:")
    st.caption("Subsurface ocean reconstruction")
    engine_obj = get_engine()
    if st.session_state.get("depth") not in list(DEPTHS):
        st.session_state.depth = int(DEPTHS[-1])
    dataset_label = "150-Day Reanalysis" if not engine_obj.is_compact else "Compact Reference Dataset"
    st.badge(f"PyTorch v3 · {dataset_label}", icon=":material/bolt:", color="green")
    st.toggle(
        "Dark mode",
        key="dark_mode",
        on_change=record_interaction,
        help="Switch between Light and Dark interface styles",
    )

    st.radio("Workspace", PAGES, key="workspace_page")
    st.divider()
    st.subheader("Regional Hotspots")
    all_presets_list = list(REGIONAL_PRESETS) + list(CYCLONE_EVENT_PRESETS)
    preset_names = {p["id"]: f"{p['name']} ({p.get('badge', 'Mission')})" for p in all_presets_list}
    preset_names["custom"] = "Custom Coordinates"
    all_preset_options = ["custom"] + [p["id"] for p in REGIONAL_PRESETS] + [p["id"] for p in CYCLONE_EVENT_PRESETS]

    def on_select_preset() -> None:
        choice = st.session_state.region_preset
        if choice != "custom":
            apply_preset(choice)
        else:
            st.session_state.active_preset = "custom"
            record_interaction()

    st.selectbox(
        "Jump to key oceanographic regime:",
        options=all_preset_options,
        format_func=lambda x: preset_names.get(x, x),
        key="region_preset",
        on_change=on_select_preset,
    )

    st.subheader("Reconstruction controls")
    st.select_slider(
        "Observation date",
        options=analysis_dates,
        format_func=lambda item: item.strftime("%d %b %Y"),
        key="analysis_date",
        on_change=on_coordinate_change,
    )
    st.slider("Latitude", *LATITUDE_RANGE, step=0.25, format="%.2f° N", key="latitude", on_change=on_coordinate_change)
    st.slider("Longitude", *LONGITUDE_RANGE, step=0.25, format="%.2f° E", key="longitude", on_change=on_coordinate_change)
    st.select_slider("Depth slice", options=list(map(int, DEPTHS)), format_func=lambda item: f"{item} m", key="depth", on_change=record_interaction)
    st.selectbox(
        "Map layer",
        ["Temperature", "Ground Truth", "Residual", "Salinity", "Density", "Sound Speed", "Embedding", "Confidence"],
        key="variable",
        on_change=record_interaction,
    )

    st.toggle("Compare second point", key="comparison_enabled", on_change=record_interaction)
    if st.session_state.comparison_enabled:
        with st.expander("Comparison location", icon=":material/compare_arrows:", expanded=True):
            st.select_slider("Comparison date", options=analysis_dates, format_func=lambda item: item.strftime("%d %b %Y"), key="comparison_date", on_change=record_interaction)
            st.slider("Comparison latitude", *LATITUDE_RANGE, step=0.25, format="%.2f° N", key="comparison_latitude", on_change=record_interaction)
            st.slider("Comparison longitude", *LONGITUDE_RANGE, step=0.25, format="%.2f° E", key="comparison_longitude", on_change=record_interaction)

    st.button("Save current location", icon=":material/bookmark_add:", width="stretch", on_click=save_current_location, key="save-location")

    with st.expander("Loaded model & data", icon=":material/verified_user:"):
        st.write(f"**Serving model:** {engine_obj.model_name}")
        st.caption(f"Weights: {Path(engine_obj.model_path).name if engine_obj.model_loaded else 'Not loaded'}")
        st.caption(f"Data mode: {engine_obj.dataset_mode}")
        st.caption("V3 predicts temperature and salinity at 15 depths through 900 m. Limited evaluation scope is documented on Validation.")



analysis_date = st.session_state.analysis_date
latitude = st.session_state.latitude
longitude = st.session_state.longitude
depth = st.session_state.depth
variable = st.session_state.variable
colors = plot_colors()

surface = get_surface_conditions(latitude, longitude, analysis_date)
profile = reconstruct(latitude, longitude, analysis_date)
grid = field(latitude, longitude, depth, variable, analysis_date) if st.session_state.workspace_page == "Ocean Explorer" else None
observations = nearby_observations(latitude, longitude, analysis_date) if st.session_state.workspace_page == "Profiles & Data" else None
validation = validation_summary()

comparison_profile = None
comparison_surface = None
if st.session_state.comparison_enabled:
    comparison_profile = reconstruct(st.session_state.comparison_latitude, st.session_state.comparison_longitude, st.session_state.comparison_date)
    comparison_surface = get_surface_conditions(st.session_state.comparison_latitude, st.session_state.comparison_longitude, st.session_state.comparison_date)

mld, thermocline = calculate_mld_and_thermocline(profile)
narrative = generate_profile_interpretation(profile, surface, latitude, longitude, analysis_date)

# Locate current depth row
depth_row = profile.loc[profile["depth_m"] == depth].iloc[0] if (profile["depth_m"] == depth).any() else profile.iloc[0]

mission_context = {
    "date": analysis_date.isoformat(),
    "requested_location": {"latitude": latitude, "longitude": longitude},
    "sampled_location": {"latitude": surface["latitude"], "longitude": surface["longitude"]},
    "dataset_mode": engine_obj.dataset_mode,
    "model": engine_obj.model_name,
}
comparison_context = None
if comparison_surface is not None:
    comparison_context = {
        "date": st.session_state.comparison_date.isoformat(),
        "requested_location": {"latitude": st.session_state.comparison_latitude, "longitude": st.session_state.comparison_longitude},
        "sampled_location": {"latitude": comparison_surface["latitude"], "longitude": comparison_surface["longitude"]},
        "dataset_mode": engine_obj.dataset_mode,
        "model": mission_context["model"],
    }
if surface["latitude"] != latitude or surface["longitude"] != longitude:
    st.caption(f"Sampled ocean cell: {surface['latitude']:.2f}°N, {surface['longitude']:.2f}°E. Coordinates are snapped to the available ocean grid.")


def render_overview():
    with st.container(key="hero"):
        hero_copy, hero_status = st.columns([1.7, 0.7], gap="large", vertical_alignment="center")
        with hero_copy:
            st.badge("SUBSURFACE OCEAN EXPLORER", icon=":material/smart_toy:", color="green")
            st.title("Make the ocean below the surface visible.", icon=":material/travel_explore:")
            st.write("Explore the North Indian Ocean from the surface to 1,000 m. Choose a location, reconstruct its water column, and inspect the patterns below the surface.")
        with hero_status:
            with st.container(border=True, key="hero-status"):
                st.caption("CURRENT WATER COLUMN")
                st.metric("Selected depth", f"{depth} m", border=False)
                st.caption(f"{latitude:.2f}°N · {longitude:.2f}°E")
                st.caption(f"{analysis_date:%d %b %Y}")
        with st.container(horizontal=True, key="hero-meta"):
            st.badge("North Indian Ocean", icon=":material/location_on:", color="blue")
            st.badge("15 depth layers", icon=":material/layers:", color="green")
            st.badge("0.25° grid", icon=":material/grid_on:", color="orange")
        st.caption("OceanEmbed · SIH26066 · Satellite-to-subsurface reconstruction")

        with st.container(key="preset-pills"):
            st.caption("START WITH A REGIONAL SCENARIO")
            p_cols = st.columns(4, gap="small")
            for i, cp in enumerate(CYCLONE_EVENT_PRESETS):
                with p_cols[i % len(p_cols)]:
                    st.button(
                        cp["name"].split(" (")[0].replace("Ganga-Brahmaputra Plume", "River plume").replace("Southwest Upwelling Zone", "Upwelling"),
                        help=cp["name"],
                        key=f"btn_p_{cp['id']}",
                        width="stretch",
                        on_click=apply_preset,
                        args=(cp["id"],),
                    )


    st.subheader("Your selected water column")
    with st.container(horizontal=True, key="metrics"):
        st.metric("Surface temperature", f"{surface['sst']:.2f} °C", border=True)
        st.metric(f"Temperature at {depth} m", f"{depth_row['temperature_c']:.2f} °C", border=True)
        st.metric("Depth layers", str(len(profile)), border=True)
    st.subheader("Choose your next step")
    destinations = [
        ("Ocean Explorer", "Explore maps & transects", "Compare basin-wide fields with a vertical water column."),
        ("Profiles & Data", "Inspect & export a profile", "Review every depth, nearby observations, and CSV / NetCDF exports."),
        ("Mission Intelligence", "Open mission diagnostics", "Inspect heat potential and acoustic profile diagnostics."),
    ]
    for column, (destination, title, description) in zip(st.columns(3), destinations):
        with column, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)
            st.button("Open " + destination, key="open-" + destination, on_click=navigate_to, args=(destination,), width="stretch")
    with st.expander("Water-column interpretation", expanded=False):
        st.markdown(narrative)
    st.caption("Suggested showcase: Overview → Ocean Explorer → Profiles & Data → Mission Intelligence → Validation.")


def render_explore():
    view_mode = st.radio(
        "Exploration View",
        ["Horizontal Map (Depth Slice)", "Vertical Transect Curtain (0–900m)"],
        horizontal=True,
        key="explore_view_mode",
        label_visibility="collapsed",
    )
    if view_mode == "Horizontal Map (Depth Slice)":
        map_col, profile_col = st.columns([1.35, 1], gap="large")
        with map_col:
            with st.container(border=True, key="map-card"):
                st.subheader(f"{variable} field at {depth} m depth", icon=":material/map:")
                render_chart(
                    map_figure(grid, latitude, longitude, variable, colors),
                    width="stretch",
                    key="map_selection",
                    on_select=update_location_from_map_selection,
                    selection_mode="points",
                    config={"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d"]},
                )
                st.caption("Click any point on the map to relocate the target sounding coordinates.")
        with profile_col:
            with st.container(border=True, key="profile-card"):
                profile_variable = "Temperature" if variable == "Embedding" else variable
                st.subheader(f"Vertical {profile_variable.lower()} profile", icon=":material/show_chart:")
                if variable == "Embedding":
                    st.caption("Embedding is a shared spatial representation, not a depth-resolved variable. The sounding shows its reconstructed temperature column.")
                render_chart(profile_figure(profile, profile_variable, colors, comparison_profile), width="stretch", config={"displayModeBar": False})
    else:
        transect_col, t_meta_col = st.columns([1.5, 1], gap="large")
        with transect_col:
            with st.container(border=True, key="transect-card"):
                st.subheader(f"{variable} basin transect", icon=":material/view_column:")
                axis_choice = st.radio("Slice Plane", ["Zonal Transect (East-West along Latitude)", "Meridional Transect (North-South along Longitude)"], horizontal=True, key="transect_axis_choice")
                axis_key = "lat" if "Zonal" in axis_choice else "lon"
                coord_val = latitude if axis_key == "lat" else longitude
                t_data = transect(coord_val, axis=axis_key, variable=variable, analysis_date=analysis_date)
                render_chart(transect_figure(t_data, variable, colors), width="stretch", config={"displayModeBar": False})
                st.caption(f"2D Basin Cross-Section along {t_data['fixed_label']}. Reveals subsurface isotherm slopes, thermocline depth variations, and water mass boundaries down to 900m.")
        with t_meta_col:
            with st.container(border=True, key="transect-meta-card"):
                st.subheader("Transect Intelligence", icon=":material/analytics:")
                st.metric("Slice orientation", "East–West" if axis_key == "lat" else "North–South")
                st.metric("Fixed latitude" if axis_key == "lat" else "Fixed longitude", f"{t_data['fixed_value']:.2f}°")
                st.metric("Depth coverage", "0–1,000 m")
                st.info("Read the curtain from the surface downward. Changes in the contours show reconstructed vertical gradients across the selected basin slice.")

    with st.container(border=True, key="summary-card"):
        st.markdown(narrative)

    with st.expander(f"Inspect exact numeric values across all 15 depths (Depth {depth} m selected)", expanded=False):
        quick_table = profile[["depth_m", "temperature_c", "argo_temperature_c", "error_c", "salinity_psu", "density_kg_m3", "sound_speed_m_s", "confidence_pct", "uncertainty_c"]].copy()
        st.dataframe(
            themed_table(quick_table, is_dark),
            hide_index=True,
            width="stretch",
            column_config={
                "depth_m": st.column_config.NumberColumn("Depth", format="%d m"),
                "temperature_c": st.column_config.NumberColumn("V3 Temp (°C)", format="%.2f °C"),
                "argo_temperature_c": st.column_config.NumberColumn("GLORYS Actual (°C)", format="%.2f °C"),
                "error_c": st.column_config.NumberColumn("Residual Error", format="%.2f °C"),
                "salinity_psu": st.column_config.NumberColumn("Salinity (PSU)", format="%.3f"),
                "density_kg_m3": st.column_config.NumberColumn("Density (kg/m³)", format="%.2f"),
                "sound_speed_m_s": st.column_config.NumberColumn("Sound Speed (m/s)", format="%.1f"),
                "confidence_pct": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%d%%"),
                "uncertainty_c": st.column_config.NumberColumn("Test RMSE band", format="±%.2f °C"),
            },
        )

    export_col1, export_col2, note_col = st.columns([1, 1, 2])
    with export_col1:
        st.download_button(
            "Download grid slice (CSV)",
            grid.to_csv(index=False).encode("utf-8"),
            file_name=f"oceanembed_{variable.lower()}_{analysis_date.isoformat()}_{depth}m.csv",
            mime="text/csv",
            icon=":material/download:",
            key="grid-download",
            on_click="ignore",
        )
    with export_col2:
        st.download_button(
            "Download sounding (NetCDF .nc)",
            export_profile_to_netcdf(profile, mission_context),
            file_name=f"oceanembed_sounding_{analysis_date.isoformat()}.nc",
            mime="application/x-netcdf",
            icon=":material/download:",
            key="explore-netcdf-download",
            on_click="ignore",
        )
    with note_col:
        st.caption("Temperature and salinity use the loaded v3 checkpoint. Density and sound speed are derived; confidence is a heuristic, not a calibrated probability. The 900 m temperature reference is interpolated from 700/1000 m.")


def render_profile():
    st.subheader(f"Water column state at selected depth ({depth} m)", icon=":material/layers:")
    with st.container(key="depth-cards"):
        depth_metric_columns = st.columns(3)
        depth_metric_columns[0].metric(
            f"Temperature ({depth}m)",
            f"{depth_row['temperature_c']:.2f} °C",
            f"{depth_row['temperature_c'] - depth_row['argo_temperature_c']:+.2f} °C vs GLORYS",
            border=True,
        )
        depth_metric_columns[1].metric(
            f"GLORYS Ground Truth",
            f"{depth_row['argo_temperature_c']:.2f} °C",
            f"Residual: {depth_row['error_c']:.2f} °C",
            border=True,
        )
        depth_metric_columns[2].metric(
            f"Salinity ({depth}m)",
            f"{depth_row['salinity_psu']:.3f} PSU",
            f"Surface SSS: {surface['sss']:.1f}",
            border=True,
        )
        depth_metric_columns[0].metric(
            f"Seawater Density (ρ)",
            f"{depth_row['density_kg_m3']:.2f} kg/m³",
            f"σ_θ: {depth_row['density_kg_m3'] - 1000:.2f}",
            border=True,
        )
        depth_metric_columns[1].metric(
            f"Sound Speed (c)",
            f"{depth_row['sound_speed_m_s']:.1f} m/s",
            "Mackenzie eqn",
            border=True,
        )
        depth_metric_columns[2].metric(
            f"Model Confidence",
            f"{depth_row['confidence_pct']}%",
            f"Uncertainty: ±{depth_row['uncertainty_c']:.2f} °C",
            border=True,
        )

    with st.expander("Surface observations & forcing", expanded=False):
        st.subheader("Surface input fields", icon=":material/satellite_alt:")
        with st.container(horizontal=True, key="surface-cards"):
            st.metric("SST (Surface Temp)", f"{surface['sst']:.2f} °C", "Surface temperature input", border=True)
            st.metric("SSS (Surface Salinity)", f"{surface['sss']:.2f} PSU", "Surface salinity input", border=True)
            st.metric("SLA (Sea Level Anomaly)", f"{surface['sla']:+.3f} m", "Altimetry (SSH)", border=True)
            st.metric("Surface Wind Vector", f"{surface['wind_speed']:.1f} m/s", f"From {surface['wind_direction']}", border=True)
            st.metric("Surface Geostrophic Drift", f"{surface['current_speed']:.2f} m/s", f"Towards {surface['current_direction']}", border=True)

    if comparison_profile is not None and comparison_surface is not None:
        comp_row = comparison_profile.loc[comparison_profile["depth_m"] == depth].iloc[0]
        st.subheader(f"Dual-Point Comparison at {depth} m ({st.session_state.comparison_latitude:.2f}°N, {st.session_state.comparison_longitude:.2f}°E)", icon=":material/compare_arrows:")
        with st.container(horizontal=True, key="comparison-cards"):
            st.metric(
                f"Δ Temperature ({depth}m)",
                f"{depth_row['temperature_c'] - comp_row['temperature_c']:+.2f} °C",
                f"Compare: {comp_row['temperature_c']:.2f} °C",
                border=True,
            )
            st.metric(
                f"Δ Salinity ({depth}m)",
                f"{depth_row['salinity_psu'] - comp_row['salinity_psu']:+.3f} PSU",
                f"Compare: {comp_row['salinity_psu']:.3f} PSU",
                border=True,
            )
            st.metric(
                f"Δ Density",
                f"{depth_row['density_kg_m3'] - comp_row['density_kg_m3']:+.2f} kg/m³",
                f"Compare: {comp_row['density_kg_m3']:.2f}",
                border=True,
            )
            st.metric(
                f"Δ Sound Speed",
                f"{depth_row['sound_speed_m_s'] - comp_row['sound_speed_m_s']:+.1f} m/s",
                f"Compare: {comp_row['sound_speed_m_s']:.1f}",
                border=True,
            )


    table_col = st.container()
    cast_col = st.expander("Nearby reference casts", expanded=False)
    with table_col:
        with st.container(border=True, key="depth-table-card"):
            st.subheader("Comprehensive Vertical Hydrographic Sounding", icon=":material/table_chart:")
            leading_columns = ["depth_m", "temperature_c", "salinity_psu", "density_kg_m3",
                               "sound_speed_m_s", "argo_temperature_c", "error_c"]
            full_display = profile[leading_columns + [name for name in profile.columns if name not in leading_columns]].copy()
            st.dataframe(
                themed_table(full_display, is_dark),
                hide_index=True,
                height=450,
                column_config={
                    "depth_m": st.column_config.NumberColumn("Depth", format="%d m"),
                    "temperature_c": st.column_config.NumberColumn("V3 Temp (°C)", format="%.2f °C"),
                    "argo_temperature_c": st.column_config.NumberColumn("GLORYS Actual (°C)", format="%.2f °C"),
                    "error_c": st.column_config.NumberColumn("Residual", format="%.2f °C"),
                    "salinity_psu": st.column_config.NumberColumn("Salinity (PSU)", format="%.3f"),
                    "density_kg_m3": st.column_config.NumberColumn("Density (kg/m³)", format="%.2f"),
                    "sound_speed_m_s": st.column_config.NumberColumn("Sound Speed (m/s)", format="%.1f"),
                    "confidence_pct": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%d%%"),
                    "uncertainty_c": st.column_config.NumberColumn("Test RMSE band", format="±%.2f °C"),
                },
            )
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    "Download full profile sounding (CSV)",
                    profile.to_csv(index=False).encode("utf-8"),
                    file_name=f"oceanembed_profile_{analysis_date.isoformat()}.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    key="profile-download",
                    on_click="ignore",
                )
            with dl_col2:
                st.download_button(
                    "Download sounding (CF-1.8 NetCDF .nc)",
                    export_profile_to_netcdf(profile, mission_context),
                    file_name=f"oceanembed_profile_{analysis_date.isoformat()}.nc",
                    mime="application/x-netcdf",
                    icon=":material/download:",
                    key="profile-netcdf-download",
                    on_click="ignore",
                )
    with cast_col:
        with st.container(border=True, key="casts-card"):
            st.subheader("Regional reference casts", icon=":material/sensors:")
            st.caption("Illustrative nearby float records for the demonstration; not a live Argo feed.")
            st.dataframe(
                themed_table(observations, is_dark),
                hide_index=True,
                height=450,
                column_config={
                    "observed_on": st.column_config.DateColumn("Observed", format="DD MMM"),
                    "latitude": st.column_config.NumberColumn("Lat", format="%.2f°"),
                    "longitude": st.column_config.NumberColumn("Lon", format="%.2f°"),
                    "distance_km": st.column_config.NumberColumn("Distance", format="%.1f km"),
                    "max_depth_m": st.column_config.NumberColumn("Profile depth", format="%d m"),
                    "quality": st.column_config.TextColumn("QC Flag"),
                },
            )

    with st.expander("Multi-Variable Hydrographic Sounding (T-S-c Overlay)", expanded=False, icon=":material/stacked_line_chart:"):
        render_chart(multi_variable_figure(profile, colors), width="stretch", key="profile-multi-var-chart")
        st.caption("Synchronized vertical soundings of reconstructed Temperature (°C), Salinity (PSU), and Mackenzie (1981) Sound Speed (m/s) across the 0–900m column.")


def render_quality():

    with st.container(border=True):
        st.subheader("Serving v3 - recorded test evaluation")
        report_path = Path(__file__).resolve().parent / "results/oceanembed_v3_qc_full/metrics.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            col_t, col_s = st.columns(2)
            col_t.metric("V3 test temperature RMSE", f"{report['test']['temperature']['rmse']:.3f} °C")
            col_s.metric("V3 test salinity RMSE", f"{report['test']['salinity']['rmse']:.3f} PSU")
            st.caption("Loaded v3 checkpoint: five GLORYS days, held-out day 5, through 900 m. These recorded scores apply to that test split, not every displayed date or region, and are not independent satellite-to-Argo validation.")
            st.download_button("Download v3 evaluation (JSON)", report_path.read_bytes(), "oceanembed_v3_evaluation.json", "application/json", key="v3-evaluation-download", on_click="ignore")
        else:
            st.info("Train the v3 candidate to generate its evaluation report.")
    v3_report = engine_obj.report
    errors = pd.DataFrame({"Depth (m)": v3_report['depths_m'],
        "Temperature RMSE (C)": v3_report['test']['temperature']['rmse_by_depth'],
        "Salinity RMSE (PSU)": v3_report['test']['salinity']['rmse_by_depth']})
    st.subheader("V3 error by depth")
    for col, (label, unit) in zip(st.columns(2), [("Temperature RMSE (C)", "C"), ("Salinity RMSE (PSU)", "PSU")]):
        with col:
            figure = go.Figure(go.Bar(x=errors['Depth (m)'].astype(str), y=errors[label],
                                     name=label, marker_color=colors['model']))
            figure.update_layout(height=300, xaxis_title="Depth (m)", yaxis_title=label,
                                 paper_bgcolor=colors['paper'], plot_bgcolor=colors['panel'])
            render_chart(figure, width="stretch", config={"displayModeBar": False})
    st.dataframe(themed_table(errors, is_dark), hide_index=True, width="stretch")
    st.caption("Benchmark and model below use the same held-out day. Test RMSE bands are not calibrated confidence intervals.")
    st.dataframe(themed_table(validation, is_dark), hide_index=True, width="stretch")
    with st.expander("Historical v2 evaluation and explainability"):
        st.subheader("Historical v2 CNN evaluation")
        chart_col, details_col = st.columns([1.2, 1], gap="large")
        with chart_col:
            with st.container(border=True, key="validation-card"):
                st.subheader("Depth-wise model validation error", icon=":material/verified:")
                render_chart(validation_figure(depth_metrics, colors), width="stretch", config={"displayModeBar": False})
        with details_col:
            with st.container(border=True, key="quality-notes"):
                st.subheader("Per-depth performance benchmarks", icon=":material/analytics:")
                st.dataframe(
                    themed_table(depth_metrics, is_dark),
                    hide_index=True,
                    height=300,
                    column_config={
                        "depth_m": st.column_config.NumberColumn("Depth", format="%d m"),
                        "rmse_c": st.column_config.NumberColumn("RMSE (°C)", format="%.3f"),
                        "mae_c": st.column_config.NumberColumn("MAE (°C)", format="%.3f"),
                        "bias_c": st.column_config.NumberColumn("Bias (°C)", format="%+.3f"),
                        "r2_corr": st.column_config.NumberColumn("R² Corr", format="%.3f"),
                    },
                )

        st.subheader("Historical v2 evaluation artifacts", icon=":material/image:")
        img_col1, img_col2 = st.columns(2, gap="large")
        with img_col1:
            scatter_img = Path(__file__).resolve().parent / "results" / "validation_scatter_agreement.png"
            if scatter_img.exists():
                st.image(str(scatter_img), caption="Prediction vs Actual Agreement Scatter Plot", width="stretch")
            else:
                st.info("Validation scatter plot in results directory.")
        with img_col2:
            error_map_img = Path(__file__).resolve().parent / "results" / "validation_error_map.png"
            if error_map_img.exists():
                st.image(str(error_map_img), caption="Basin-wide Validation Error Distribution Map", width="stretch")
            else:
                st.info("Validation error map in results directory.")

        with st.expander("Feature importance & SHAP attribution analysis", icon=":material/psychology:"):
            shap_img = Path(__file__).resolve().parent / "results" / "shap_summary_v2.png"
            if shap_img.exists():
                st.image(str(shap_img), caption="SHAP Summary: Importance of Surface Satellite Channels on Subsurface Inversion", width="stretch")


def render_data():
    st.subheader("Model Architecture & Inversion Pipeline", icon=":material/memory:")
    st.markdown(
        """
        The **OceanEmbedNet** architecture is a deep convolutional neural network trained on multi-source satellite remote sensing data and GLORYS reanalysis physics.
        """
    )

    spec_col1, spec_col2 = st.columns(2, gap="large")
    with spec_col1:
        with st.container(border=True, key="data-card"):
            st.subheader("V3 model specifications", icon=":material/code:")
            st.markdown(
                """
                - **Network Backbone**: `OceanEmbedNet_v3` with three depth-wise residual SE blocks
                - **Inputs**: SST, SSS, SSH, current u/v, latitude, longitude, and seasonal sine/cosine
                - **Feature Pipeline**: 26 engineered features plus 26 missing-value indicators
                - **Wind**: Available as surface context; this checkpoint does not use wind features
                - **Latent Embedding**: 128-dimensional shared ocean representation
                - **Outputs**: Separate temperature and salinity heads, 15 depths (0-900m)
                - **Training**: Joint T/S loss with density stability and thermocline penalties
                - **Resolution**: 0.25° horizontal mesh (100 lat × 240 lon = 24,000 nodes)
                """
            )
    with spec_col2:
        with st.container(border=True, key="saved-card"):
            st.subheader("Saved locations", icon=":material/bookmarks:")
            saved_locations = list_saved_locations()
            if not saved_locations.empty:
                st.dataframe(themed_table(saved_locations, is_dark), hide_index=True)
                st.button("Clear saved locations", icon=":material/delete_sweep:", on_click=clear_locations)
            else:
                st.caption("Save points from the sidebar to persist them locally in the SQLite store.")


def render_mission():
    render_mission_intelligence(profile, colors, mission_context, comparison_profile, comparison_context)


page_renderers = {
    "Overview": render_overview,
    "Ocean Explorer": render_explore,
    "Profiles & Data": render_profile,
    "Mission Intelligence": render_mission,
    "Validation": render_quality,
    "About & Saved Places": render_data,
}
selected_page = st.session_state.workspace_page
if selected_page != "Overview":
    st.title(selected_page)
    st.caption(f"{latitude:.2f}°N · {longitude:.2f}°E  |  {analysis_date:%d %b %Y}  |  {depth} m  |  {variable}")
page_renderers[selected_page]()
if st.session_state.get("_last_rendered_page") != selected_page:
    st.session_state._last_rendered_page = selected_page
    st.html("""<script>
      requestAnimationFrame(() => {
        document.querySelector('[data-testid="stMain"]')?.scrollTo({top: 0, behavior: 'instant'});
        window.scrollTo(0, 0);
      });
    </script>""", unsafe_allow_javascript=True)
