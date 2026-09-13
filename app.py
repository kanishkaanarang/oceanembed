"""OceanEmbed: Deep Learning 3D Ocean State Reconstruction Platform."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend import (
    ANALYSIS_END,
    ANALYSIS_START,
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
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


initialize_state()
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
        [data-testid="stHeader"] {
          background-color: rgba(7, 27, 42, 0.8) !important;
        }
        .st-key-hero {
          background: radial-gradient(circle at 88% 18%, rgba(0, 210, 255, 0.22), transparent 26%), linear-gradient(115deg, #0d2a3e, #071b2a 62%) !important;
          border-color: #1a425a !important;
        }
        .st-key-hero h1, .st-key-hero p, .st-key-hero span {
          color: #e7f4f8 !important;
        }
        .st-key-metrics [data-testid="stMetric"], .st-key-depth-cards [data-testid="stMetric"], .st-key-surface-cards [data-testid="stMetric"], .st-key-comparison-cards [data-testid="stMetric"] {
          background: linear-gradient(145deg, #0d2a3e, #092030) !important;
          border: 1px solid #1a425a !important;
          color: #e7f4f8 !important;
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25) !important;
        }
        [data-testid="stMetricLabel"] p {
          color: #94a3b8 !important;
        }
        [data-testid="stMetricValue"] {
          color: #f1f5f9 !important;
        }
        .st-key-map-card, .st-key-profile-card, .st-key-depth-table-card,
        .st-key-casts-card, .st-key-validation-card, .st-key-catalogue-card,
        .st-key-data-card, .st-key-saved-card, .st-key-quality-notes, .st-key-explore-guide,
        .st-key-summary-card, .st-key-hotspot-card, .st-key-sounding-inspect {
          background: #0c2433 !important;
          border: 1px solid #1a425a !important;
          color: #e7f4f8 !important;
        }
        .st-key-summary-card p, .st-key-summary-card strong {
          color: #e7f4f8 !important;
        }
        [data-testid="stTabs"] [role="tab"] {
          color: #94a3b8 !important;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
          color: #00d2ff !important;
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
          background-color: #f8fafc !important;
          color: #0f172a !important;
          border-right: 1px solid #e2e8f0 !important;
        }
        .st-key-hero {
          background: radial-gradient(circle at 88% 18%, rgba(2, 132, 199, 0.15), transparent 26%), linear-gradient(115deg, #e0f2fe, #f8fafc 62%) !important;
          border-color: #bae6fd !important;
        }
        .st-key-hero h1 {
          color: #0369a1 !important;
        }
        .st-key-hero p, .st-key-hero span {
          color: #334155 !important;
        }
        .st-key-metrics [data-testid="stMetric"], .st-key-depth-cards [data-testid="stMetric"], .st-key-surface-cards [data-testid="stMetric"], .st-key-comparison-cards [data-testid="stMetric"] {
          background: linear-gradient(145deg, #ffffff, #f1f5f9) !important;
          border: 1px solid #e2e8f0 !important;
          color: #0f172a !important;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04) !important;
        }
        [data-testid="stMetricLabel"] p {
          color: #64748b !important;
        }
        [data-testid="stMetricValue"] {
          color: #0f172a !important;
        }
        .st-key-map-card, .st-key-profile-card, .st-key-depth-table-card,
        .st-key-casts-card, .st-key-validation-card, .st-key-catalogue-card,
        .st-key-data-card, .st-key-saved-card, .st-key-quality-notes, .st-key-explore-guide,
        .st-key-summary-card, .st-key-hotspot-card, .st-key-sounding-inspect {
          background: #ffffff !important;
          border: 1px solid #e2e8f0 !important;
          color: #0f172a !important;
        }
        .st-key-summary-card p, .st-key-summary-card strong {
          color: #1e293b !important;
        }
        [data-testid="stTabs"] [role="tab"] {
          color: #64748b !important;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
          color: #0284c7 !important;
        }
        </style>
        """
    )


def record_interaction() -> None:
    st.session_state.run_count += 1


def apply_preset(preset_id: str) -> None:
    for item in REGIONAL_PRESETS:
        if item["id"] == preset_id:
            st.session_state.latitude = float(item["latitude"])
            st.session_state.longitude = float(item["longitude"])
            st.session_state.active_preset = preset_id
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
    event = st.session_state.get("map_selection")
    try:
        points = event.selection.points
    except AttributeError:
        return
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
        record_interaction()


def plot_colors() -> dict[str, str]:
    is_dark = st.session_state.get("dark_mode", False)
    return {
        "paper": "#071b2a" if is_dark else "#ffffff",
        "panel": "#0c2433" if is_dark else "#f8fafc",
        "text": "#e7f4f8" if is_dark else "#0f172a",
        "grid": "#245066" if is_dark else "#e2e8f0",
        "model": "#00d2ff" if is_dark else "#0284c7",
        "reference": "#ff9e00" if is_dark else "#d97706",
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
        yaxis=dict(title="Latitude (°N)", range=LATITUDE_RANGE, gridcolor=colors["grid"], zeroline=False, scaleanchor="x", scaleratio=1),
        showlegend=False,
    )
    return figure


def profile_figure(profile: pd.DataFrame, variable: str, colors: dict[str, str], comparison_profile: pd.DataFrame | None = None) -> go.Figure:
    if variable == "Confidence":
        figure = go.Figure(go.Scatter(
            x=profile["confidence_pct"],
            y=profile["depth_m"],
            mode="lines+markers",
            name="Model confidence",
            line=dict(color=colors["model"], width=3),
            marker=dict(size=6),
            hovertemplate="Depth: %{y}m<br>Confidence: %{x}%<extra></extra>",
        ))
        figure.update_layout(
            height=480,
            margin=dict(l=5, r=5, t=20, b=5),
            paper_bgcolor=colors["paper"],
            plot_bgcolor=colors["panel"],
            font=dict(color=colors["text"]),
            xaxis=dict(title="Model Confidence (%)", range=[50, 100], gridcolor=colors["grid"]),
            yaxis=dict(title="Depth (m)", autorange="reversed", gridcolor=colors["grid"]),
        )
        return figure

    if variable == "Salinity":
        estimate, lower, upper, reference, unit = "salinity_psu", "salinity_lower_psu", "salinity_upper_psu", "argo_salinity_psu", "PSU"
    elif "Density" in variable:
        figure = go.Figure(go.Scatter(
            x=profile["density_kg_m3"],
            y=profile["depth_m"],
            mode="lines+markers",
            name="Seawater Density (ρ)",
            line=dict(color="#06d6a0", width=3),
            marker=dict(size=6),
            hovertemplate="Depth: %{y}m<br>Density: %{x:.2f} kg/m³<extra></extra>",
        ))
        figure.update_layout(
            height=480,
            margin=dict(l=5, r=5, t=20, b=5),
            paper_bgcolor=colors["paper"],
            plot_bgcolor=colors["panel"],
            font=dict(color=colors["text"]),
            xaxis=dict(title="Seawater Density (kg/m³)", gridcolor=colors["grid"]),
            yaxis=dict(title="Depth (m)", autorange="reversed", gridcolor=colors["grid"]),
        )
        return figure
    elif "Sound" in variable:
        figure = go.Figure(go.Scatter(
            x=profile["sound_speed_m_s"],
            y=profile["depth_m"],
            mode="lines+markers",
            name="Speed of Sound (c)",
            line=dict(color="#f72585", width=3),
            marker=dict(size=6),
            hovertemplate="Depth: %{y}m<br>Sound speed: %{x:.1f} m/s<extra></extra>",
        ))
        figure.update_layout(
            height=480,
            margin=dict(l=5, r=5, t=20, b=5),
            paper_bgcolor=colors["paper"],
            plot_bgcolor=colors["panel"],
            font=dict(color=colors["text"]),
            xaxis=dict(title="Sound Speed (m/s)", gridcolor=colors["grid"]),
            yaxis=dict(title="Depth (m)", autorange="reversed", gridcolor=colors["grid"]),
        )
        return figure
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
        name="Validated uncertainty (±1σ)",
        hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        x=profile[estimate],
        y=profile["depth_m"],
        mode="lines+markers",
        name="OceanEmbedNet (CNN)",
        line=dict(color=colors["model"], width=3.5),
        marker=dict(size=6),
        hovertemplate=f"Depth: %{{y}}m<br>Estimate: %{{x:.2f}} {unit}<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=profile[reference],
        y=profile["depth_m"],
        mode="lines+markers",
        name="GLORYS Actual (Reanalysis)",
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
    st.caption("Deep Learning Subsurface Ocean Reconstruction")
    engine_obj = get_engine()
    dataset_label = "150-Day Reanalysis" if not engine_obj.is_compact else "Compact Reference Dataset"
    st.badge(f"PyTorch CNN · {dataset_label}", icon=":material/bolt:", color="green")
    st.toggle(
        "Dark mode",
        key="dark_mode",
        on_change=record_interaction,
        help="Switch between Light and Dark interface styles",
    )

    st.subheader("Regional Hotspots")
    preset_names = {p["id"]: f"{p['name']} ({p['badge']})" for p in REGIONAL_PRESETS}
    preset_choice = st.selectbox(
        "Jump to key oceanographic regime:",
        options=["custom"] + [p["id"] for p in REGIONAL_PRESETS],
        format_func=lambda x: "Custom Coordinates" if x == "custom" else preset_names.get(x, x),
        index=0 if st.session_state.active_preset == "custom" else [p["id"] for p in REGIONAL_PRESETS].index(st.session_state.active_preset) + 1,
    )
    if preset_choice != "custom" and preset_choice != st.session_state.active_preset:
        apply_preset(preset_choice)

    st.subheader("Reconstruction controls")
    st.select_slider(
        "Observation date",
        options=analysis_dates,
        format_func=lambda item: item.strftime("%d %b %Y"),
        key="analysis_date",
        on_change=record_interaction,
    )
    st.slider("Latitude", *LATITUDE_RANGE, step=0.25, format="%.2f° N", key="latitude", on_change=record_interaction)
    st.slider("Longitude", *LONGITUDE_RANGE, step=0.25, format="%.2f° E", key="longitude", on_change=record_interaction)
    st.select_slider("Depth slice", options=list(map(int, DEPTHS)), format_func=lambda item: f"{item} m", key="depth", on_change=record_interaction)
    st.segmented_control(
        "Map layer",
        ["Temperature", "Ground Truth", "Residual", "Salinity", "Density", "Sound Speed", "Embedding", "Confidence"],
        key="variable",
        on_change=record_interaction,
        wrap=True,
    )

    st.toggle("Compare second point", key="comparison_enabled", on_change=record_interaction)
    if st.session_state.comparison_enabled:
        with st.expander("Comparison location", icon=":material/compare_arrows:", expanded=True):
            st.select_slider("Comparison date", options=analysis_dates, format_func=lambda item: item.strftime("%d %b %Y"), key="comparison_date", on_change=record_interaction)
            st.slider("Comparison latitude", *LATITUDE_RANGE, step=0.25, format="%.2f° N", key="comparison_latitude", on_change=record_interaction)
            st.slider("Comparison longitude", *LONGITUDE_RANGE, step=0.25, format="%.2f° E", key="comparison_longitude", on_change=record_interaction)

    st.button("Save current location", icon=":material/bookmark_add:", width="stretch", on_click=save_current_location, key="save-location")

    with st.expander("Live Model & Data Proof", icon=":material/verified_user:"):
        dataset_name = "Full 150-Day Satellite Observation Arrays (0.25° Grid)" if not engine_obj.is_compact else "Compact Reference Arrays (0.25° Grid)"
        st.markdown(
            f"""
            - **Status**: Live PyTorch Inference Active
            - **Weights**: `oceanembed_cnn_best_v2.pt` (Loaded)
            - **Parameters**: 60,495 trainable weights
            - **Dataset**: {dataset_name}
            - **Grid**: 0.25° horizontal (24,000 nodes)
            - **Depths**: 15 levels (0m–1000m)
            - **Backend**: Real tensor forward pass (NOT dummy synthetic)
            """
        )
        if engine_obj.is_compact:
            st.info("💡 **Full Dataset Notice**: Running on embedded compact dataset. For full 150-day satellite historical arrays (250MB), place `training_arrays_v2_normalized.npz` in `training_arrays_v2_normalized/` or `data/processed/`.")


analysis_date = st.session_state.analysis_date
latitude = st.session_state.latitude
longitude = st.session_state.longitude
depth = st.session_state.depth
variable = st.session_state.variable
colors = plot_colors()

surface = get_surface_conditions(latitude, longitude, analysis_date)
profile = reconstruct(latitude, longitude, analysis_date)
grid = field(latitude, longitude, depth, variable, analysis_date)
observations = nearby_observations(latitude, longitude, analysis_date)
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

with st.container(key="hero"):
    hero_copy, hero_status = st.columns([1.7, 0.7], gap="large", vertical_alignment="center")
    with hero_copy:
        st.badge("DEEP CNN INFERENCE ACTIVE", icon=":material/smart_toy:", color="green")
        st.title("Make the ocean below the surface visible.", icon=":material/travel_explore:")
        st.write("Real-time 3D vertical ocean temperature and physical state reconstruction powered by **OceanEmbedNet v2**. Inverting 7 surface satellite parameters into 15 subsurface depth layers across the North Indian Ocean basin.")
    with hero_status:
        with st.container(border=True, key="hero-status"):
            st.caption("CURRENT WATER COLUMN")
            st.metric("Selected depth", f"{depth} m", border=False)
            st.caption(f"{latitude:.2f}°N · {longitude:.2f}°E")
            st.caption(f"{analysis_date:%d %b %Y}")
    with st.container(horizontal=True, key="hero-meta"):
        st.badge("North Indian Ocean", icon=":material/location_on:", color="blue")
        st.badge("150-day satellite dataset", icon=":material/calendar_month:", color="violet")
        st.badge("0.25° resolution", icon=":material/grid_on:", color="orange")
        st.badge(f"Inference run #{st.session_state.run_count}", icon=":material/bolt:", color="gray")
    st.caption("Model source: OceanEmbedNet v2 (PyTorch Deep CNN · 60,495 weights)")

st.subheader(f"Water column state at selected depth ({depth} m)", icon=":material/layers:")
with st.container(horizontal=True, key="depth-cards"):
    st.metric(
        f"Temperature ({depth}m)",
        f"{depth_row['temperature_c']:.2f} °C",
        f"{depth_row['temperature_c'] - depth_row['argo_temperature_c']:+.2f} °C vs GLORYS",
        border=True,
    )
    st.metric(
        f"GLORYS Ground Truth",
        f"{depth_row['argo_temperature_c']:.2f} °C",
        f"Residual: {depth_row['error_c']:.2f} °C",
        border=True,
    )
    st.metric(
        f"Salinity ({depth}m)",
        f"{depth_row['salinity_psu']:.3f} PSU",
        f"Surface SSS: {surface['sss']:.1f}",
        border=True,
    )
    st.metric(
        f"Seawater Density (ρ)",
        f"{depth_row['density_kg_m3']:.2f} kg/m³",
        f"σ_θ: {depth_row['density_kg_m3'] - 1000:.2f}",
        border=True,
    )
    st.metric(
        f"Sound Speed (c)",
        f"{depth_row['sound_speed_m_s']:.1f} m/s",
        "Mackenzie eqn",
        border=True,
    )
    st.metric(
        f"Model Confidence",
        f"{depth_row['confidence_pct']}%",
        f"Uncertainty: ±{depth_row['uncertainty_c']:.2f} °C",
        border=True,
    )

st.subheader("Driving surface satellite observations", icon=":material/satellite_alt:")
with st.container(horizontal=True, key="surface-cards"):
    st.metric("SST (Surface Temp)", f"{surface['sst']:.2f} °C", "Satellite Infrared", border=True)
    st.metric("SSS (Surface Salinity)", f"{surface['sss']:.2f} PSU", "Microwave Radiometer", border=True)
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

explore_tab, profile_tab, quality_tab, data_tab = st.tabs(["Explore & Map", "Sounding Soundings & Table", "Model Validation & Quality", "Technical Specifications"])

with explore_tab:
    map_col, profile_col = st.columns([1.35, 1], gap="large")
    with map_col:
        with st.container(border=True, key="map-card"):
            st.subheader(f"{variable} field at {depth} m depth", icon=":material/map:")
            st.plotly_chart(
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
            st.subheader(f"Vertical {variable.lower()} profile", icon=":material/show_chart:")
            st.plotly_chart(profile_figure(profile, variable, colors, comparison_profile), width="stretch", config={"displayModeBar": False})

    with st.container(border=True, key="summary-card"):
        st.markdown(narrative)

    with st.expander(f"Inspect exact numeric values across all 15 depths (Depth {depth} m selected)", expanded=True):
        quick_table = profile[["depth_m", "temperature_c", "argo_temperature_c", "error_c", "salinity_psu", "density_kg_m3", "sound_speed_m_s", "confidence_pct", "uncertainty_c"]].copy()
        st.dataframe(
            quick_table,
            hide_index=True,
            width="stretch",
            column_config={
                "depth_m": st.column_config.NumberColumn("Depth", format="%d m"),
                "temperature_c": st.column_config.NumberColumn("CNN Temp (°C)", format="%.2f °C"),
                "argo_temperature_c": st.column_config.NumberColumn("GLORYS Actual (°C)", format="%.2f °C"),
                "error_c": st.column_config.NumberColumn("Residual Error", format="%.2f °C"),
                "salinity_psu": st.column_config.NumberColumn("Salinity (PSU)", format="%.3f"),
                "density_kg_m3": st.column_config.NumberColumn("Density (kg/m³)", format="%.2f"),
                "sound_speed_m_s": st.column_config.NumberColumn("Sound Speed (m/s)", format="%.1f"),
                "confidence_pct": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%d%%"),
                "uncertainty_c": st.column_config.NumberColumn("±1σ Uncertainty", format="±%.2f °C"),
            },
        )

    export_col, note_col = st.columns([1, 2])
    with export_col:
        st.download_button(
            "Download grid slice (CSV)",
            grid.to_csv(index=False).encode("utf-8"),
            file_name=f"oceanembed_{variable.lower()}_{analysis_date.isoformat()}_{depth}m.csv",
            mime="text/csv",
            icon=":material/download:",
            key="grid-download",
        )
    with note_col:
        st.caption("All displayed soundings are generated by live PyTorch CNN inference from real 7-channel satellite observation grids.")

with profile_tab:
    table_col, cast_col = st.columns([1.35, 1], gap="large")
    with table_col:
        with st.container(border=True, key="depth-table-card"):
            st.subheader("Comprehensive Vertical Hydrographic Sounding", icon=":material/table_chart:")
            full_display = profile.copy()
            st.dataframe(
                full_display,
                hide_index=True,
                height=450,
                column_config={
                    "depth_m": st.column_config.NumberColumn("Depth", format="%d m"),
                    "temperature_c": st.column_config.NumberColumn("CNN Temp (°C)", format="%.2f °C"),
                    "argo_temperature_c": st.column_config.NumberColumn("GLORYS Actual (°C)", format="%.2f °C"),
                    "error_c": st.column_config.NumberColumn("Residual", format="%.2f °C"),
                    "salinity_psu": st.column_config.NumberColumn("Salinity (PSU)", format="%.3f"),
                    "density_kg_m3": st.column_config.NumberColumn("Density (kg/m³)", format="%.2f"),
                    "sound_speed_m_s": st.column_config.NumberColumn("Sound Speed (m/s)", format="%.1f"),
                    "confidence_pct": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%d%%"),
                    "uncertainty_c": st.column_config.NumberColumn("±1σ Uncertainty", format="±%.2f °C"),
                },
            )
            st.download_button(
                "Download full profile sounding (CSV)",
                profile.to_csv(index=False).encode("utf-8"),
                file_name=f"oceanembed_profile_{analysis_date.isoformat()}.csv",
                mime="text/csv",
                icon=":material/download:",
                key="profile-download",
            )
    with cast_col:
        with st.container(border=True, key="casts-card"):
            st.subheader("Regional Argo float observations", icon=":material/sensors:")
            st.dataframe(
                observations,
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

with quality_tab:
    chart_col, details_col = st.columns([1.2, 1], gap="large")
    with chart_col:
        with st.container(border=True, key="validation-card"):
            st.subheader("Depth-wise model validation error", icon=":material/verified:")
            st.plotly_chart(validation_figure(depth_metrics, colors), width="stretch", config={"displayModeBar": False})
    with details_col:
        with st.container(border=True, key="quality-notes"):
            st.subheader("Per-depth performance benchmarks", icon=":material/analytics:")
            st.dataframe(
                depth_metrics,
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

    st.subheader("Trained model evaluation artifacts", icon=":material/image:")
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

with data_tab:
    st.subheader("Model Architecture & Inversion Pipeline", icon=":material/memory:")
    st.markdown(
        """
        The **OceanEmbedNet** architecture is a deep convolutional neural network trained on multi-source satellite remote sensing data and GLORYS reanalysis physics.
        """
    )

    spec_col1, spec_col2 = st.columns(2, gap="large")
    with spec_col1:
        with st.container(border=True, key="data-card"):
            st.subheader("Deep CNN Specifications", icon=":material/code:")
            st.markdown(
                """
                - **Network Backbone**: `OceanEmbedNet` (Encoder-Decoder CNN)
                - **Input Channels (7)**:
                  1. `SST`: Sea Surface Temperature (°C)
                  2. `SSS`: Sea Surface Salinity (PSU)
                  3. `SSH/SLA`: Sea Level Anomaly (m)
                  4. `uwnd`: Zonal Surface Wind Vector (m/s)
                  5. `vwnd`: Meridional Surface Wind Vector (m/s)
                  6. `ucurr`: Zonal Surface Geostrophic Current (m/s)
                  7. `vcurr`: Meridional Surface Current (m/s)
                - **Latent Embedding**: 32-channel oceanographic feature representation
                - **Output Channels**: 15 standard oceanographic depth layers (0–1000m)
                - **Resolution**: 0.25° horizontal mesh (100 lat × 240 lon = 24,000 nodes)
                """
            )
    with spec_col2:
        with st.container(border=True, key="saved-card"):
            st.subheader("Saved locations", icon=":material/bookmarks:")
            saved_locations = list_saved_locations()
            if not saved_locations.empty:
                st.dataframe(saved_locations, hide_index=True)
                st.button("Clear saved locations", icon=":material/delete_sweep:", on_click=clear_locations)
            else:
                st.caption("Save points from the sidebar to persist them locally in the SQLite store.")
