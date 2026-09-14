"""Mission cards, annotated scientific charts, and portable analysis exports."""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.ui.theme import render_chart

from src.models.mission_metrics import (
    compute_acoustic_ray_paths,
    compute_mission_metrics,
    export_profile_to_netcdf,
    generate_mission_briefing,
)


def profile_metrics(profile):
    salinities = profile.salinity_psu.to_numpy() if "salinity_psu" in profile.columns else None
    densities = profile.density_kg_m3.to_numpy() if "density_kg_m3" in profile.columns else None
    return compute_mission_metrics(
        profile.depth_m, profile.temperature_c, profile.sound_speed_m_s,
        salinities=salinities, densities=densities,
    )


def chart_layout(figure, colors, xlabel, height=340):
    figure.update_layout(
        height=height, margin=dict(l=10, r=16, t=35, b=10),
        paper_bgcolor=colors["paper"], plot_bgcolor=colors["panel"],
        font=dict(color=colors["text"], size=12),
        legend=dict(orientation="h", y=1.15, x=0),
        xaxis=dict(title=xlabel, gridcolor=colors["grid"], zeroline=False),
        yaxis=dict(title="Depth (m)", autorange="reversed", gridcolor=colors["grid"]),
    )
    return figure


def heat_content_figure(profile, heat, colors, comparison=None):
    """Shade precisely the same surface-connected integral used by the metric."""
    ordered = profile.sort_values("depth_m")
    z, t = ordered.depth_m.to_numpy(), ordered.temperature_c.to_numpy()
    limit = heat["integrated_to_m"]
    warm_z = np.append(z[z < limit], limit)
    warm_t = np.interp(warm_z, z, t)
    figure = go.Figure()
    if limit > 0:
        figure.add_trace(go.Scatter(
            x=np.r_[np.full(len(warm_z), 26.0), warm_t[::-1]],
            y=np.r_[warm_z, warm_z[::-1]], fill="toself", mode="lines",
            line=dict(width=0), fillcolor="rgba(231,163,84,.26)",
            name="Integrated warm layer", hoverinfo="skip",
        ))
    view_bottom = min(float(z[-1]), max(150.0, limit * 1.25))
    show_z = np.append(z[z < view_bottom], view_bottom)
    figure.add_trace(go.Scatter(
        x=np.interp(show_z, z, t), y=show_z, mode="lines+markers",
        name="Reconstructed temperature", line=dict(color=colors["model"], width=3),
        marker=dict(size=5), hovertemplate="%{x:.2f} °C · %{y:.1f} m<extra></extra>",
    ))
    if comparison is not None:
        other = comparison.sort_values("depth_m")
        figure.add_trace(go.Scatter(
            x=np.interp(show_z, other.depth_m, other.temperature_c), y=show_z,
            mode="lines", name="Comparison", line=dict(color="#a78bfa", dash="dot", width=2),
        ))
    figure.add_vline(x=26, line_dash="dash", line_color="#d99a4c")
    if heat["d26_m"] is not None and heat["d26_m"] > 0:
        figure.add_hline(y=heat["d26_m"], line_dash="dot", line_color=colors["text"],
                         annotation_text=f"D26 · {heat['d26_m']:.1f} m", annotation_position="bottom right")
    return chart_layout(figure, colors, "Temperature (°C)")


def sound_profile_figure(profile, acoustic, colors, comparison=None):
    figure = go.Figure()
    for frame, name, color, dash in [(profile, "Sound speed", colors["model"], "solid"),
                                      (comparison, "Comparison", "#a78bfa", "dot")]:
        if frame is not None:
            ordered = frame.sort_values("depth_m")
            figure.add_trace(go.Scatter(
                x=ordered.sound_speed_m_s, y=ordered.depth_m, mode="lines+markers", name=name,
                line=dict(color=color, width=3, dash=dash), marker=dict(size=5),
                hovertemplate="%{x:.1f} m/s · %{y:g} m<extra></extra>",
            ))
    interval = acoustic["strongest_cooling_interval_m"]
    if interval:
        figure.add_hrect(y0=interval[0], y1=interval[1], fillcolor="#e7a354", opacity=.18,
                         line_width=0, annotation_text="Strongest cooling interval", layer="below")
    return chart_layout(figure, colors, "Sound speed (m/s)", height=420)


def acoustic_ray_figure(profile, colors, source_depth=15.0):
    """Render Snell's Law acoustic ray refraction and surface ducting."""
    ordered = profile.sort_values("depth_m")
    depths = ordered.depth_m.to_numpy()
    speeds = ordered.sound_speed_m_s.to_numpy()
    rays = compute_acoustic_ray_paths(depths, speeds, source_depth_m=source_depth, max_range_km=25.0)

    figure = go.Figure()
    ray_colors = ["#38bdf8", "#0ea5e9", "#0284c7", "#e7a354", "#f97316", "#ef4444", "#a855f7"]
    for i, ray in enumerate(rays):
        c = ray_colors[i % len(ray_colors)]
        figure.add_trace(go.Scatter(
            x=ray["range_km"],
            y=ray["depth_m"],
            mode="lines",
            name=f"{ray['launch_angle_deg']:+g}°",
            line=dict(color=c, width=2),
            hovertemplate="Range: %{x:.1f} km<br>Depth: %{y:.1f} m<extra></extra>",
        ))

    figure.update_layout(
        height=380,
        margin=dict(l=10, r=16, t=35, b=10),
        paper_bgcolor=colors["paper"],
        plot_bgcolor=colors["panel"],
        font=dict(color=colors["text"], size=12),
        legend=dict(orientation="h", y=1.15, x=0, title=dict(text="Launch Angle")),
        xaxis=dict(title="Horizontal Sonar Range (km)", range=[0, 25], gridcolor=colors["grid"], zeroline=False),
        yaxis=dict(title="Depth (m)", autorange="reversed", range=[500, 0], gridcolor=colors["grid"]),
    )
    return figure


def multi_variable_figure(profile: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    """Multi-Variable Hydrographic Sounding (T-S-c Overlay) showing thermocline, halocline, and acoustic waveguide."""
    ordered = profile.sort_values("depth_m")
    z = ordered["depth_m"]
    t = ordered["temperature_c"]
    s = ordered["salinity_psu"]
    c = ordered["sound_speed_m_s"]

    temperature_color = colors.get("temperature", "#c2410c")
    salinity_color = colors.get("salinity", "#047857")
    acoustic_color = colors.get("acoustic", "#0369a1")
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=t, y=z, mode="lines+markers", name="Temperature (°C)",
        line=dict(color=temperature_color, width=2.5),
        marker=dict(size=4),
        hovertemplate="Depth: %{y}m<br>Temp: %{x:.2f} °C<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=s, y=z, mode="lines+markers", name="Salinity (PSU)",
        line=dict(color=salinity_color, width=2.5, dash="dash"),
        marker=dict(size=4),
        xaxis="x2",
        hovertemplate="Depth: %{y}m<br>Salinity: %{x:.3f} PSU<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=c, y=z, mode="lines+markers", name="Sound Speed (m/s)",
        line=dict(color=acoustic_color, width=2.5, dash="dot"),
        marker=dict(size=4),
        xaxis="x3",
        hovertemplate="Depth: %{y}m<br>Sound Speed: %{x:.1f} m/s<extra></extra>",
    ))

    figure.update_layout(
        height=480,
        margin=dict(l=10, r=16, t=65, b=10),
        paper_bgcolor=colors["paper"],
        plot_bgcolor=colors["panel"],
        font=dict(color=colors["text"], size=11),
        legend=dict(orientation="h", y=1.22, x=0),
        yaxis=dict(title="Depth (m)", autorange="reversed", domain=[0.16, 0.94], gridcolor=colors["grid"]),
        xaxis=dict(
            title=dict(text="Temperature (°C)", font=dict(color=temperature_color)),
            tickfont=dict(color=temperature_color),
            gridcolor=colors["grid"],
            zeroline=False,
            side="bottom",
        ),
        xaxis2=dict(
            title=dict(text="Salinity (PSU)", font=dict(color=salinity_color)),
            tickfont=dict(color=salinity_color),
            overlaying="x",
            side="top",
            showgrid=False,
        ),
        xaxis3=dict(
            title=dict(text="Sound Speed (m/s)", font=dict(color=acoustic_color)),
            tickfont=dict(color=acoustic_color),
            overlaying="x",
            side="bottom",
            anchor="free",
            position=0,
            showgrid=False,
        ),
    )
    return figure


def analysis_export(profile, metrics, context, comparison=None, comparison_context=None):
    """Create strict JSON with source context, units, and optional comparison."""
    report = {
        "schema_version": "1.0", "context": context,
        "units": {"depth": "m", "temperature": "°C", "heat_content": "kJ/cm²", "sound_speed": "m/s"},
        "mission_intelligence": metrics,
        "profile": json.loads(profile.to_json(orient="records")),
    }
    if comparison is not None:
        report["comparison"] = {"context": comparison_context, "mission_intelligence": profile_metrics(comparison)}
    return json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)


def render_mission_intelligence(profile, colors, context, comparison=None, comparison_context=None):
    metrics = profile_metrics(profile)
    heat, acoustic, physics = metrics["cyclone"], metrics["acoustics"], metrics.get("physics")
    mhw, tactics = metrics.get("mhw"), metrics.get("tactics")
    st.caption("METEOROLOGY, ACOUSTICS & OCEAN STRATIFICATION  /  Selected water column")
    heat_col, acoustic_col = st.columns(2, gap="medium")
    physics_col = st.container()
    with heat_col, st.container(border=True, key="mission-cyclone-card"):
        st.caption("01 / CYCLONE ENERGY & THERMAL STRESS")
        st.subheader("Heat beneath the surface")
        if heat is None:
            st.info(f"Heat potential unavailable: {metrics['errors']['cyclone']}")
        else:
            with st.container(key=f"mission-risk-{heat['risk_level']}"):
                st.badge(heat["risk_label"], color=heat["risk_color"])
            values = st.columns([1.3, 1])
            prefix = "≥ " if heat["is_lower_bound"] else ""
            delta = None
            if comparison is not None:
                other_heat = profile_metrics(comparison)["cyclone"]
                if other_heat and not heat["is_lower_bound"] and not other_heat["is_lower_bound"]:
                    delta = f"{heat['tchp_kj_cm2'] - other_heat['tchp_kj_cm2']:+.2f} vs comparison"
            values[0].metric("TCHP / OHC above 26°C", f"{prefix}{heat['tchp_kj_cm2']:.2f} kJ/cm²", delta, delta_color="off")
            d26 = f"> {heat['integrated_to_m']:.1f} m" if heat["d26_m"] is None else f"{heat['d26_m']:.1f} m"
            values[1].metric("26°C isotherm depth (D26)", d26)
            if mhw:
                st.badge(f"Marine heatwave: {mhw['category_label']}", color=mhw['badge_color'])
                st.caption(f"Coral Bleaching: {mhw['bleaching_status']} (SST {mhw['sst_c']}°C, Anomaly {mhw['sst_anomaly_c']:+.2f}°C)")
            st.html('<style>.oe-scale{display:flex;flex-wrap:wrap;gap:.75rem;font-size:.85rem}.oe-scale span{padding:.15rem .4rem;border-left:3px solid}</style><div class="oe-scale"><span style="border-color:#41b59a">LOW · &lt;50</span>'
                    '<span style="border-color:#e7a354">MODERATE · 50–80</span>'
                    '<span style="border-color:#e87575">HIGH · &gt;80 kJ/cm²</span></div>')
            if heat["is_lower_bound"]:
                st.caption("Profile ends above 26°C: heat is a lower bound and D26 is unresolved.")
            st.caption("Project heat-potential bands. Atmospheric conditions also affect intensification; this is not a cyclone category forecast.")
    with acoustic_col, st.container(border=True, key="mission-acoustic-card"):
        st.caption("02 / OCEAN ACOUSTICS & TACTICAL SONAR")
        st.subheader("Read the water column")
        if acoustic is None:
            st.info(f"Profile diagnostics unavailable: {metrics['errors']['acoustics']}")
        else:
            st.badge(f"{acoustic['sample_count']} depth samples", color="blue")
            values = st.columns([1.3, 1])
            values[0].metric("Sampled sound-speed range",
                             f"{acoustic['minimum_sound_speed_m_s']:.1f}–{acoustic['maximum_sound_speed_m_s']:.1f}")
            interval = acoustic["strongest_cooling_interval_m"]
            values[1].metric("Strongest sampled cooling interval",
                             f"{interval[0]:g}–{interval[1]:g} m" if interval else "None")
            if tactics:
                tac_cols = st.columns([1.3, 1])
                tac_cols[0].metric("Sonic Layer Depth (SLD)", f"{tactics['sld_m']:.1f} m", help="Depth of maximum sound speed in upper 150m")
                f_cut = f"{tactics['f_cutoff_hz']:.0f} Hz" if tactics['f_cutoff_hz'] else "N/A"
                tac_cols[1].metric("Duct Cutoff (fc)", f_cut, help="Surface duct trapping cutoff frequency (1420 / sqrt(SLD³))")
                if tactics["shadow_zone_active"]:
                    st.badge("Acoustic Shadow Zone Active Beneath SLD", color="orange")
            st.caption(f"Sound speed in m/s · Maximum sampled cooling: {acoustic['maximum_cooling_gradient_c_m']:.3f} °C/m")
            st.caption("Mackenzie (1981), nine terms. Salinity comes from the existing T–S estimate. "
                       "The steepest sampled interval does not establish the full thermocline extent.")
            t, s, z = (profile[name].to_numpy() for name in ("temperature_c", "salinity_psu", "depth_m"))
            if np.any((t < 2) | (t > 30) | (s < 25) | (s > 40) | (z > 8000)):
                st.caption("Some speeds are extrapolations outside Mackenzie's 2–30°C, 25–40 salinity, 0–8000 m domain.")
    with physics_col, st.container(border=True, key="mission-physics-card"):
        st.caption("03 / STRATIFICATION & PHYSICS")
        st.subheader("Physical consistency")
        if physics is None:
            st.info(f"Physics verification unavailable: {metrics['errors'].get('physics', 'No data')}")
        else:
            st.badge(physics["stability_status"], color=physics["stability_badge_color"])
            values = st.columns([1.3, 1])
            mld_str = f"{physics['mld_m']:.1f} m" if physics["mld_m"] is not None else "N/A"
            values[0].metric("Mixed Layer Depth (MLD)", mld_str, help="de Boyer Montégut (2004): Δσ = 0.03 kg/m³")
            blt_str = f"{physics['blt_m']:.1f} m"
            delta_blt = "Barrier layer active" if physics["has_barrier_layer"] else "Normal stratification"
            values[1].metric("Barrier Layer (BLT)", blt_str, delta_blt, delta_color="off", help="BLT = max(0, ILD - MLD)")
            st.caption(f"Isothermal Layer (ILD): {physics['ild_m'] or 'N/A'} m · Hydrostatic N² min: {physics['min_n2_s2']:.2e} s⁻²")
            st.caption("UNESCO EOS-80 density. In the Northern Bay of Bengal, fresh river runoff forms barrier layers that trap upper ocean heat.")
    with st.expander("Explore the heat reservoir", expanded=False, icon=":material/waves:"):
        if heat:
            render_chart(heat_content_figure(profile, heat, colors, comparison), width="stretch", key="mission-heat-chart")
            st.caption("The amber area is temperature excess above 26°C, integrated from the surface to the first crossing. "
                       "The dashed vertical line marks 26°C.")
        else:
            st.info("A complete surface-connected temperature profile is needed for this chart.")
    if acoustic and st.checkbox("Show sound velocity profile preview", key="mission-svp-preview"):
        render_chart(sound_profile_figure(profile, acoustic, colors, comparison), width="stretch", key="mission-svp-chart")
        st.caption("The shaded band marks the strongest sampled cooling interval. This chart shows the profile, not simulated ray paths.")
    if acoustic and st.checkbox("Show Snell's Law acoustic ray refraction preview", key="mission-ray-preview"):
        render_chart(acoustic_ray_figure(profile, colors), width="stretch", key="mission-ray-chart")
        st.caption("Snell's Law ray paths for a surface sonar transducer (15m depth). Downward-refracting rays illustrate the acoustic shadow zone blind cone beneath the thermocline.")
    with st.expander("Methodology & export", icon=":material/science:"):
        st.latex(r"\mathrm{TCHP}=\frac{1025\times3993}{10^7}\int_0^{D_{26}}[T(z)-26]\,dz\quad[\mathrm{kJ/cm^2}]")
        st.write("Linear interpolation resolves the first 26°C crossing. Trapezoidal integration uses the irregular depth spacing. "
                 "The reported OHC is the heat content above 26°C, and all metrics inherit the source reconstruction's limitations.")
        st.caption(f"Source: {context['dataset_mode']} · {context['date']} · Model: {context['model']}")
        export_btn1, export_btn2, export_btn3 = st.columns(3)
        with export_btn1:
            st.download_button("Download mission analysis (JSON)",
                               analysis_export(profile, metrics, context, comparison, comparison_context),
                               file_name=f"oceanembed_mission_{context['date']}.json", mime="application/json",
                               key="mission-download", icon=":material/download:", on_click="ignore")
        with export_btn2:
            st.download_button("Download sounding (CF-1.8 NetCDF .nc)",
                               export_profile_to_netcdf(profile, context),
                               file_name=f"oceanembed_sounding_{context['date']}.nc", mime="application/x-netcdf",
                               key="mission-netcdf-download", icon=":material/download:", on_click="ignore")
        with export_btn3:
            st.download_button("Download executive briefing (.md)",
                               generate_mission_briefing(context, metrics),
                               file_name=f"oceanembed_executive_briefing_{context['date']}.md", mime="text/markdown",
                               key="mission-briefing-download", icon=":material/description:", on_click="ignore")
