"""NumPy-only heat-content and descriptive ocean-profile diagnostics.

TCHP definition: https://www.aoml.noaa.gov/phod/cyclone/method.php
Mackenzie coefficients and validity domain:
https://resource.npl.co.uk/acoustics/techguides/soundseawater/underlying-phys.html

The 50/80 kJ/cm2 bands are project heuristics, not cyclone-category forecasts.
No acoustic propagation, detection coverage or concealment is inferred here.
"""

from __future__ import annotations

import numpy as np

RHO_KG_M3 = 1025.0
CP_J_KG_K = 3993.0
HEAT_FACTOR = RHO_KG_M3 * CP_J_KG_K / 1e7  # J/m2 -> kJ/cm2


def _profile(depths, *values):
    """Require paired, finite 1D data; sort without modifying caller arrays."""
    arrays = [np.asarray(a, dtype=float) for a in (depths, *values)]
    z = arrays[0]
    if z.ndim != 1 or z.size < 2:
        raise ValueError("A profile requires at least two depth samples.")
    if any(a.ndim != 1 or a.size != z.size for a in arrays):
        raise ValueError("Profile arrays must be one-dimensional and equal length.")
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("Profile arrays must contain only finite values.")
    order = np.argsort(z)
    arrays = [a[order] for a in arrays]
    if arrays[0][0] < 0 or np.any(np.diff(arrays[0]) <= 0):
        raise ValueError("Depths must be nonnegative and unique.")
    return arrays


def compute_cyclone_heat_potential(depths: list[float], temps: list[float]) -> dict:
    """Integrate surface-connected excess heat using linear interpolation.

    The first sample must be at 0 m. Stop at the first T <= 26 C, even if
    deeper inversions are warmer. Surface T <= 26 gives D26=0 and zero heat.
    If no crossing is sampled, d26_m is None and heat is a lower bound.
    Invalid input raises ValueError rather than silently bridging data gaps.
    """
    z, t = _profile(depths, temps)
    if z[0] != 0:
        raise ValueError("TCHP requires a surface sample at depth 0 m.")
    lower_bound = False
    if t[0] <= 26:
        d26, integral, status = 0.0, 0.0, "surface_at_or_below_26"
        integrated_to = 0.0
    else:
        crossings = np.flatnonzero(t <= 26)
        if crossings.size:
            i = int(crossings[0])
            d26 = float(z[i - 1] + (26 - t[i - 1]) *
                        (z[i] - z[i - 1]) / (t[i] - t[i - 1]))
            zi = np.append(z[:i], d26)
            excess = np.append(t[:i] - 26, 0.0)
            status = "resolved"
            integrated_to = d26
        else:
            d26, zi, excess = None, z, t - 26
            lower_bound, status = True, "below_profile"
            integrated_to = float(z[-1])
        # Explicit trapezoids support both NumPy 1.x and 2.x.
        integral = float(np.sum(np.diff(zi) * (excess[:-1] + excess[1:]) * 0.5))
    heat = integral * HEAT_FACTOR
    band = "high" if heat > 80 else "moderate" if heat >= 50 else "low"
    # A truncated low/moderate integral cannot determine the full-column band.
    risk = "undetermined" if lower_bound and band != "high" else band
    return {
        "d26_m": d26,
        "d26_status": status,
        "integrated_to_m": integrated_to,
        "tchp_kj_cm2": heat,
        "ohc_kj_cm2": heat,
        "is_lower_bound": lower_bound,
        "risk_level": risk,
        "risk_color": {"high": "red", "moderate": "orange", "low": "green",
                       "undetermined": "gray"}[risk],
        "risk_label": {"high": "High heat potential", "moderate": "Moderate heat potential",
                       "low": "Low heat potential", "undetermined": "Full-column band unresolved"}[risk],
        "classification_basis": "Project heuristic: low <50, moderate 50–80, high >80 kJ/cm²; not an RI or category forecast.",
    }


def mackenzie_sound_speed(temp_c, sal_psu, depth_m):
    """Nine-term Mackenzie (1981), m/s; broadcasts scalars/NumPy arrays.

    Published domain: T 2–30 C, S 25–40 ppt, D 0–8000 m. Values outside
    that domain are extrapolations; they are not silently clipped. The
    existing application supplies practical salinity as its approximation.
    """
    t, s, d = np.broadcast_arrays(
        np.asarray(temp_c, dtype=float), np.asarray(sal_psu, dtype=float),
        np.asarray(depth_m, dtype=float),
    )
    c = (1448.96 + 4.591*t - 5.304e-2*t**2 + 2.374e-4*t**3
         + 1.340*(s-35) + 1.630e-2*d + 1.675e-7*d**2
         - 1.025e-2*t*(s-35) - 7.139e-13*t*d**3)
    return float(c) if c.ndim == 0 else c


def compute_acoustic_profile_diagnostics(
    depths: list[float], temps: list[float], sound_speeds: list[float],
) -> dict:
    """Descriptive speed range and strongest sampled cooling interval.

    An interval containing the steepest cooling is not a thermocline base.
    These statistics do not establish ray paths or acoustic coverage.
    """
    z, t, c = _profile(depths, temps, sound_speeds)
    if np.any(c <= 0):
        raise ValueError("Sound speeds must be positive.")
    cooling = -np.diff(t) / np.diff(z)
    i = int(np.argmax(cooling))
    interval = [float(z[i]), float(z[i + 1])] if cooling[i] > 0 else None
    return {
        "minimum_sound_speed_m_s": float(c.min()),
        "maximum_sound_speed_m_s": float(c.max()),
        "strongest_cooling_interval_m": interval,
        "maximum_cooling_gradient_c_m": max(0.0, float(cooling[i])),
        "sample_count": int(z.size),
        "profile_depth_range_m": [float(z[0]), float(z[-1])],
    }


def compute_physics_integrity(
    depths: list[float], temps: list[float], salinities: list[float], densities: list[float],
) -> dict:
    """Evaluate hydrostatic stability (N^2), Mixed Layer Depth (MLD), and Barrier Layer Thickness (BLT).

    MLD criterion: de Boyer Montegut et al. (2004) delta_sigma = 0.03 kg/m3 from 10m depth.
    ILD criterion: delta_T = 0.2 C drop from 10m depth.
    Barrier Layer Thickness: BLT = max(0.0, ILD - MLD).
    """
    z, t, s, rho = _profile(depths, temps, salinities, densities)
    drho = np.diff(rho)
    dz = np.diff(z)
    drho_dz = drho / dz
    g = 9.81
    rho0 = RHO_KG_M3
    n2 = (g / rho0) * drho_dz

    # Flag negative density gradients exceeding numerical noise threshold
    inversion_idx = np.where(drho_dz < -1e-4)[0]
    inversion_count = int(len(inversion_idx))
    inversion_layers = [[float(z[i]), float(z[i + 1])] for i in inversion_idx]

    # Reference depth index closest to 10m
    ref_idx = int(np.argmin(np.abs(z - 10.0)))
    ref_rho = rho[ref_idx]
    ref_t = t[ref_idx]

    # MLD (delta_sigma = 0.03 kg/m3 increase from 10m)
    target_rho = ref_rho + 0.03
    mld = None
    crossing_rho = np.where((z >= z[ref_idx]) & (rho >= target_rho))[0]
    if len(crossing_rho) > 0:
        i = int(crossing_rho[0])
        if i == ref_idx or rho[i] == rho[i - 1]:
            mld = float(z[i])
        else:
            mld = float(z[i - 1] + (target_rho - rho[i - 1]) * (z[i] - z[i - 1]) / (rho[i] - rho[i - 1]))

    # ILD (delta_T = 0.2 C drop from 10m)
    target_t = ref_t - 0.2
    ild = None
    crossing_t = np.where((z >= z[ref_idx]) & (t <= target_t))[0]
    if len(crossing_t) > 0:
        i = int(crossing_t[0])
        if i == ref_idx or t[i] == t[i - 1]:
            ild = float(z[i])
        else:
            ild = float(z[i - 1] + (target_t - t[i - 1]) * (z[i] - z[i - 1]) / (t[i] - t[i - 1]))

    blt = 0.0
    if mld is not None and ild is not None:
        blt = max(0.0, float(ild - mld))

    status = "100% Hydrostatically Stable" if inversion_count == 0 else f"{inversion_count} Inversion Layer(s) Detected"
    return {
        "stability_status": status,
        "is_stable": inversion_count == 0,
        "inversion_count": inversion_count,
        "inversion_layers": inversion_layers,
        "min_n2_s2": float(np.min(n2)),
        "mld_m": round(mld, 1) if mld is not None else None,
        "ild_m": round(ild, 1) if ild is not None else None,
        "blt_m": round(blt, 1),
        "has_barrier_layer": blt >= 5.0,
        "stability_badge_color": "green" if inversion_count == 0 else "orange",
    }


def export_profile_to_netcdf(profile_df, metadata: dict | None = None) -> bytes:
    """Generate CF-1.8 compliant NetCDF binary bytes for a vertical hydrographic sounding."""
    import os
    import tempfile
    from scipy.io import netcdf_file

    meta = metadata or {}
    ordered = profile_df.sort_values("depth_m").copy()
    # UI context stores the selected ocean cell under sampled_location.
    location = meta.get("sampled_location") or meta

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with netcdf_file(tmp_path, "w") as f:
            title = meta.get("title", "OceanEmbed 3D Hydrographic Ocean Sounding")
            f.title = title.encode("utf-8") if isinstance(title, str) else title
            f.institution = b"SIH26066 - MoES / INCOIS Team"
            f.source = b"OceanEmbedNet v2 Multi-Task CNN"
            f.Conventions = b"CF-1.8"
            f.history = b"Generated by OceanEmbed platform"
            f.references = b"Mackenzie (1981), UNESCO (1980) EOS-80"
            if "date" in meta:
                f.analysis_date = str(meta["date"]).encode("utf-8")
            if "latitude" in location:
                f.latitude = float(location["latitude"])
            if "longitude" in location:
                f.longitude = float(location["longitude"])

            n_depths = len(ordered)
            f.createDimension("depth", n_depths)

            var_specs = [
                ("depth_m", "depth", "depth", "m", "down"),
                ("temperature_c", "temperature", "sea_water_temperature", "degree_Celsius", None),
                ("salinity_psu", "salinity", "sea_water_salinity", "psu", None),
                ("density_kg_m3", "density", "sea_water_potential_density", "kg m-3", None),
                ("sound_speed_m_s", "sound_speed", "speed_of_sound_in_sea_water", "m s-1", None),
            ]
            if "confidence_pct" in ordered.columns:
                var_specs.append(("confidence_pct", "confidence", "reconstruction_confidence", "percent", None))
            if "uncertainty_c" in ordered.columns:
                var_specs.append(("uncertainty_c", "uncertainty", "temperature_uncertainty", "degree_Celsius", None))

            for col, vname, std_name, units, positive in var_specs:
                if col in ordered.columns:
                    v = f.createVariable(vname, "f", ("depth",))
                    v[:] = ordered[col].to_numpy(dtype="float32")
                    v.units = units.encode("ascii")
                    v.standard_name = std_name.encode("ascii")
                    if positive:
                        v.positive = positive.encode("ascii")
                        v.axis = b"Z"
        with open(tmp_path, "rb") as f_read:
            return f_read.read()
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def compute_marine_heatwave_metrics(
    sst_c: float,
    climatology_sst_c: float = 28.2,
    baseline_threshold_c: float = 29.2,
) -> dict:
    """Classify Marine Heatwave (MHW) according to Hobday et al. (2016) and evaluate thermal stress.

    Hobday et al. (2016) categorisation:
    - delta_t = sst - climatology
    - threshold = baseline_threshold - climatology (e.g. 90th percentile difference, ~1.0 C)
    - If delta_t < threshold: No MHW (Normal or Mild anomaly)
    - Category I (Moderate): 1x to 2x threshold
    - Category II (Strong): 2x to 3x threshold
    - Category III (Severe): 3x to 4x threshold
    - Category IV (Extreme): >= 4x threshold

    Thermal stress for coral reef bleaching (Lakshadweep / Gulf of Mannar / Andaman Sea):
    - Bleaching Watch: SST >= 29.0 C
    - Bleaching Alert Level 1: SST >= 29.5 C (Significant bleaching expected)
    - Bleaching Alert Level 2: SST >= 30.5 C (Severe widespread bleaching & mortality)
    """
    sst = float(sst_c)
    clim = float(climatology_sst_c)
    thresh = float(baseline_threshold_c)
    delta_thresh = max(0.5, thresh - clim)
    delta_t = sst - clim

    if delta_t < 0:
        category = "No Heatwave"
        category_level = 0
        cat_name = "Normal (Below Climatology)"
        badge_color = "green"
        intensity_multiple = 0.0
    elif delta_t < delta_thresh:
        category = "Sub-threshold Warm Anomaly"
        category_level = 0
        cat_name = "Mild Anomaly"
        badge_color = "blue"
        intensity_multiple = round(delta_t / delta_thresh, 2)
    else:
        multiple = delta_t / delta_thresh
        intensity_multiple = round(multiple, 2)
        if multiple < 2.0:
            category = "Category I"
            category_level = 1
            cat_name = "Category I (Moderate MHW)"
            badge_color = "orange"
        elif multiple < 3.0:
            category = "Category II"
            category_level = 2
            cat_name = "Category II (Strong MHW)"
            badge_color = "orange"
        elif multiple < 4.0:
            category = "Category III"
            category_level = 3
            cat_name = "Category III (Severe MHW)"
            badge_color = "red"
        else:
            category = "Category IV"
            category_level = 4
            cat_name = "Category IV (Extreme MHW)"
            badge_color = "red"

    # Coral reef bleaching alert status
    if sst >= 30.5:
        bleaching_status = "Alert Level 2 (Severe Mortality Risk)"
        bleaching_color = "red"
    elif sst >= 29.5:
        bleaching_status = "Alert Level 1 (Bleaching Likely)"
        bleaching_color = "orange"
    elif sst >= 29.0:
        bleaching_status = "Bleaching Watch (Thermal Stress Accumulating)"
        bleaching_color = "yellow"
    else:
        bleaching_status = "No Thermal Stress"
        bleaching_color = "green"

    return {
        "sst_c": round(sst, 2),
        "climatology_sst_c": round(clim, 2),
        "sst_anomaly_c": round(delta_t, 2),
        "category": category,
        "category_level": category_level,
        "category_label": cat_name,
        "badge_color": badge_color,
        "intensity_multiple": intensity_multiple,
        "bleaching_status": bleaching_status,
        "bleaching_color": bleaching_color,
        "is_heatwave": category_level >= 1,
    }


def compute_naval_sonar_tactics(
    depths: list[float], sound_speeds: list[float], temps: list[float] | None = None,
) -> dict:
    """Tactical Oceanography & Naval Sonar Decision Aid metrics (Urick 1983).

    Calculates:
    - Sonic Layer Depth (SLD): Depth of the near-surface maximum sound speed (in top 150m).
    - Surface Duct Trapping Cutoff Frequency: f_cutoff = 1420 / sqrt(SLD^3) in Hz (for SLD in m).
    - In-Layer Gradient: (c[SLD] - c[0]) / SLD (positive indicates upward refraction / surface trapping).
    - Below-Layer Gradient: Strongest negative gradient immediately beneath SLD leading to shadow zone.
    - SOFAR Channel Axis (Sound Fixing and Ranging): Depth of absolute minimum sound speed in water column.
    """
    if temps is not None:
        z, c, _ = _profile(depths, sound_speeds, temps)
    else:
        z, c = _profile(depths, sound_speeds)

    if np.any(c <= 0):
        raise ValueError("Sound speeds must be positive.")

    # SLD: Sound speed maximum within the upper 150m
    upper_mask = z <= 150.0
    upper_idx = np.where(upper_mask)[0]
    if len(upper_idx) > 0:
        max_upper_idx = int(upper_idx[np.argmax(c[upper_idx])])
        sld_m = float(z[max_upper_idx])
        c_sld = float(c[max_upper_idx])
    else:
        sld_m = 0.0
        c_sld = float(c[0])

    # Trapping cutoff frequency: f_cutoff = 1420 / sqrt(SLD^3) (Hz)
    # Acoustic waves with frequency > f_cutoff will be trapped within the surface duct.
    f_cutoff_hz = None
    if sld_m >= 5.0:
        f_cutoff_hz = round(1420.0 / np.sqrt(sld_m ** 3), 1)

    # In-layer gradient
    in_layer_grad = 0.0
    if sld_m > 0:
        in_layer_grad = float((c_sld - c[0]) / sld_m)

    # Below-layer gradient & shadow zone:
    # Sound rays passing below SLD bend downward rapidly into the cold thermocline.
    below_sld_idx = np.where(z > sld_m)[0]
    below_grad = None
    if len(below_sld_idx) > 0:
        first_below = int(below_sld_idx[0])
        dz = z[first_below] - sld_m
        if dz > 0:
            below_grad = float((c[first_below] - c_sld) / dz)

    # SOFAR channel axis: depth of minimum sound speed across entire profile
    min_c_idx = int(np.argmin(c))
    sofar_axis_m = float(z[min_c_idx])
    min_sound_speed_m_s = float(c[min_c_idx])

    # Shadow zone extent approximation (from SLD to where speed matches c_sld again or deep channel)
    shadow_zone_active = bool(sld_m >= 10.0 and below_grad is not None and below_grad < -0.05)

    return {
        "sld_m": round(sld_m, 1),
        "surface_sound_speed_m_s": round(float(c[0]), 1),
        "sld_sound_speed_m_s": round(c_sld, 1),
        "f_cutoff_hz": f_cutoff_hz,
        "has_surface_duct": bool(sld_m >= 15.0 and in_layer_grad >= 0.01),
        "in_layer_gradient_s_1": round(in_layer_grad, 4),
        "below_layer_gradient_s_1": round(below_grad, 4) if below_grad is not None else None,
        "sofar_axis_m": round(sofar_axis_m, 1),
        "min_sound_speed_m_s": round(min_sound_speed_m_s, 1),
        "shadow_zone_active": shadow_zone_active,
    }


def generate_mission_briefing(context: dict, metrics: dict, surface: dict | None = None) -> str:
    """Generate executive operational briefing in Markdown format for MoES/INCOIS and Navy commanders."""
    date_str = context.get("date", "N/A")
    loc = context.get("sampled_location", context.get("requested_location", {}))
    lat = loc.get("latitude", 0.0)
    lon = loc.get("longitude", 0.0)
    basin = "Arabian Sea" if lon < 77.0 else "Bay of Bengal" if lat >= 5.0 else "Equatorial Indian Ocean"

    cyclone = metrics.get("cyclone") or {}
    acoustics = metrics.get("acoustics") or {}
    physics = metrics.get("physics") or {}
    mhw = metrics.get("mhw") or {}
    tactics = metrics.get("tactics") or {}

    tchp = cyclone.get("tchp_kj_cm2", 0.0)
    d26 = cyclone.get("d26_m")
    risk_label = cyclone.get("risk_label", "N/A")

    mld = physics.get("mld_m", "N/A")
    blt = physics.get("blt_m", 0.0)
    stability = physics.get("stability_status", "N/A")

    sld = tactics.get("sld_m", "N/A")
    f_cut = tactics.get("f_cutoff_hz")
    f_cut_str = f"{f_cut} Hz" if f_cut else "N/A (Layer too shallow)"
    sofar = tactics.get("sofar_axis_m", "N/A")

    mhw_cat = mhw.get("category_label", "Normal")
    bleach = mhw.get("bleaching_status", "Normal")

    lines = [
        "# EXECUTIVE OPERATIONAL MISSION BRIEFING",
        "**Platform**: OceanEmbed 3D Subsurface Operational Suite (SIH26066)",
        f"**Operational Theater**: {basin} ({lat:.2f}°N, {lon:.2f}°E)",
        f"**Assessment Date**: {date_str} | **Model Source**: OceanEmbedNet v2 (Multi-Task CNN)",
        "",
        "---",
        "",
        "## 1. METEOROLOGY & CYCLONIC THREAT STATUS",
        f"- **Tropical Cyclone Heat Potential (TCHP)**: {tchp:.2f} kJ/cm² ({risk_label})",
    ]
    if d26 is not None:
        lines.append(f"- **26°C Isotherm Depth (D26)**: {d26:.1f} m")
    else:
        lines.append("- **26°C Isotherm Depth (D26)**: Unresolved (> column depth)")

    lines.extend([
        f"- **Marine Heatwave (MHW) Category**: {mhw_cat} (Hobday et al., 2016)",
        f"- **Coral Bleaching Alert**: {bleach}",
        f"- **Operational Impact**: {'HIGH - Deep thermal reservoir supports rapid cyclonic intensification (RI). Alert coastal disaster management.' if tchp > 80 else 'MODERATE - Heat potential sufficient for sustained convective activity.' if tchp >= 50 else 'LOW - Limited ocean heat energy available for tropical cyclone intensification.'}",
        "",
        "## 2. NAVAL TACTICAL OCEANOGRAPHY & SONAR ENVIRONMENT",
        f"- **Sonic Layer Depth (SLD)**: {sld} m",
        f"- **Surface Duct Trapping Cutoff Frequency**: {f_cut_str}",
        f"- **Deep Sound Channel (SOFAR) Axis**: {sofar} m (Sound speed min: {tactics.get('min_sound_speed_m_s', 'N/A')} m/s)",
        f"- **Acoustic Shadow Zone (Blind Cone)**: {'ACTIVE immediately beneath ' + str(sld) + 'm layer. Submarines below SLD shielded from surface active sonar.' if tactics.get('shadow_zone_active') else 'WEAK / INACTIVE. Linear or mild sound speed gradient.'}",
        "- **Tactical Sonar Recommendation**: Use variable-depth sonar (VDS) or dipping sonar lowered below SLD to counter shadow zone concealment.",
        "",
        "## 3. PHYSICAL STRATIFICATION & HYDROSTATIC STABILITY",
        f"- **Mixed Layer Depth (MLD)**: {mld} m (de Boyer Montégut Δσ=0.03)",
        f"- **Barrier Layer Thickness (BLT)**: {blt:.1f} m" + (" (Active fresh salinity barrier layer inhibits vertical cooling mixing)" if blt >= 5.0 else " (Normal unstratified barrier layer)"),
        f"- **Hydrostatic Stability (N²)**: {stability}",
        "",
        "---",
        "*Confidential - Formatted for Ministry of Earth Sciences (MoES), INCOIS & Naval Headquarters.*",
    ])
    return "\n".join(lines)


def compute_mission_metrics(
    depths, temps, sound_speeds, salinities=None, densities=None, sst=None,
) -> dict:
    """Keep missing/invalid derived data from breaking an existing profile.

    Each module fails independently. None is JSON-safe and never represents
    a zero-heat observation. Direct calculation functions remain strict.
    """
    result = {"cyclone": None, "acoustics": None, "physics": None, "mhw": None, "tactics": None, "errors": {}}
    modules = [
        ("cyclone", compute_cyclone_heat_potential, (depths, temps)),
        ("acoustics", compute_acoustic_profile_diagnostics, (depths, temps, sound_speeds)),
        ("tactics", compute_naval_sonar_tactics, (depths, sound_speeds, temps)),
    ]
    if salinities is not None and densities is not None:
        modules.append(
            ("physics", compute_physics_integrity, (depths, temps, salinities, densities))
        )

    # Compute MHW if surface temperature is available
    surface_temp = sst
    if surface_temp is None:
        try:
            z_arr, t_arr = _profile(depths, temps)
            surface_temp = float(t_arr[0])
        except Exception:
            surface_temp = None

    if surface_temp is not None:
        modules.append(
            ("mhw", compute_marine_heatwave_metrics, (surface_temp,))
        )

    for key, fn, args in modules:
        try:
            result[key] = fn(*args)
        except (ValueError, TypeError) as exc:
            result["errors"][key] = str(exc)
    return result


def compute_acoustic_ray_paths(
    depths: list[float] | np.ndarray,
    sound_speeds: list[float] | np.ndarray,
    source_depth_m: float = 15.0,
    max_range_km: float = 25.0,
    launch_angles_deg: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Simulate Snell's Law acoustic ray refraction through stratified sound velocity profile."""
    z = np.asarray(depths, dtype=float)
    c = np.asarray(sound_speeds, dtype=float)
    if launch_angles_deg is None:
        launch_angles_deg = [-8.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0]

    c_source = float(np.interp(source_depth_m, z, c))
    r_step = 250.0  # 250 m step
    num_steps = int((max_range_km * 1000.0) / r_step)

    ray_paths = []
    for theta0_deg in launch_angles_deg:
        theta0 = np.radians(theta0_deg)
        p = np.cos(theta0) / c_source

        curr_r = 0.0
        curr_z = source_depth_m
        curr_theta = theta0

        rx = [0.0]
        rz = [source_depth_m]

        for _ in range(num_steps):
            c_curr = float(np.interp(curr_z, z, c))
            val = p * c_curr
            if val >= 1.0 or val <= -1.0:
                curr_theta = -curr_theta
            else:
                curr_theta = np.sign(curr_theta) * np.arccos(np.clip(val, -1.0, 1.0))

            dz = r_step * np.tan(curr_theta)
            curr_z += dz
            curr_r += r_step / 1000.0

            if curr_z < 0.0:
                curr_z = -curr_z
                curr_theta = -curr_theta
            if curr_z > 1000.0:
                curr_z = 2000.0 - curr_z
                curr_theta = -curr_theta

            rx.append(round(curr_r, 2))
            rz.append(round(curr_z, 1))

        ray_paths.append({
            "launch_angle_deg": theta0_deg,
            "range_km": rx,
            "depth_m": rz,
        })
    return ray_paths
