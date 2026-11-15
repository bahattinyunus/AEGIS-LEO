"""
LEO Trapped Radiation Environment, South Atlantic Anomaly (SAA) & Dose Engine
==============================================================================

First-order, physics-structured engineering model of the ionising radiation
environment seen by a LEO spacecraft:

  1. Eccentric (offset) tilted geomagnetic dipole: field magnitude |B|,
     McIlwain L-shell and geomagnetic latitude along the orbit.
  2. Trapped inner-belt protons (>30 MeV) and outer-belt electrons (>1 MeV)
     with an atmospheric loss-cone cutoff. Because the dipole is displaced
     ~550 km towards the western Pacific, drift shells dip lowest over the
     South Atlantic, which reproduces the South Atlantic Anomaly.
  3. Stormer vertical geomagnetic cutoff rigidity (R_c = 14.9 / L^2 GV) and
     storm-time polar-cap access of Solar Energetic Particles (SEP).
  4. Orbit-averaged fluence, Total Ionising Dose (TID) behind aluminium
     shielding and proton-induced Single Event Upset (SEU) rate, used to size
     the TMR memory scrubbing interval of the onboard Edge-AI computer.

The belt intensity profiles are parametric fits (order-of-magnitude agreement
with AP-8/AE-8 at LEO). They are intended for onboard situational awareness
and trade studies, not as a replacement for AP-9/AE-9 or SPENVIS analyses.
Known limitations: the dose only integrates the >30 MeV proton and >1 MeV
electron channels (a lower bound on the true TID), and the pure dipole
geometry produces a weak northern conjugate lobe that the real SAA lacks.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

from aegis_leo.constants import EARTH_MU, EARTH_RADIUS_KM, EARTH_ROTATION_RATE_RAD_S

# --- Geomagnetic eccentric dipole (IGRF epoch ~2020) ---
DIPOLE_B0_NT: float = 29_900.0                 # Equatorial surface field of the dipole term (nT)
GEOMAG_NORTH_POLE_LAT_DEG: float = 80.65
GEOMAG_NORTH_POLE_LON_DEG: float = -72.68
DIPOLE_CENTER_OFFSET_KM = np.array([-401.9, 300.3, 212.5])  # ECEF offset of the eccentric dipole centre
ATMOSPHERIC_LOSS_ALTITUDE_KM: float = 100.0

# --- Parametric belt profiles ---
PROTON_PEAK_FLUX_CM2_S: float = 3.0e4          # >30 MeV equatorial omnidirectional peak
PROTON_PEAK_L: float = 1.5
PROTON_INNER_L_WIDTH: float = 0.10       # Steep atmospheric inner edge of the inner belt
PROTON_OUTER_L_WIDTH: float = 0.28
PROTON_PITCH_EXPONENT: float = 4.0
ELECTRON_PEAK_FLUX_CM2_S: float = 1.0e6        # >1 MeV outer-belt equatorial peak (quiet)
ELECTRON_PEAK_L: float = 4.5
ELECTRON_L_WIDTH: float = 1.0
ELECTRON_PITCH_EXPONENT: float = 1.5
ELECTRON_DRIFT_LOSS_CONE_FLOOR: float = 2.0e-4  # Quasi-trapped population outside the stable region
ELECTRON_STORM_LOG10_PER_KP: float = 0.15

# --- Dose & SEU conversion ---
RAD_PER_MEV_PER_G: float = 1.602e-8
PROTON_STOPPING_POWER_MEV_CM2_G: float = 10.0
ELECTRON_STOPPING_POWER_MEV_CM2_G: float = 1.6
PROTON_SHIELD_EFOLD_MM_AL: float = 12.0
ELECTRON_SHIELD_EFOLD_MM_AL: float = 0.9
BREMSSTRAHLUNG_FLOOR: float = 5.0e-3
PROTON_30MEV_RIGIDITY_GV: float = 0.239
SAA_FLUX_THRESHOLD_CM2_S: float = 10.0


def _unit_from_lat_lon(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    return np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])


_DIPOLE_AXIS = _unit_from_lat_lon(GEOMAG_NORTH_POLE_LAT_DEG, GEOMAG_NORTH_POLE_LON_DEG)
_CUTOFF_RADIUS_KM = EARTH_RADIUS_KM + float(np.linalg.norm(DIPOLE_CENTER_OFFSET_KM)) + ATMOSPHERIC_LOSS_ALTITUDE_KM


def geomagnetic_coordinates(r_ecef_km: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Eccentric-dipole field magnitude (nT), McIlwain L-shell and geomagnetic
    latitude (deg) for ECEF positions of shape (..., 3) in km.
    """
    r_d = np.asarray(r_ecef_km, dtype=float) - DIPOLE_CENTER_OFFSET_KM
    r_mag = np.linalg.norm(r_d, axis=-1)
    sin_lam = np.clip((r_d @ _DIPOLE_AXIS) / r_mag, -1.0, 1.0)
    cos2_lam = np.maximum(1.0 - sin_lam**2, 1e-9)
    b_nt = DIPOLE_B0_NT * (EARTH_RADIUS_KM / r_mag) ** 3 * np.sqrt(1.0 + 3.0 * sin_lam**2)
    l_shell = r_mag / (EARTH_RADIUS_KM * cos2_lam)
    return {
        "b_nt": b_nt,
        "l_shell": l_shell,
        "mag_lat_deg": np.degrees(np.arcsin(sin_lam)),
        "dipole_radius_km": r_mag,
    }


def _pitch_angle_factor(b_nt: np.ndarray, l_shell: np.ndarray, exponent: float) -> np.ndarray:
    """Fraction of the equatorial flux mirroring at or below the local point, zero inside the loss cone."""
    b_eq = DIPOLE_B0_NT / l_shell**3
    cos2_c = _CUTOFF_RADIUS_KM / (l_shell * EARTH_RADIUS_KM)
    stable = cos2_c < 1.0
    cos2_c = np.where(stable, cos2_c, 1.0)
    b_cut = b_eq * np.sqrt(4.0 - 3.0 * cos2_c) / cos2_c**3
    x = np.where(stable, (b_cut - b_nt) / np.maximum(b_cut - b_eq, 1e-9), 0.0)
    return np.clip(x, 0.0, 1.0) ** exponent


def trapped_proton_flux(r_ecef_km: np.ndarray) -> np.ndarray:
    """Omnidirectional trapped proton flux >30 MeV (protons / cm^2 / s)."""
    g = geomagnetic_coordinates(r_ecef_km)
    width = np.where(g["l_shell"] < PROTON_PEAK_L, PROTON_INNER_L_WIDTH, PROTON_OUTER_L_WIDTH)
    radial = np.exp(-(((g["l_shell"] - PROTON_PEAK_L) / width) ** 2))
    return PROTON_PEAK_FLUX_CM2_S * radial * _pitch_angle_factor(g["b_nt"], g["l_shell"], PROTON_PITCH_EXPONENT)


def trapped_electron_flux(r_ecef_km: np.ndarray, kp: float = 2.0) -> np.ndarray:
    """Omnidirectional outer-belt electron flux >1 MeV (electrons / cm^2 / s), enhanced with Kp."""
    g = geomagnetic_coordinates(r_ecef_km)
    radial = np.exp(-(((g["l_shell"] - ELECTRON_PEAK_L) / ELECTRON_L_WIDTH) ** 2))
    pitch = np.maximum(
        _pitch_angle_factor(g["b_nt"], g["l_shell"], ELECTRON_PITCH_EXPONENT),
        ELECTRON_DRIFT_LOSS_CONE_FLOOR,
    )
    storm_gain = 10.0 ** (ELECTRON_STORM_LOG10_PER_KP * max(0.0, kp - 2.0))
    return ELECTRON_PEAK_FLUX_CM2_S * radial * pitch * storm_gain


def stormer_cutoff_rigidity_gv(l_shell: np.ndarray) -> np.ndarray:
    """Stormer vertical geomagnetic cutoff rigidity (GV)."""
    return 14.9 / np.asarray(l_shell, dtype=float) ** 2


def sep_access_l_shell(kp: float) -> float:
    """Lowest L-shell reachable by >30 MeV solar protons; the cutoff is suppressed during storms."""
    quiet_l = math.sqrt(14.9 / PROTON_30MEV_RIGIDITY_GV)
    return max(3.5, quiet_l - 0.35 * max(0.0, kp))


def circular_orbit_ecef_track(
    altitude_km: float,
    inclination_deg: float,
    duration_s: float,
    step_s: float = 30.0,
    raan_deg: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Sample a circular orbit in the Earth-fixed frame (geocentric latitude/longitude)."""
    r = EARTH_RADIUS_KM + altitude_km
    n = math.sqrt(EARTH_MU / r**3)
    t = np.arange(0.0, duration_s + 0.5 * step_s, step_s)
    u = n * t
    inc, raan = math.radians(inclination_deg), math.radians(raan_deg)
    x = math.cos(raan) * np.cos(u) - math.sin(raan) * np.sin(u) * math.cos(inc)
    y = math.sin(raan) * np.cos(u) + math.cos(raan) * np.sin(u) * math.cos(inc)
    z = np.sin(u) * math.sin(inc)
    theta = EARTH_ROTATION_RATE_RAD_S * t
    x_e = x * np.cos(theta) + y * np.sin(theta)
    y_e = -x * np.sin(theta) + y * np.cos(theta)
    return {
        "t_s": t,
        "r_ecef_km": r * np.column_stack([x_e, y_e, z]),
        "lat_deg": np.degrees(np.arcsin(np.clip(z, -1.0, 1.0))),
        "lon_deg": np.degrees(np.arctan2(y_e, x_e)),
    }


def compute_saa_flux_map(altitude_km: float, lat_step_deg: float = 5.0, lon_step_deg: float = 5.0) -> Dict[str, Any]:
    """Latitude/longitude grid of trapped proton flux at a fixed altitude (for SAA contour maps)."""
    lats = np.arange(-90.0, 90.0 + 1e-9, lat_step_deg)
    lons = np.arange(-180.0, 180.0 + 1e-9, lon_step_deg)
    lon_g, lat_g = np.meshgrid(np.radians(lons), np.radians(lats))
    r = EARTH_RADIUS_KM + altitude_km
    pos = r * np.stack([np.cos(lat_g) * np.cos(lon_g), np.cos(lat_g) * np.sin(lon_g), np.sin(lat_g)], axis=-1)
    flux = trapped_proton_flux(pos)
    i, j = np.unravel_index(int(np.argmax(flux)), flux.shape)
    return {
        "lat_deg": lats.tolist(),
        "lon_deg": lons.tolist(),
        "proton_flux_cm2_s": np.round(flux, 3).tolist(),
        "peak_flux_cm2_s": float(flux[i, j]),
        "peak_lat_deg": float(lats[i]),
        "peak_lon_deg": float(lons[j]),
    }


@dataclass
class RadiationEnvironmentReport:
    altitude_km: float
    inclination_deg: float
    kp: float
    shielding_mm_al: float
    duration_hours: float
    min_field_nt: float
    max_field_nt: float
    max_l_shell: float
    saa_exposure_fraction: float
    saa_passes_per_day: float
    saa_peak_lat_deg: float
    saa_peak_lon_deg: float
    peak_proton_flux_cm2_s: float
    mean_proton_flux_cm2_s: float
    peak_electron_flux_cm2_s: float
    mean_electron_flux_cm2_s: float
    min_cutoff_rigidity_gv: float
    polar_cap_exposure_fraction: float
    sep_mean_flux_cm2_s: float
    daily_proton_fluence_cm2: float
    daily_electron_fluence_cm2: float
    tid_rad_per_day: float
    tid_krad_per_year: float
    mission_tid_krad: float
    tid_margin_factor: float
    seu_per_day: float
    peak_seu_per_hour: float
    recommended_scrub_interval_s: float
    radiation_status: str
    timeline: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_radiation_environment(
    altitude_km: float = 550.0,
    inclination_deg: float = 53.0,
    kp: float = 2.0,
    shielding_mm_al: float = 2.5,
    duration_hours: float = 24.0,
    mission_years: float = 5.0,
    component_tid_tolerance_krad: float = 30.0,
    memory_mbit: float = 512.0,
    seu_cross_section_cm2_per_bit: float = 1.0e-14,
    sep_flux_pfu: float = 0.0,
    step_s: float = 30.0,
) -> RadiationEnvironmentReport:
    """
    Fly a circular orbit through the trapped-radiation model and integrate
    fluence, shielded dose and SEU rate.

    sep_flux_pfu is the interplanetary >30 MeV solar proton flux
    (protons / cm^2 / s / sr) of an ongoing Solar Particle Event; it reaches
    the spacecraft only where the storm-suppressed geomagnetic cutoff allows.
    """
    if altitude_km <= ATMOSPHERIC_LOSS_ALTITUDE_KM:
        raise ValueError("altitude_km must be above the atmospheric loss altitude (100 km).")
    if shielding_mm_al < 0.0 or duration_hours <= 0.0 or step_s <= 0.0:
        raise ValueError("shielding_mm_al must be >= 0 and duration_hours/step_s must be > 0.")

    track = circular_orbit_ecef_track(altitude_km, inclination_deg, duration_hours * 3600.0, step_s)
    pos = track["r_ecef_km"]
    geo = geomagnetic_coordinates(pos)
    p_flux = trapped_proton_flux(pos)
    e_flux = trapped_electron_flux(pos, kp)

    polar_cap = geo["l_shell"] >= sep_access_l_shell(kp)
    # Half of the sky is shadowed by the Earth in LEO: omnidirectional flux = 2*pi*J
    sep_flux = np.where(polar_cap, 2.0 * math.pi * max(0.0, sep_flux_pfu), 0.0)
    total_p = p_flux + sep_flux

    n = len(p_flux)
    days = duration_hours / 24.0
    in_saa = p_flux > SAA_FLUX_THRESHOLD_CM2_S
    saa_entries = int(np.count_nonzero(in_saa[1:] & ~in_saa[:-1])) + int(in_saa[0])
    i_peak = int(np.argmax(p_flux))

    p_fluence_day = float(np.sum(total_p) * step_s / days)
    e_fluence_day = float(np.sum(e_flux) * step_s / days)

    p_shield = math.exp(-shielding_mm_al / PROTON_SHIELD_EFOLD_MM_AL)
    e_shield = math.exp(-shielding_mm_al / ELECTRON_SHIELD_EFOLD_MM_AL) + BREMSSTRAHLUNG_FLOOR
    tid_day = RAD_PER_MEV_PER_G * (
        p_fluence_day * PROTON_STOPPING_POWER_MEV_CM2_G * p_shield
        + e_fluence_day * ELECTRON_STOPPING_POWER_MEV_CM2_G * e_shield
    )
    tid_year_krad = tid_day * 365.25 / 1000.0
    mission_tid = tid_year_krad * mission_years
    margin = component_tid_tolerance_krad / mission_tid if mission_tid > 0.0 else float("inf")

    bits = memory_mbit * 1.0e6
    seu_day = p_fluence_day * p_shield * seu_cross_section_cm2_per_bit * bits
    peak_seu_hour = float(np.max(total_p)) * p_shield * seu_cross_section_cm2_per_bit * bits * 3600.0
    # Scrub fast enough that <0.1 upsets accumulate between scrubs at the peak rate
    scrub_s = min(3600.0, max(1.0, 0.1 * 3600.0 / peak_seu_hour)) if peak_seu_hour > 0.0 else 3600.0

    if margin < 1.0:
        status = "TID_LIMIT_EXCEEDED"
    elif margin < 2.0:
        status = "RADIATION_DESIGN_MARGIN_LOW"
    elif sep_flux_pfu > 0.0 and bool(np.any(polar_cap)):
        status = "SEP_STORM_EXPOSURE"
    else:
        status = "NOMINAL_RADIATION_ENVIRONMENT"

    stride = max(1, n // 240)
    timeline = [
        {
            "t_min": round(float(track["t_s"][k]) / 60.0, 2),
            "lat_deg": round(float(track["lat_deg"][k]), 3),
            "lon_deg": round(float(track["lon_deg"][k]), 3),
            "b_nt": round(float(geo["b_nt"][k]), 1),
            "l_shell": round(float(geo["l_shell"][k]), 3),
            "proton_flux_cm2_s": round(float(total_p[k]), 3),
            "electron_flux_cm2_s": round(float(e_flux[k]), 3),
        }
        for k in range(0, n, stride)
    ]

    return RadiationEnvironmentReport(
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        kp=kp,
        shielding_mm_al=shielding_mm_al,
        duration_hours=duration_hours,
        min_field_nt=round(float(np.min(geo["b_nt"])), 1),
        max_field_nt=round(float(np.max(geo["b_nt"])), 1),
        max_l_shell=round(min(float(np.max(geo["l_shell"])), 100.0), 3),
        saa_exposure_fraction=round(float(np.mean(in_saa)), 5),
        saa_passes_per_day=round(saa_entries / days, 2),
        saa_peak_lat_deg=round(float(track["lat_deg"][i_peak]), 2),
        saa_peak_lon_deg=round(float(track["lon_deg"][i_peak]), 2),
        peak_proton_flux_cm2_s=round(float(np.max(total_p)), 3),
        mean_proton_flux_cm2_s=round(float(np.mean(total_p)), 4),
        peak_electron_flux_cm2_s=round(float(np.max(e_flux)), 3),
        mean_electron_flux_cm2_s=round(float(np.mean(e_flux)), 4),
        min_cutoff_rigidity_gv=round(float(np.min(stormer_cutoff_rigidity_gv(geo["l_shell"]))), 4),
        polar_cap_exposure_fraction=round(float(np.mean(polar_cap)), 5),
        sep_mean_flux_cm2_s=round(float(np.mean(sep_flux)), 4),
        daily_proton_fluence_cm2=p_fluence_day,
        daily_electron_fluence_cm2=e_fluence_day,
        tid_rad_per_day=round(tid_day, 5),
        tid_krad_per_year=round(tid_year_krad, 5),
        mission_tid_krad=round(mission_tid, 4),
        tid_margin_factor=round(margin, 3) if math.isfinite(margin) else margin,
        seu_per_day=round(seu_day, 4),
        peak_seu_per_hour=round(peak_seu_hour, 4),
        recommended_scrub_interval_s=round(scrub_s, 1),
        radiation_status=status,
        timeline=timeline,
    )
