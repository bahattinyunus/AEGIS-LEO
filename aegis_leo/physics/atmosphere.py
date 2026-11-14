"""
Physics-Informed Thermospheric Density & Aerodynamic Drag Model
Couples altitude, geomagnetic storm index (Kp), and solar radio flux (F10.7)
Calibrated against Jacchia-77 / NRLMSISE-00 thermospheric response.
"""

from typing import Tuple, Dict, Any
import numpy as np
from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    QUIET_KP,
    QUIET_F10_7,
)
from aegis_leo.physics.coordinates import relative_atmospheric_velocity


# Base reference atmospheric table [altitude_km, base_density_kg_m3, scale_height_km]
# Derived from US Standard Atmosphere & Jacchia Reference Atmosphere
ATMOSPHERE_REF_TABLE = np.array([
    [100.0, 5.297e-7, 5.877],
    [150.0, 2.076e-9, 26.74],
    [200.0, 2.541e-10, 37.10],
    [250.0, 6.073e-11, 45.54],
    [300.0, 1.916e-11, 53.59],
    [350.0, 7.014e-12, 53.29],
    [400.0, 2.803e-12, 58.51],
    [450.0, 1.184e-12, 60.83],
    [500.0, 5.215e-13, 63.82],
    [550.0, 2.378e-13, 68.70],
    [600.0, 1.137e-13, 75.33],
    [650.0, 5.670e-14, 83.15],
    [700.0, 3.070e-14, 91.80],
    [750.0, 1.770e-14, 100.8],
    [800.0, 1.170e-14, 110.1],
    [900.0, 5.200e-15, 131.0],
    [1000.0, 3.019e-15, 268.0],
])


def get_base_density_and_scale_height(altitude_km: float) -> Tuple[float, float]:
    """
    Interpolate base quiet density (kg/m^3) and scale height (km) for a given altitude.
    """
    alt = np.clip(altitude_km, 100.0, 1000.0)
    table_alts = ATMOSPHERE_REF_TABLE[:, 0]

    # Find bounding index
    idx = np.searchsorted(table_alts, alt)
    if idx == 0:
        idx = 1
    elif idx >= len(table_alts):
        idx = len(table_alts) - 1

    h0 = ATMOSPHERE_REF_TABLE[idx - 1, 0]
    h1 = ATMOSPHERE_REF_TABLE[idx, 0]
    rho0 = ATMOSPHERE_REF_TABLE[idx - 1, 1]
    rho1 = ATMOSPHERE_REF_TABLE[idx, 1]
    H0 = ATMOSPHERE_REF_TABLE[idx - 1, 2]
    H1 = ATMOSPHERE_REF_TABLE[idx, 2]

    # Log-linear interpolation for density, linear for scale height
    fraction = (alt - h0) / (h1 - h0)
    log_rho = np.log(rho0) + fraction * (np.log(rho1) - np.log(rho0))
    rho = float(np.exp(log_rho))
    H = float(H0 + fraction * (H1 - H0))

    return rho, H


def space_weather_density_multiplier(
    altitude_km: float,
    kp: float = QUIET_KP,
    f10_7: float = QUIET_F10_7,
) -> float:
    """
    Compute density enhancement factor resulting from:
    1. Geomagnetic storm Joule heating and auroral excitation (Kp >= 4).
    2. Solar Extreme Ultraviolet (EUV) irradiance variation (F10.7 flux).

    Notice: Thermospheric expansion has higher relative impact at altitudes
    between 350 km and 650 km where the atmosphere swells dramatically.
    """
    kp_val = max(0.0, min(float(kp), 9.0))
    f107_val = max(50.0, float(f10_7))

    # Solar EUV expansion factor (normalized to quiet F10.7 = 70.0)
    f_diff = max(0.0, f107_val - 70.0)
    # EUV heating effect scales with altitude
    alt_factor = np.clip((altitude_km - 200.0) / 400.0, 0.2, 1.8)
    euv_mult = 1.0 + 0.0035 * f_diff * alt_factor

    # Geomagnetic storm factor (Kp index effect)
    # When Kp rises above quiet (Kp > 2.5), exponential thermospheric heating occurs
    if kp_val > 2.5:
        delta_kp = kp_val - 2.5
        # Joule heating scales quadratically with geomagnetic disturbance
        # For Kp=8 (G4 storm), density typically rises 3x to 5x at 500 km
        # For Kp=9 (G5 storm), density can rise 6x to 8x
        storm_mult = 1.0 + 0.12 * (delta_kp ** 1.65) * (1.0 + 0.4 * alt_factor)
    else:
        storm_mult = 1.0

    return float(euv_mult * storm_mult)


def compute_atmospheric_density(
    altitude_km: float,
    kp: float = QUIET_KP,
    f10_7: float = QUIET_F10_7,
) -> float:
    """
    Calculate dynamic thermospheric density rho (kg/m^3) at specified altitude and space weather conditions.
    """
    if altitude_km < 80.0:
        return 1.0e-5  # Lower boundary
    if altitude_km > 1200.0:
        return 1.0e-17 # Upper exosphere limit

    base_rho, H = get_base_density_and_scale_height(altitude_km)
    multiplier = space_weather_density_multiplier(altitude_km, kp, f10_7)
    return float(base_rho * multiplier)


def compute_drag_acceleration(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    cd: float,
    area_m2: float,
    mass_kg: float,
    kp: float = QUIET_KP,
    f10_7: float = QUIET_F10_7,
) -> np.ndarray:
    """
    Calculate aerodynamic drag acceleration vector in ECI frame:
    a_drag = - 1/2 * rho(h, Kp, F10.7) * (Cd * A / m) * v_rel^2 * v_rel_hat
    Returns acceleration in km/s^2.
    """
    r_mag_km = np.linalg.norm(r_eci_km)
    altitude_km = r_mag_km - EARTH_RADIUS_KM

    if altitude_km > 1200.0 or mass_kg <= 0.0 or area_m2 <= 0.0:
        return np.zeros(3, dtype=float)

    # Thermospheric density (kg / m^3)
    rho = compute_atmospheric_density(altitude_km, kp, f10_7)

    # Relative velocity to rotating atmosphere (km/s)
    v_rel_km_s = relative_atmospheric_velocity(r_eci_km, v_eci_km_s)
    v_rel_mag_km_s = np.linalg.norm(v_rel_km_s)

    if v_rel_mag_km_s < 1e-9:
        return np.zeros(3, dtype=float)

    v_rel_unit = v_rel_km_s / v_rel_mag_km_s

    # Convert to SI units for acceleration calculation:
    # v_rel in m/s: v_rel_mag_km_s * 1000.0
    # a_drag_m_s2 = 0.5 * rho * (cd * area / mass) * (v_rel_m_s)^2
    v_rel_m_s = v_rel_mag_km_s * 1000.0
    b_star_inv = (cd * area_m2) / mass_kg  # m^2 / kg

    a_drag_m_s2 = 0.5 * rho * b_star_inv * (v_rel_m_s ** 2)

    # Convert back to km/s^2
    a_drag_km_s2 = (a_drag_m_s2 / 1000.0) * (-v_rel_unit)

    return a_drag_km_s2


def compute_ballistic_coefficient(mass_kg: float, cd: float, area_m2: float) -> float:
    """
    Ballistic coefficient BC = m / (Cd * A) in kg / m^2.
    Higher BC = less sensitive to atmospheric drag deceleration.
    """
    if cd <= 0 or area_m2 <= 0:
        return float("inf")
    return float(mass_kg / (cd * area_m2))
