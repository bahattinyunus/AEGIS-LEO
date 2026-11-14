"""
Orbital Mechanics & Coordinate Transformations
ECI (J2000), Keplerian Elements, Geodetic Coordinates, and Relative Velocities.
"""

from typing import Tuple, Dict, Any
import numpy as np
from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    EARTH_MU,
    EARTH_ROTATION_RATE_RAD_S,
    EARTH_FLATTENING,
)


def kepler_to_cartesian(
    a_km: float,
    ecc: float,
    inc_deg: float,
    raan_deg: float,
    argp_deg: float,
    nu_deg: float,
    mu: float = EARTH_MU,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert classical Keplerian orbital elements to ECI Cartesian position (km) and velocity (km/s).
    """
    inc = np.radians(inc_deg)
    raan = np.radians(raan_deg)
    argp = np.radians(argp_deg)
    nu = np.radians(nu_deg)

    # Semi-latus rectum
    p = a_km * (1.0 - ecc**2)
    # Radial distance
    r_mag = p / (1.0 + ecc * np.cos(nu))

    # Position and velocity in orbital perifocal coordinate system (PQW)
    r_pqw = np.array([r_mag * np.cos(nu), r_mag * np.sin(nu), 0.0], dtype=float)
    v_factor = np.sqrt(mu / p) if p > 0 else 0.0
    v_pqw = np.array([-v_factor * np.sin(nu), v_factor * (ecc + np.cos(nu)), 0.0], dtype=float)

    # Rotation matrix from PQW to ECI
    # R = R_z(-raan) * R_x(-inc) * R_z(-argp)
    cos_raan, sin_raan = np.cos(raan), np.sin(raan)
    cos_inc, sin_inc = np.cos(inc), np.sin(inc)
    cos_argp, sin_argp = np.cos(argp), np.sin(argp)

    P_x = cos_raan * cos_argp - sin_raan * sin_argp * cos_inc
    P_y = sin_raan * cos_argp + cos_raan * sin_argp * cos_inc
    P_z = sin_argp * sin_inc

    Q_x = -cos_raan * sin_argp - sin_raan * cos_argp * cos_inc
    Q_y = -sin_raan * sin_argp + cos_raan * cos_argp * cos_inc
    Q_z = cos_argp * sin_inc

    W_x = sin_raan * sin_inc
    W_y = -cos_raan * sin_inc
    W_z = cos_inc

    rot_matrix = np.column_stack([np.array([P_x, P_y, P_z]),
                                  np.array([Q_x, Q_y, Q_z]),
                                  np.array([W_x, W_y, W_z])])

    r_eci = rot_matrix @ r_pqw
    v_eci = rot_matrix @ v_pqw

    return r_eci, v_eci


def cartesian_to_kepler(
    r_eci: np.ndarray,
    v_eci: np.ndarray,
    mu: float = EARTH_MU,
) -> Dict[str, float]:
    """
    Convert ECI position (km) and velocity (km/s) to classical Keplerian elements.
    """
    r_vec = np.asarray(r_eci, dtype=float)
    v_vec = np.asarray(v_eci, dtype=float)
    r_mag = np.linalg.norm(r_vec)
    v_mag = np.linalg.norm(v_vec)

    # Angular momentum vector
    h_vec = np.cross(r_vec, v_vec)
    h_mag = np.linalg.norm(h_vec)

    # Specific mechanical energy
    energy = 0.5 * (v_mag**2) - (mu / r_mag)

    # Semi-major axis (km)
    if abs(energy) > 1e-9:
        a_km = -mu / (2.0 * energy)
    else:
        a_km = float("inf")

    # Eccentricity vector
    e_vec = (1.0 / mu) * ((v_mag**2 - mu / r_mag) * r_vec - np.dot(r_vec, v_vec) * v_vec)
    ecc = float(np.linalg.norm(e_vec))

    # Inclination
    inc_rad = np.arccos(np.clip(h_vec[2] / h_mag, -1.0, 1.0))
    inc_deg = float(np.degrees(inc_rad))

    # Node vector n = k x h
    k_unit = np.array([0.0, 0.0, 1.0])
    n_vec = np.cross(k_unit, h_vec)
    n_mag = np.linalg.norm(n_vec)

    # Right ascension of ascending node (RAAN)
    if n_mag > 1e-9:
        raan_rad = np.arccos(np.clip(n_vec[0] / n_mag, -1.0, 1.0))
        if n_vec[1] < 0:
            raan_rad = 2.0 * np.pi - raan_rad
        raan_deg = float(np.degrees(raan_rad))
    else:
        raan_deg = 0.0

    # Argument of perigee
    if n_mag > 1e-9 and ecc > 1e-8:
        argp_rad = np.arccos(np.clip(np.dot(n_vec, e_vec) / (n_mag * ecc), -1.0, 1.0))
        if e_vec[2] < 0:
            argp_rad = 2.0 * np.pi - argp_rad
        argp_deg = float(np.degrees(argp_rad))
    else:
        argp_deg = 0.0

    # True anomaly
    if ecc > 1e-8:
        nu_rad = np.arccos(np.clip(np.dot(e_vec, r_vec) / (ecc * r_mag), -1.0, 1.0))
        if np.dot(r_vec, v_vec) < 0:
            nu_rad = 2.0 * np.pi - nu_rad
        nu_deg = float(np.degrees(nu_rad))
    else:
        nu_deg = 0.0

    period_s = 2.0 * np.pi * np.sqrt(abs(a_km)**3 / mu) if a_km > 0 else 0.0

    return {
        "semi_major_axis_km": float(a_km),
        "eccentricity": ecc,
        "inclination_deg": inc_deg,
        "raan_deg": raan_deg,
        "arg_of_perigee_deg": argp_deg,
        "true_anomaly_deg": nu_deg,
        "period_minutes": float(period_s / 60.0),
        "perigee_altitude_km": float(a_km * (1.0 - ecc) - EARTH_RADIUS_KM),
        "apogee_altitude_km": float(a_km * (1.0 + ecc) - EARTH_RADIUS_KM),
    }


def eci_to_geodetic(r_eci: np.ndarray, gmst_rad: float = 0.0) -> Tuple[float, float, float]:
    """
    Convert ECI position vector (km) to Geodetic latitude (deg), longitude (deg), altitude (km)
    accounting for Earth rotation (gmst_rad) and ellipsoidal flattening.
    """
    x, y, z = r_eci[0], r_eci[1], r_eci[2]

    # Rotate by GMST to ECEF
    cos_g, sin_g = np.cos(gmst_rad), np.sin(gmst_rad)
    x_ecef = cos_g * x + sin_g * y
    y_ecef = -sin_g * x + cos_g * y
    z_ecef = z

    lon_deg = float(np.degrees(np.arctan2(y_ecef, x_ecef)))

    # Bowring's method for geodetic latitude & altitude
    p = np.sqrt(x_ecef**2 + y_ecef**2)
    a = EARTH_RADIUS_KM
    e2 = 2.0 * EARTH_FLATTENING - EARTH_FLATTENING**2
    b = a * (1.0 - EARTH_FLATTENING)

    if p < 1e-6:
        lat_deg = 90.0 if z_ecef > 0 else -90.0
        alt_km = abs(z_ecef) - b
        return lat_deg, lon_deg, float(alt_km)

    theta = np.arctan2(z_ecef * a, p * b)
    lat_rad = np.arctan2(
        z_ecef + (e2 * a**2 / b) * (np.sin(theta)**3),
        p - e2 * a * (np.cos(theta)**3),
    )
    lat_deg = float(np.degrees(lat_rad))

    n_val = a / np.sqrt(1.0 - e2 * (np.sin(lat_rad)**2))
    alt_km = float(p / np.cos(lat_rad) - n_val)

    return lat_deg, lon_deg, alt_km


def relative_atmospheric_velocity(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    """
    Compute satellite velocity vector relative to the rotating atmosphere (km/s).
    v_rel = v_eci - omega_earth x r_eci
    """
    omega = np.array([0.0, 0.0, EARTH_ROTATION_RATE_RAD_S])  # rad/s
    omega_cross_r = np.cross(omega, r_eci)  # km/s
    return v_eci - omega_cross_r
