"""
High-Precision Perturbation Engine: Solar Radiation Pressure & Third-Body Gravity
==================================================================================
Implements Solar Radiation Pressure (SRP) cannonball model with cylindrical and dual-cone
(umbra/penumbra) eclipse geometry, and Third-Body Gravitational Perturbations (Sun and Moon)
using Battin's algebraic formulation to eliminate subtractive precision loss.

Designed for high-precision orbit propagation across LEO, MEO (e.g. GPS, Galileo), and GEO.
Zero external dependencies beyond numpy and Python standard library.
"""

from typing import Tuple, Dict, Any, Optional, Union
import math
from datetime import datetime, timezone
import numpy as np

from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    EARTH_MU,
    SPEED_OF_LIGHT,
    SOLAR_CONSTANT,
)

# ---------------------------------------------------------------------------
# Physical & Astrodynamical Constants (IAU / IERS Standards)
# ---------------------------------------------------------------------------
ASTRONOMICAL_UNIT_KM: float = 149597870.7          # 1 AU in km
ASTRONOMICAL_UNIT_M: float = 149597870700.0        # 1 AU in meters
SUN_MU: float = 1.32712440018e11                   # Solar gravitational parameter (km^3 / s^2)
MOON_MU: float = 4902.800066                       # Lunar gravitational parameter (km^3 / s^2)
SUN_RADIUS_KM: float = 696340.0                    # Nominal solar radius (km)
MOON_RADIUS_KM: float = 1737.4                     # Nominal lunar radius (km)

# Solar Radiation Pressure at 1 AU (N / m^2)
SOLAR_PRESSURE_1AU: float = SOLAR_CONSTANT / SPEED_OF_LIGHT  # ~4.540199e-6 N/m^2

# Default Spacecraft SRP Parameters
DEFAULT_CR: float = 1.2                            # Radiation pressure coefficient (1.0 = absorption, 2.0 = reflection)
DEFAULT_AREA_M2: float = 1.5                       # Effective cross-sectional area (m^2)
DEFAULT_MASS_KG: float = 120.0                     # Spacecraft dry + wet mass (kg)


# ---------------------------------------------------------------------------
# Time & Ephemeris Conversions
# ---------------------------------------------------------------------------

def datetime_to_jd(dt: datetime) -> float:
    """
    Convert a UTC datetime object to Julian Date (JD).
    Uses the standard Meeus / Astronomical Almanac algorithm.
    """
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)

    y, m, d = dt.year, dt.month, dt.day
    h, minute, s, us = dt.hour, dt.minute, dt.second, dt.microsecond

    if m <= 2:
        y -= 1
        m += 12

    a = int(y / 100)
    b = 2 - a + int(a / 4)
    day_fraction = (h + minute / 60.0 + (s + us * 1e-6) / 3600.0) / 24.0
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + day_fraction + b - 1524.5
    return float(jd)


def mjd_to_jd(mjd: float) -> float:
    """Convert Modified Julian Date (MJD) to Julian Date (JD)."""
    return float(mjd + 2400000.5)


def jd_to_mjd(jd: float) -> float:
    """Convert Julian Date (JD) to Modified Julian Date (MJD)."""
    return float(jd - 2400000.5)


def resolve_epoch_jd(epoch: Union[float, int, datetime]) -> float:
    """
    Standardize an epoch representation into a Julian Date (JD).
    Accepts float JD, float MJD (< 100000), or datetime object.
    """
    if isinstance(epoch, datetime):
        return datetime_to_jd(epoch)
    epoch_float = float(epoch)
    # Heuristic: MJD is typically < 100000 (e.g. MJD 60000 is year 2023)
    if epoch_float < 100000.0:
        return mjd_to_jd(epoch_float)
    return epoch_float


# ---------------------------------------------------------------------------
# Analytical Solar & Lunar Ephemerides (ECI / GCRF Frame)
# ---------------------------------------------------------------------------

def sun_position_eci(jd: float) -> np.ndarray:
    """
    Compute low-precision geocentric position of the Sun in ECI (GCRF/J2000) coordinates.
    Based on Montenbruck & Gill / Astronomical Almanac.
    Accuracy: ~0.01 deg (~few arcmin), superior for third-body and SRP calculations.

    Parameters
    ----------
    jd : float
        Julian Date (UTC/TT).

    Returns
    -------
    np.ndarray
        Sun position vector [X, Y, Z] relative to Earth center in km.
    """
    # Julian centuries from J2000.0
    t = (jd - 2451545.0) / 36525.0

    # Geometric mean longitude of Sun (deg)
    l0 = (280.46646 + 36000.76983 * t) % 360.0

    # Mean anomaly of Sun (deg)
    m = (357.52911 + 35999.05029 * t - 0.0001537 * t**2) % 360.0
    m_rad = math.radians(m)

    # Equation of center / true longitude (deg)
    c = (
        (1.914602 - 0.004817 * t - 0.000014 * t**2) * math.sin(m_rad)
        + (0.019993 - 0.000101 * t) * math.sin(2.0 * m_rad)
        + 0.000289 * math.sin(3.0 * m_rad)
    )
    sun_lon_deg = (l0 + c) % 360.0
    sun_lon_rad = math.radians(sun_lon_deg)

    # Sun-Earth distance in AU
    r_au = (
        1.00014061
        - (0.01670862 - 0.000042037 * t) * math.cos(m_rad)
        - 0.000139589 * math.cos(2.0 * m_rad)
    )
    r_km = r_au * ASTRONOMICAL_UNIT_KM

    # Mean obliquity of the ecliptic (deg)
    eps = (23.439291 - 0.0130042 * t) % 360.0
    eps_rad = math.radians(eps)

    # Equatorial ECI coordinates (km)
    x = r_km * math.cos(sun_lon_rad)
    y = r_km * math.cos(eps_rad) * math.sin(sun_lon_rad)
    z = r_km * math.sin(eps_rad) * math.sin(sun_lon_rad)

    return np.array([x, y, z], dtype=float)


def moon_position_eci(jd: float) -> np.ndarray:
    """
    Compute geocentric position of the Moon in ECI (GCRF/J2000) coordinates.
    Based on Meeus / Montenbruck & Gill truncated lunar series.
    Accuracy: ~0.1 deg (~100-300 km), well within required tolerances for
    spacecraft third-body gravitational perturbations.

    Parameters
    ----------
    jd : float
        Julian Date (UTC/TT).

    Returns
    -------
    np.ndarray
        Moon position vector [X, Y, Z] relative to Earth center in km.
    """
    t = (jd - 2451545.0) / 36525.0

    # Fundamental lunar arguments (deg)
    l_prime = (218.3164477 + 481267.88123421 * t) % 360.0  # Mean longitude of Moon
    d = (297.8501921 + 445267.1114034 * t) % 360.0        # Mean elongation of Moon
    m = (357.5291092 + 35999.0502909 * t) % 360.0         # Sun's mean anomaly
    m_prime = (134.9633964 + 477198.8675055 * t) % 360.0  # Moon's mean anomaly
    f = (93.2720950 + 483202.0175233 * t) % 360.0         # Moon's argument of latitude

    d2r = math.pi / 180.0
    d_r = d * d2r
    m_r = m * d2r
    mp_r = m_prime * d2r
    f_r = f * d2r

    # Geocentric ecliptic longitude (deg)
    lon_deg = (
        l_prime
        + 6.288774 * math.sin(mp_r)
        + 1.274027 * math.sin(2.0 * d_r - mp_r)
        + 0.658314 * math.sin(2.0 * d_r)
        + 0.213618 * math.sin(2.0 * mp_r)
        - 0.185116 * math.sin(m_r)
        - 0.114332 * math.sin(2.0 * f_r)
        + 0.058793 * math.sin(2.0 * d_r - 2.0 * mp_r)
        + 0.057066 * math.sin(2.0 * d_r - m_r - mp_r)
        + 0.053322 * math.sin(2.0 * d_r + mp_r)
        + 0.045758 * math.sin(2.0 * d_r - m_r)
        - 0.040923 * math.sin(m_r - mp_r)
        - 0.034720 * math.sin(d_r)
        - 0.030383 * math.sin(m_r + mp_r)
    ) % 360.0

    # Geocentric ecliptic latitude (deg)
    lat_deg = (
        5.128122 * math.sin(f_r)
        + 0.280602 * math.sin(mp_r + f_r)
        + 0.277693 * math.sin(mp_r - f_r)
        + 0.173238 * math.sin(2.0 * d_r - f_r)
        + 0.055413 * math.sin(2.0 * d_r - mp_r + f_r)
        + 0.046271 * math.sin(2.0 * d_r - mp_r - f_r)
        + 0.032573 * math.sin(2.0 * d_r + f_r)
        + 0.017198 * math.sin(2.0 * mp_r + f_r)
    )

    # Distance Earth-Moon (km)
    r_km = (
        385000.56
        - 20905.355 * math.cos(mp_r)
        - 3699.111 * math.cos(2.0 * d_r - mp_r)
        - 2955.968 * math.cos(2.0 * d_r)
        - 569.925 * math.cos(2.0 * mp_r)
        + 48.888 * math.cos(m_r)
        - 3.149 * math.cos(2.0 * f_r)
        + 246.158 * math.cos(2.0 * d_r - 2.0 * mp_r)
        - 152.138 * math.cos(2.0 * d_r - m_r - mp_r)
        - 170.733 * math.cos(2.0 * d_r + mp_r)
        - 204.586 * math.cos(2.0 * d_r - m_r)
        - 129.620 * math.cos(m_r - mp_r)
        + 108.743 * math.cos(d_r)
    )

    # Obliquity of the ecliptic
    eps = (23.439291 - 0.0130042 * t) % 360.0
    eps_rad = eps * d2r
    lon_rad = lon_deg * d2r
    lat_rad = lat_deg * d2r

    # Ecliptic Cartesian coordinates
    x_ecl = r_km * math.cos(lat_rad) * math.cos(lon_rad)
    y_ecl = r_km * math.cos(lat_rad) * math.sin(lon_rad)
    z_ecl = r_km * math.sin(lat_rad)

    # Rotate to Equatorial ECI (GCRF)
    x_eci = x_ecl
    y_eci = y_ecl * math.cos(eps_rad) - z_ecl * math.sin(eps_rad)
    z_eci = y_ecl * math.sin(eps_rad) + z_ecl * math.cos(eps_rad)

    return np.array([x_eci, y_eci, z_eci], dtype=float)


# ---------------------------------------------------------------------------
# Eclipse & Shadow Geometry Models
# ---------------------------------------------------------------------------

def compute_cylindrical_shadow(
    r_sc_km: np.ndarray,
    r_sun_km: np.ndarray,
    r_earth_km: float = EARTH_RADIUS_KM,
) -> float:
    """
    Binary cylindrical shadow model.
    Checks whether the spacecraft falls within Earth's cylindrical shadow cylinder.

    Returns
    -------
    float
        1.0 if illuminated (sunlight), 0.0 if eclipsed (shadow).
    """
    sun_dist = float(np.linalg.norm(r_sun_km))
    if sun_dist < 1.0:
        return 1.0

    sun_unit = r_sun_km / sun_dist
    # Projection along Earth-Sun line
    proj_sun = float(np.dot(r_sc_km, sun_unit))

    # Day side: satellite is facing the Sun
    if proj_sun >= 0.0:
        return 1.0

    # Night side: distance from cylinder center axis
    perp_dist_sq = float(np.dot(r_sc_km, r_sc_km)) - proj_sun**2
    if perp_dist_sq < r_earth_km**2:
        return 0.0

    return 1.0


def compute_dual_cone_shadow(
    r_sc_km: np.ndarray,
    r_sun_km: np.ndarray,
    r_earth_km: float = EARTH_RADIUS_KM,
    r_sun_body_km: float = SUN_RADIUS_KM,
) -> float:
    """
    Continuous dual-cone shadow model modeling Umbra, Penumbra, and Annular eclipse.
    Calculates the visible fraction of the Sun disk occulted by Earth as viewed
    from the spacecraft.

    Returns
    -------
    float
        Shadow factor nu in [0.0, 1.0], where 0.0 is total eclipse (umbra),
        1.0 is full sunlight, and intermediate values represent penumbral transitions.
    """
    d_sun = r_sun_km - r_sc_km
    norm_d_sun = float(np.linalg.norm(d_sun))
    norm_r_sc = float(np.linalg.norm(r_sc_km))

    # Spacecraft at or inside Earth surface
    if norm_r_sc <= r_earth_km:
        return 0.0

    if norm_d_sun < r_sun_body_km:
        return 1.0

    # Apparent angular radii of Sun and Earth as viewed from the spacecraft
    theta_sun = math.asin(min(1.0, r_sun_body_km / norm_d_sun))
    theta_earth = math.asin(min(1.0, r_earth_km / norm_r_sc))

    # Vector from satellite to Earth center is -r_sc_km
    d_earth = -r_sc_km
    dot_val = float(np.dot(d_sun, d_earth))
    denom = norm_d_sun * norm_r_sc
    cos_theta = dot_val / denom if denom > 0.0 else 1.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta = math.acos(cos_theta)

    # Case 1: No occultation (full sunlight)
    if theta >= (theta_sun + theta_earth):
        return 1.0

    # Case 2: Full occultation / Umbra (total eclipse)
    if theta <= (theta_earth - theta_sun):
        return 0.0

    # Case 3: Annular eclipse (Earth disk entirely within Sun disk)
    if theta <= (theta_sun - theta_earth):
        return max(0.0, 1.0 - (theta_earth / theta_sun)**2)

    # Case 4: Partial occultation / Penumbra (overlapping circular disks)
    r1, r2, d = theta_sun, theta_earth, theta
    c1 = (d**2 + r1**2 - r2**2) / (2.0 * d)
    c2 = d - c1

    c1 = max(-r1, min(r1, c1))
    c2 = max(-r2, min(r2, c2))

    term1 = r1**2 * math.acos(c1 / r1) - c1 * math.sqrt(max(0.0, r1**2 - c1**2))
    term2 = r2**2 * math.acos(c2 / r2) - c2 * math.sqrt(max(0.0, r2**2 - c2**2))
    area_overlap = term1 + term2
    area_sun = math.pi * r1**2

    nu = 1.0 - (area_overlap / area_sun)
    return max(0.0, min(1.0, nu))


def compute_shadow_factor(
    r_sc_km: np.ndarray,
    r_sun_km: np.ndarray,
    model: str = "dual_cone",
    r_earth_km: float = EARTH_RADIUS_KM,
) -> float:
    """
    Dispatcher for solar shadow factor evaluation.

    Parameters
    ----------
    r_sc_km : np.ndarray
        Spacecraft position in ECI (km).
    r_sun_km : np.ndarray
        Sun position in ECI (km).
    model : str
        Shadow model: 'dual_cone' (default), 'cylindrical', or 'none'.
    r_earth_km : float
        Earth radius in km.

    Returns
    -------
    float
        Shadow illumination factor nu in [0.0, 1.0].
    """
    model_lower = model.lower()
    if model_lower == "dual_cone":
        return compute_dual_cone_shadow(r_sc_km, r_sun_km, r_earth_km)
    elif model_lower == "cylindrical":
        return compute_cylindrical_shadow(r_sc_km, r_sun_km, r_earth_km)
    elif model_lower in ("none", "illuminated"):
        return 1.0
    else:
        raise ValueError(f"Unknown shadow model '{model}'. Choose 'dual_cone', 'cylindrical', or 'none'.")


# ---------------------------------------------------------------------------
# Solar Radiation Pressure (SRP) Cannonball Acceleration
# ---------------------------------------------------------------------------

def compute_srp_acceleration(
    r_sc_km: np.ndarray,
    r_sun_km: Optional[np.ndarray] = None,
    jd: Optional[float] = None,
    cr: float = DEFAULT_CR,
    area_m2: float = DEFAULT_AREA_M2,
    mass_kg: float = DEFAULT_MASS_KG,
    shadow_factor: Optional[float] = None,
    shadow_model: str = "dual_cone",
) -> np.ndarray:
    """
    Calculate Solar Radiation Pressure (SRP) acceleration vector in km/s^2 using
    the canonical spherical cannonball model.

    The force is directed radially outward from the Sun.
    Formula:
        a_srp = nu * P0 * (AU / d_sun)^2 * Cr * (A / m) * (r_sc - r_sun) / ||r_sc - r_sun|| * 1e-3

    Parameters
    ----------
    r_sc_km : np.ndarray
        Spacecraft geocentric position in ECI (km).
    r_sun_km : Optional[np.ndarray]
        Sun geocentric position in ECI (km). If None, computed from `jd`.
    jd : Optional[float]
        Julian Date. Required if `r_sun_km` is None.
    cr : float
        Radiation pressure coefficient (typically 1.0 to 2.0).
    area_m2 : float
        Effective solar cross-sectional area in m^2.
    mass_kg : float
        Spacecraft mass in kg.
    shadow_factor : Optional[float]
        Explicit shadow illumination factor [0, 1]. If None, computed using `shadow_model`.
    shadow_model : str
        Shadow geometry: 'dual_cone' (default) or 'cylindrical'.

    Returns
    -------
    np.ndarray
        SRP perturbing acceleration vector [ax, ay, az] in km/s^2.
    """
    if mass_kg <= 0.0 or area_m2 <= 0.0:
        return np.zeros(3, dtype=float)

    if r_sun_km is None:
        if jd is None:
            raise ValueError("Either r_sun_km or jd must be provided to compute SRP acceleration.")
        r_sun_km = sun_position_eci(jd)

    # Relative vector from Sun to spacecraft (direction of sunlight propagation)
    d_vec = r_sc_km - r_sun_km
    d_norm = float(np.linalg.norm(d_vec))
    if d_norm < 1.0:
        return np.zeros(3, dtype=float)

    unit_sun_to_sc = d_vec / d_norm

    # Compute shadow factor if not explicitly given
    if shadow_factor is None:
        nu = compute_shadow_factor(r_sc_km, r_sun_km, model=shadow_model)
    else:
        nu = float(np.clip(shadow_factor, 0.0, 1.0))

    if nu <= 1e-12:
        return np.zeros(3, dtype=float)

    # Scaling with inverse square solar distance
    flux_ratio = (ASTRONOMICAL_UNIT_KM / d_norm)**2
    p_local = SOLAR_PRESSURE_1AU * flux_ratio  # N / m^2

    # Acceleration in m/s^2 = (N / m^2) * (m^2) / kg = m / s^2
    a_si = nu * p_local * cr * (area_m2 / mass_kg)

    # Convert m/s^2 to km/s^2
    a_km_s2 = (a_si * 1e-3) * unit_sun_to_sc
    return a_km_s2


# ---------------------------------------------------------------------------
# Third-Body Gravitational Perturbations (Sun & Moon)
# ---------------------------------------------------------------------------

def compute_third_body_acceleration(
    r_sc_km: np.ndarray,
    r_body_km: np.ndarray,
    mu_body: float,
    use_battin: bool = True,
) -> np.ndarray:
    """
    Calculate third-body point-mass gravitational acceleration in km/s^2.

    Uses Battin's algebraic formulation by default to prevent catastrophic
    cancellation when evaluating (d / ||d||^3 - r_b / ||r_b||^3) for ||r_sc|| << ||r_b||.

    Mathematical Formulation:
        Direct:   a = mu * ( (r_b - r_sc) / ||r_b - r_sc||^3 - r_b / ||r_b||^3 )
        Battin:   q = r_sc . (2 * r_b - r_sc) / ||r_b||^2
                  F(q) = (q / (1 - q)^1.5) * (3 - 3q + q^2) / (1 + (1 - q)^1.5)
                  a = (mu / ||r_b||^3) * [ r_b * F(q) - r_sc * (1 + F(q)) ]

    Parameters
    ----------
    r_sc_km : np.ndarray
        Spacecraft geocentric position in ECI (km).
    r_body_km : np.ndarray
        Third body geocentric position in ECI (km).
    mu_body : float
        Gravitational parameter of third body (km^3 / s^2).
    use_battin : bool
        If True (default), use Battin's high-precision formulation.

    Returns
    -------
    np.ndarray
        Third-body perturbing acceleration vector [ax, ay, az] in km/s^2.
    """
    r_body_sq = float(np.dot(r_body_km, r_body_km))
    norm_rb = math.sqrt(r_body_sq)
    if norm_rb < 1.0:
        return np.zeros(3, dtype=float)

    if not use_battin:
        # Standard direct vector formulation
        d_vec = r_body_km - r_sc_km
        norm_d = float(np.linalg.norm(d_vec))
        if norm_d < 1.0:
            return np.zeros(3, dtype=float)
        return mu_body * (d_vec / (norm_d**3) - r_body_km / (norm_rb**3))

    # Battin Formulation:
    # q = r_sc . (2 r_body - r_sc) / ||r_body||^2
    two_rb_minus_r = 2.0 * r_body_km - r_sc_km
    q = float(np.dot(r_sc_km, two_rb_minus_r)) / r_body_sq

    # If spacecraft is exceptionally close to the third body (e.g. lunar mission),
    # fallback to direct formula
    if q >= 0.9 or q <= -10.0:
        d_vec = r_body_km - r_sc_km
        norm_d = float(np.linalg.norm(d_vec))
        return mu_body * (d_vec / (norm_d**3) - r_body_km / (norm_rb**3))

    # Numerically stable F(q) = (1 - q)^(-1.5) - 1
    d_sq = 1.0 - q
    d_15 = d_sq**1.5
    denom = 1.0 + d_15
    f_q = (q / d_15) * (3.0 - 3.0 * q + q**2) / denom

    factor = mu_body / (norm_rb**3)
    a_3b = factor * (r_body_km * f_q - r_sc_km * (1.0 + f_q))
    return a_3b


def compute_solar_third_body_acceleration(
    r_sc_km: np.ndarray,
    jd_or_epoch: Union[float, int, datetime],
    r_sun_km: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute gravitational perturbation of the Sun on an Earth-orbiting spacecraft (km/s^2).
    """
    if r_sun_km is None:
        jd = resolve_epoch_jd(jd_or_epoch)
        r_sun_km = sun_position_eci(jd)
    return compute_third_body_acceleration(r_sc_km, r_sun_km, SUN_MU)


def compute_lunar_third_body_acceleration(
    r_sc_km: np.ndarray,
    jd_or_epoch: Union[float, int, datetime],
    r_moon_km: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute gravitational perturbation of the Moon on an Earth-orbiting spacecraft (km/s^2).
    """
    if r_moon_km is None:
        jd = resolve_epoch_jd(jd_or_epoch)
        r_moon_km = moon_position_eci(jd)
    return compute_third_body_acceleration(r_sc_km, r_moon_km, MOON_MU)


# ---------------------------------------------------------------------------
# Comprehensive Perturbation Evaluator
# ---------------------------------------------------------------------------

def compute_all_perturbations(
    r_sc_km: np.ndarray,
    jd_or_epoch: Union[float, int, datetime],
    cr: float = DEFAULT_CR,
    area_m2: float = DEFAULT_AREA_M2,
    mass_kg: float = DEFAULT_MASS_KG,
    include_sun: bool = True,
    include_moon: bool = True,
    include_srp: bool = True,
    shadow_model: str = "dual_cone",
) -> Dict[str, Any]:
    """
    Evaluate all high-precision non-spherical extra orbital perturbations:
    Sun Third-Body, Moon Third-Body, and Solar Radiation Pressure with eclipse modeling.

    Parameters
    ----------
    r_sc_km : np.ndarray
        Spacecraft position in ECI (km).
    jd_or_epoch : Union[float, int, datetime]
        Epoch as Julian Date, MJD, or datetime.
    cr : float
        SRP reflectivity coefficient.
    area_m2 : float
        SRP cross-sectional area (m^2).
    mass_kg : float
        Spacecraft mass (kg).
    include_sun : bool
        Whether to calculate Sun third-body gravity.
    include_moon : bool
        Whether to calculate Moon third-body gravity.
    include_srp : bool
        Whether to calculate Solar Radiation Pressure.
    shadow_model : str
        Shadow model for SRP ('dual_cone', 'cylindrical', or 'none').

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'a_total_km_s2': sum of all active perturbing accelerations
        - 'a_srp_km_s2': SRP acceleration vector (km/s^2)
        - 'a_sun_km_s2': Sun third-body acceleration vector (km/s^2)
        - 'a_moon_km_s2': Moon third-body acceleration vector (km/s^2)
        - 'shadow_factor': shadow factor nu [0, 1]
        - 'r_sun_km': computed Sun position (km)
        - 'r_moon_km': computed Moon position (km)
        - 'jd': Julian Date used
    """
    jd = resolve_epoch_jd(jd_or_epoch)
    r_sc = np.asarray(r_sc_km, dtype=float)

    # Ephemerides
    r_sun = sun_position_eci(jd) if (include_sun or include_srp) else np.zeros(3)
    r_moon = moon_position_eci(jd) if include_moon else np.zeros(3)

    a_sun = np.zeros(3, dtype=float)
    a_moon = np.zeros(3, dtype=float)
    a_srp = np.zeros(3, dtype=float)
    shadow_factor = 1.0

    if include_sun:
        a_sun = compute_third_body_acceleration(r_sc, r_sun, SUN_MU)

    if include_moon:
        a_moon = compute_third_body_acceleration(r_sc, r_moon, MOON_MU)

    if include_srp:
        shadow_factor = compute_shadow_factor(r_sc, r_sun, model=shadow_model)
        a_srp = compute_srp_acceleration(
            r_sc_km=r_sc,
            r_sun_km=r_sun,
            cr=cr,
            area_m2=area_m2,
            mass_kg=mass_kg,
            shadow_factor=shadow_factor,
        )

    a_total = a_sun + a_moon + a_srp

    return {
        "a_total_km_s2": a_total,
        "a_srp_km_s2": a_srp,
        "a_sun_km_s2": a_sun,
        "a_moon_km_s2": a_moon,
        "shadow_factor": shadow_factor,
        "r_sun_km": r_sun,
        "r_moon_km": r_moon,
        "jd": jd,
    }


def total_perturbation_acceleration(
    r_sc_km: np.ndarray,
    jd_or_epoch: Union[float, int, datetime],
    cr: float = DEFAULT_CR,
    area_m2: float = DEFAULT_AREA_M2,
    mass_kg: float = DEFAULT_MASS_KG,
    include_sun: bool = True,
    include_moon: bool = True,
    include_srp: bool = True,
    shadow_model: str = "dual_cone",
) -> np.ndarray:
    """
    Convenience function returning directly the total perturbing acceleration vector (km/s^2).
    """
    res = compute_all_perturbations(
        r_sc_km=r_sc_km,
        jd_or_epoch=jd_or_epoch,
        cr=cr,
        area_m2=area_m2,
        mass_kg=mass_kg,
        include_sun=include_sun,
        include_moon=include_moon,
        include_srp=include_srp,
        shadow_model=shadow_model,
    )
    return res["a_total_km_s2"]


# ---------------------------------------------------------------------------
# High-Precision RK4 Single Step with Perturbations
# ---------------------------------------------------------------------------

def rk4_step_high_precision(
    r: np.ndarray,
    v: np.ndarray,
    dt: float,
    epoch_jd: float,
    cd: float = 2.2,
    area_drag_m2: float = 1.5,
    mass_kg: float = DEFAULT_MASS_KG,
    cr: float = DEFAULT_CR,
    area_srp_m2: float = DEFAULT_AREA_M2,
    kp: float = 1.5,
    f10_7: float = 70.0,
    include_drag: bool = True,
    include_sun: bool = True,
    include_moon: bool = True,
    include_srp: bool = True,
    shadow_model: str = "dual_cone",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute a single 4th-order Runge-Kutta numerical step including all primary forces:
    - Earth point mass gravity + J2 oblateness
    - Thermospheric aerodynamic drag (NRLMSISE-00 / space weather dynamic)
    - Solar third-body gravitational attraction
    - Lunar third-body gravitational attraction
    - Solar Radiation Pressure (cannonball with eclipse modeling)

    All accelerations are in km/s^2.
    """
    from aegis_leo.physics.propagator import compute_gravitational_and_j2_acceleration
    from aegis_leo.physics.atmosphere import compute_drag_acceleration

    def full_derivatives(pos: np.ndarray, vel: np.ndarray, t_offset_sec: float) -> Tuple[np.ndarray, np.ndarray]:
        dr_dt = vel
        # Primary gravity + J2
        a_grav = compute_gravitational_and_j2_acceleration(pos)

        # Drag (if within sensible atmospheric regime)
        alt = float(np.linalg.norm(pos)) - EARTH_RADIUS_KM
        if include_drag and alt < 1200.0:
            a_drag = compute_drag_acceleration(pos, vel, cd, area_drag_m2, mass_kg, kp, f10_7)
        else:
            a_drag = np.zeros(3, dtype=float)

        # Perturbations (Sun, Moon, SRP) at current sub-step epoch
        step_jd = epoch_jd + (t_offset_sec / 86400.0)
        a_pert = total_perturbation_acceleration(
            r_sc_km=pos,
            jd_or_epoch=step_jd,
            cr=cr,
            area_m2=area_srp_m2,
            mass_kg=mass_kg,
            include_sun=include_sun,
            include_moon=include_moon,
            include_srp=include_srp,
            shadow_model=shadow_model,
        )

        dv_dt = a_grav + a_drag + a_pert
        return dr_dt, dv_dt

    # RK4 coefficients
    k1_r, k1_v = full_derivatives(r, v, 0.0)
    k2_r, k2_v = full_derivatives(r + 0.5 * dt * k1_r, v + 0.5 * dt * k1_v, 0.5 * dt)
    k3_r, k3_v = full_derivatives(r + 0.5 * dt * k2_r, v + 0.5 * dt * k2_v, 0.5 * dt)
    k4_r, k4_v = full_derivatives(r + dt * k3_r, v + dt * k3_v, dt)

    r_next = r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    v_next = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return r_next, v_next


# ---------------------------------------------------------------------------
# Object-Oriented Perturbation Engine
# ---------------------------------------------------------------------------

class HighPrecisionPerturbationEngine:
    """
    High-Precision Perturbation Engine for LEO, MEO, and GEO mission simulation.
    Caches ephemeris states when appropriate and provides unified perturbation evaluation.
    """

    def __init__(
        self,
        cr: float = DEFAULT_CR,
        area_srp_m2: float = DEFAULT_AREA_M2,
        mass_kg: float = DEFAULT_MASS_KG,
        include_sun: bool = True,
        include_moon: bool = True,
        include_srp: bool = True,
        shadow_model: str = "dual_cone",
    ):
        self.cr = float(cr)
        self.area_srp_m2 = float(area_srp_m2)
        self.mass_kg = float(mass_kg)
        self.include_sun = bool(include_sun)
        self.include_moon = bool(include_moon)
        self.include_srp = bool(include_srp)
        self.shadow_model = str(shadow_model)

    def evaluate(
        self,
        r_sc_km: np.ndarray,
        jd_or_epoch: Union[float, int, datetime],
    ) -> Dict[str, Any]:
        """Evaluate all configured perturbations at specified state and epoch."""
        return compute_all_perturbations(
            r_sc_km=r_sc_km,
            jd_or_epoch=jd_or_epoch,
            cr=self.cr,
            area_m2=self.area_srp_m2,
            mass_kg=self.mass_kg,
            include_sun=self.include_sun,
            include_moon=self.include_moon,
            include_srp=self.include_srp,
            shadow_model=self.shadow_model,
        )

    def total_acceleration(
        self,
        r_sc_km: np.ndarray,
        jd_or_epoch: Union[float, int, datetime],
    ) -> np.ndarray:
        """Return total perturbing acceleration vector in km/s^2."""
        return self.evaluate(r_sc_km, jd_or_epoch)["a_total_km_s2"]
