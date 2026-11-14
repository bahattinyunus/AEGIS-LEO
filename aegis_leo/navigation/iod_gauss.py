"""
AEGIS-LEO: Optical Initial Orbit Determination (IOD) Engine
High-fidelity angles-only orbit determination from Right Ascension and Declination triplets.

Algorithms implemented:
1. Gauss Angles-Only Initial Orbit Determination (8th-degree polynomial formulation)
   with optional Universal Keplerian iterative refinement and light-time correction.
2. Laplace Initial Orbit Determination (vector differentiation via Lagrange interpolation).
3. Gibbs 3-Position Vector Method (coplanarity verification and angular momentum formulation).
4. Herrick-Gibbs 3-Position Method (4th-order Taylor series for closely spaced passes).
5. Synthetic Optical Observation Generator (with optional Gaussian astrometric noise).

References:
- Gauss, C. F., 'Theoria Motus Corporum Coelestium' (1809).
- Laplace, P. S., 'Mécanique Céleste' (1799).
- Bate, R. R., Mueller, D. D., White, J. E., 'Fundamentals of Astrodynamics' (1971).
- Vallado, D. A., 'Fundamentals of Astrodynamics and Applications', 4th Ed., Microcosm Press (2013).
- Curtis, H. D., 'Orbital Mechanics for Engineering Students', 4th Ed., Elsevier (2020).
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
import math
import numpy as np

from aegis_leo.constants import (
    EARTH_MU,
    EARTH_RADIUS_KM,
    SPEED_OF_LIGHT,
    EARTH_ROTATION_RATE_RAD_S,
    EARTH_FLATTENING,
)
from aegis_leo.physics.coordinates import cartesian_to_kepler


# Speed of light in km/s
SPEED_OF_LIGHT_KM_S: float = SPEED_OF_LIGHT / 1000.0


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class OpticalObservation:
    """
    Topocentric optical angles-only observation from an optical sensor or ground station.

    Attributes:
        timestamp_s: Observation time in seconds (relative to an epoch).
        ra_deg: Right Ascension in degrees, in range [0, 360).
        dec_deg: Declination in degrees, in range [-90, +90].
        site_eci_km: Observer position vector [X, Y, Z] in km in ECI frame.
    """
    timestamp_s: float
    ra_deg: float
    dec_deg: float
    site_eci_km: np.ndarray

    def __post_init__(self):
        self.site_eci_km = np.asarray(self.site_eci_km, dtype=float).reshape(3,)
        self.ra_deg = float(self.ra_deg) % 360.0
        if not (-90.0 <= self.dec_deg <= 90.0):
            raise ValueError(f"Declination must be in [-90, 90] deg, got {self.dec_deg}")

    @property
    def ra_rad(self) -> float:
        """Right Ascension in radians."""
        return float(np.radians(self.ra_deg))

    @property
    def dec_rad(self) -> float:
        """Declination in radians."""
        return float(np.radians(self.dec_deg))

    @property
    def los_vector(self) -> np.ndarray:
        """
        Unit line-of-sight (LOS) vector L in ECI coordinates:
        L = [cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec)]^T.
        """
        return ra_dec_to_los(self.ra_deg, self.dec_deg)

    def to_dict(self) -> Dict[str, Any]:
        """Convert observation to dictionary format."""
        return {
            "timestamp_s": self.timestamp_s,
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "site_eci_km": self.site_eci_km.tolist(),
            "los_vector": self.los_vector.tolist(),
        }


@dataclass
class IODResult:
    """
    Initial Orbit Determination result containing state vector and orbital elements.

    Attributes:
        r2_eci_km: Estimated geocentric position vector [X, Y, Z] at epoch t2 (km).
        v2_eci_km: Estimated geocentric velocity vector [Vx, Vy, Vz] at epoch t2 (km/s).
        epoch_s: Reference epoch timestamp t2 (seconds).
        slant_ranges_km: Slant ranges (rho1, rho2, rho3) in km.
        elements: Classical Keplerian orbital elements dictionary.
        method: IOD algorithm used ('gauss', 'gauss_iterative', 'laplace', 'gibbs', 'herrick_gibbs').
        residual_rms_arcsec: Root-Mean-Square optical angular residual in arcseconds.
        converged: Whether the iterative solver converged.
        iterations: Number of refinement iterations performed.
        light_time_corrected: Whether finite speed of light travel time was compensated.
    """
    r2_eci_km: np.ndarray
    v2_eci_km: np.ndarray
    epoch_s: float
    slant_ranges_km: Tuple[float, float, float]
    elements: Dict[str, float]
    method: str
    residual_rms_arcsec: float
    converged: bool = True
    iterations: int = 1
    light_time_corrected: bool = False

    @property
    def orbital_elements(self) -> Dict[str, float]:
        """Keplerian orbital elements dictionary alias."""
        return self.elements

    def __post_init__(self):
        self.r2_eci_km = np.asarray(self.r2_eci_km, dtype=float).reshape(3,)
        self.v2_eci_km = np.asarray(self.v2_eci_km, dtype=float).reshape(3,)

    def summary(self) -> str:
        """Format a human-readable text summary of the determined orbit."""
        lines = [
            f"=== Initial Orbit Determination ({self.method.upper()}) ===",
            f"Epoch (t2): {self.epoch_s:.3f} s",
            f"Converged: {self.converged} (iterations: {self.iterations})",
            f"Light-time correction: {self.light_time_corrected}",
            f"Slant Ranges (rho1, rho2, rho3): {self.slant_ranges_km[0]:.2f}, "
            f"{self.slant_ranges_km[1]:.2f}, {self.slant_ranges_km[2]:.2f} km",
            f"Position r2: [{self.r2_eci_km[0]:.3f}, {self.r2_eci_km[1]:.3f}, {self.r2_eci_km[2]:.3f}] km "
            f"(norm: {np.linalg.norm(self.r2_eci_km):.3f} km)",
            f"Velocity v2: [{self.v2_eci_km[0]:.4f}, {self.v2_eci_km[1]:.4f}, {self.v2_eci_km[2]:.4f}] km/s "
            f"(norm: {np.linalg.norm(self.v2_eci_km):.4f} km/s)",
            f"Semi-Major Axis (a): {self.elements.get('semi_major_axis_km', 0.0):.3f} km",
            f"Eccentricity (e): {self.elements.get('eccentricity', 0.0):.6f}",
            f"Inclination (i): {self.elements.get('inclination_deg', 0.0):.3f} deg",
            f"RAAN (Omega): {self.elements.get('raan_deg', 0.0):.3f} deg",
            f"Arg of Perigee (omega): {self.elements.get('arg_of_perigee_deg', 0.0):.3f} deg",
            f"True Anomaly (nu): {self.elements.get('true_anomaly_deg', 0.0):.3f} deg",
            f"Orbital Period: {self.elements.get('period_minutes', 0.0):.2f} min",
            f"Perigee Altitude: {self.elements.get('perigee_altitude_km', 0.0):.2f} km",
            f"Apogee Altitude: {self.elements.get('apogee_altitude_km', 0.0):.2f} km",
            f"Optical Residual RMS: {self.residual_rms_arcsec:.3f} arcsec",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert IOD result to dictionary format."""
        return {
            "epoch_s": self.epoch_s,
            "method": self.method,
            "converged": self.converged,
            "iterations": self.iterations,
            "light_time_corrected": self.light_time_corrected,
            "r2_eci_km": self.r2_eci_km.tolist(),
            "v2_eci_km": self.v2_eci_km.tolist(),
            "slant_ranges_km": list(self.slant_ranges_km),
            "elements": self.elements,
            "residual_rms_arcsec": self.residual_rms_arcsec,
        }


# ---------------------------------------------------------------------------
# Geometric & Coordinate Transformations
# ---------------------------------------------------------------------------

def ra_dec_to_los(ra_deg: float, dec_deg: float) -> np.ndarray:
    """
    Convert Right Ascension and Declination angles (degrees) to a 3D unit
    line-of-sight (LOS) direction vector in the ECI frame.
    """
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    cos_dec = np.cos(dec)
    return np.array([
        cos_dec * np.cos(ra),
        cos_dec * np.sin(ra),
        np.sin(dec),
    ], dtype=float)


def los_to_ra_dec(los_vec: np.ndarray) -> Tuple[float, float]:
    """
    Convert a 3D line-of-sight vector in ECI to Right Ascension (deg in [0, 360))
    and Declination (deg in [-90, +90]).
    """
    v = np.asarray(los_vec, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Line-of-sight vector norm is near zero.")
    u = v / norm

    dec_rad = np.arcsin(np.clip(u[2], -1.0, 1.0))
    ra_rad = np.arctan2(u[1], u[0])
    if ra_rad < 0.0:
        ra_rad += 2.0 * np.pi

    return float(np.degrees(ra_rad)), float(np.degrees(dec_rad))


def ground_station_eci(
    lat_deg: float,
    lon_deg: float,
    alt_km: float,
    gmst_rad: float,
) -> np.ndarray:
    """
    Compute geocentric ECI position vector [X, Y, Z] of a ground observer station
    on the WGS84 ellipsoid accounting for Greenwich Mean Sidereal Time (GMST).

    Args:
        lat_deg: Geodetic latitude (degrees, -90 to +90).
        lon_deg: Geodetic longitude (degrees, -180 to +180 or 0 to 360).
        alt_km: Height above WGS84 ellipsoid in km.
        gmst_rad: Greenwich Mean Sidereal Time in radians.

    Returns:
        3D numpy array [X, Y, Z] in km in ECI frame.
    """
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    a = EARTH_RADIUS_KM
    e2 = 2.0 * EARTH_FLATTENING - EARTH_FLATTENING**2

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    n_val = a / np.sqrt(1.0 - e2 * (sin_phi**2))

    # WGS84 Cartesian coordinates in ECEF
    x_ecef = (n_val + alt_km) * cos_phi * np.cos(lam)
    y_ecef = (n_val + alt_km) * cos_phi * np.sin(lam)
    z_ecef = (n_val * (1.0 - e2) + alt_km) * sin_phi

    # Rotate ECEF to ECI by GMST (rotation about Z-axis)
    cos_g = np.cos(gmst_rad)
    sin_g = np.sin(gmst_rad)
    x_eci = cos_g * x_ecef - sin_g * y_ecef
    y_eci = sin_g * x_ecef + cos_g * y_ecef
    z_eci = z_ecef

    return np.array([x_eci, y_eci, z_eci], dtype=float)


def apparent_ra_dec(
    r_sat_eci: np.ndarray,
    r_site_eci: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute topocentric apparent Right Ascension (deg), Declination (deg),
    and slant range (km) of a satellite as viewed from an observer.
    """
    rho_vec = np.asarray(r_sat_eci, dtype=float) - np.asarray(r_site_eci, dtype=float)
    slant_range_km = float(np.linalg.norm(rho_vec))
    ra_deg, dec_deg = los_to_ra_dec(rho_vec)
    return ra_deg, dec_deg, slant_range_km


def angular_separation_deg(los1: np.ndarray, los2: np.ndarray) -> float:
    """
    Compute the great-circle angular separation in degrees between two unit vectors.
    """
    u1 = np.asarray(los1, dtype=float) / np.linalg.norm(los1)
    u2 = np.asarray(los2, dtype=float) / np.linalg.norm(los2)
    cos_ang = np.clip(np.dot(u1, u2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_ang)))


def compute_optical_residuals(
    r_sat_eci: np.ndarray,
    obs: OpticalObservation,
) -> Tuple[float, float, float]:
    """
    Compute angular residuals in arcseconds between observed and predicted line-of-sight.

    Returns:
        (delta_ra_arcsec, delta_dec_arcsec, total_angular_residual_arcsec)
    """
    pred_ra, pred_dec, _ = apparent_ra_dec(r_sat_eci, obs.site_eci_km)

    # RA difference wrapped to [-180, 180] deg, scaled by cos(dec)
    d_ra = (obs.ra_deg - pred_ra + 180.0) % 360.0 - 180.0
    d_dec = obs.dec_deg - pred_dec

    d_ra_scaled_deg = d_ra * np.cos(np.radians(obs.dec_deg))
    d_ra_arcsec = float(d_ra_scaled_deg * 3600.0)
    d_dec_arcsec = float(d_dec * 3600.0)

    total_arcsec = float(np.sqrt(d_ra_arcsec**2 + d_dec_arcsec**2))
    return d_ra_arcsec, d_dec_arcsec, total_arcsec


# ---------------------------------------------------------------------------
# Universal Keplerian Propagator
# ---------------------------------------------------------------------------

def stumpff_c(z: float) -> float:
    """
    Stumpff function C(z) = (1 - cos(sqrt(z))) / z with series fallback.
    """
    if z > 1e-6:
        return float((1.0 - np.cos(np.sqrt(z))) / z)
    elif z < -1e-6:
        return float((np.cosh(np.sqrt(-z)) - 1.0) / (-z))
    else:
        # Taylor series expansion
        return float(0.5 - z / 24.0 + (z**2) / 720.0 - (z**3) / 40320.0)


def stumpff_s(z: float) -> float:
    """
    Stumpff function S(z) = (sqrt(z) - sin(sqrt(z))) / (sqrt(z))^3 with series fallback.
    """
    if z > 1e-6:
        sz = np.sqrt(z)
        return float((sz - np.sin(sz)) / (sz**3))
    elif z < -1e-6:
        sz = np.sqrt(-z)
        return float((np.sinh(sz) - sz) / (sz**3))
    else:
        # Taylor series expansion
        return float(1.0 / 6.0 - z / 120.0 + (z**2) / 5040.0 - (z**3) / 362880.0)


def kepler_universal_propagate(
    r0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    mu: float = EARTH_MU,
    tol: float = 1e-11,
    max_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """
    Propagate Cartesian orbital state (r0, v0) by time dt using the Universal
    Keplerian variable formulation (Battin / Bate-Mueller-White).

    Returns:
        (r, v, f, g, f_dot, g_dot)
    """
    r0_vec = np.asarray(r0, dtype=float)
    v0_vec = np.asarray(v0, dtype=float)

    if abs(dt) < 1e-12:
        return np.copy(r0_vec), np.copy(v0_vec), 1.0, 0.0, 0.0, 1.0

    r0_norm = float(np.linalg.norm(r0_vec))
    v0_norm = float(np.linalg.norm(v0_vec))
    alpha = 2.0 / r0_norm - (v0_norm**2) / mu
    sqrt_mu = np.sqrt(mu)
    r0_dot_v0 = float(np.dot(r0_vec, v0_vec))

    # Initial guess for universal anomaly chi
    if alpha > 1e-6:
        # Elliptic orbit
        chi = sqrt_mu * dt * alpha
    elif alpha < -1e-6:
        # Hyperbolic orbit
        a = 1.0 / alpha
        chi = np.sign(dt) * np.sqrt(-a) * np.log(
            -2.0 * mu * alpha * dt / (
                r0_dot_v0 + np.sign(dt) * np.sqrt(-mu * a) * (1.0 - r0_norm * alpha)
            )
        )
    else:
        # Parabolic orbit
        h_vec = np.cross(r0_vec, v0_vec)
        p = np.dot(h_vec, h_vec) / mu
        s = 0.5 * np.arctan(1.0 / (3.0 * np.sqrt(mu / (p**3)) * dt))
        w = np.arctan(np.tan(s)**(1.0 / 3.0))
        chi = np.sqrt(p) * 2.0 / np.tan(2.0 * w)

    # Newton-Raphson iteration on universal Kepler equation
    for _ in range(max_iter):
        z = alpha * (chi**2)
        c = stumpff_c(z)
        s = stumpff_s(z)
        f_val = (
            (r0_dot_v0 / sqrt_mu) * (chi**2) * c
            + (1.0 - alpha * r0_norm) * (chi**3) * s
            + r0_norm * chi
            - sqrt_mu * dt
        )
        f_prime = (
            (r0_dot_v0 / sqrt_mu) * chi * (1.0 - z * s)
            + (1.0 - alpha * r0_norm) * (chi**2) * c
            + r0_norm
        )
        delta_chi = f_val / f_prime
        chi -= delta_chi
        if abs(delta_chi) < tol:
            break

    z = alpha * (chi**2)
    c = stumpff_c(z)
    s = stumpff_s(z)

    # Lagrange f and g coefficients
    f = 1.0 - ((chi**2) / r0_norm) * c
    g = dt - ((chi**3) / sqrt_mu) * s

    r = f * r0_vec + g * v0_vec
    r_norm = float(np.linalg.norm(r))

    f_dot = (sqrt_mu / (r_norm * r0_norm)) * (z * s - 1.0) * chi
    g_dot = 1.0 - ((chi**2) / r_norm) * c

    v = f_dot * r0_vec + g_dot * v0_vec

    return r, v, float(f), float(g), float(f_dot), float(g_dot)


# ---------------------------------------------------------------------------
# 3-Position Vector Methods: Gibbs & Herrick-Gibbs
# ---------------------------------------------------------------------------

def gibbs_iod(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    mu: float = EARTH_MU,
    coplanar_tol_deg: float = 3.0,
) -> np.ndarray:
    """
    Gibbs method of Initial Orbit Determination from three geocentric position vectors.
    Computes the velocity vector v2 at the intermediate position r2.

    Args:
        r1: Geocentric position vector at t1 (km).
        r2: Geocentric position vector at t2 (km).
        r3: Geocentric position vector at t3 (km).
        mu: Gravitational parameter (km^3/s^2).
        coplanar_tol_deg: Maximum allowable non-coplanarity angle (degrees).

    Returns:
        v2: Geocentric velocity vector at t2 (km/s).

    Raises:
        ValueError: If position vectors are collinear, non-coplanar, or disordered.
    """
    r1_vec = np.asarray(r1, dtype=float).reshape(3,)
    r2_vec = np.asarray(r2, dtype=float).reshape(3,)
    r3_vec = np.asarray(r3, dtype=float).reshape(3,)

    r1_norm = float(np.linalg.norm(r1_vec))
    r2_norm = float(np.linalg.norm(r2_vec))
    r3_norm = float(np.linalg.norm(r3_vec))

    if r1_norm < 1e-3 or r2_norm < 1e-3 or r3_norm < 1e-3:
        raise ValueError("Position vector norm is too small (< 1 meter).")

    # Cross products
    c12 = np.cross(r1_vec, r2_vec)
    c23 = np.cross(r2_vec, r3_vec)
    c31 = np.cross(r3_vec, r1_vec)

    c23_norm = float(np.linalg.norm(c23))
    if c23_norm < 1e-9:
        raise ValueError("r2 and r3 are collinear.")

    # Check coplanarity: angle between r1 and the orbital plane normal (r2 x r3)
    u_c23 = c23 / c23_norm
    sin_coplanar = abs(np.dot(u_c23, r1_vec)) / r1_norm
    coplanar_angle_deg = float(np.degrees(np.arcsin(np.clip(sin_coplanar, 0.0, 1.0))))
    if coplanar_angle_deg > coplanar_tol_deg:
        raise ValueError(
            f"Position vectors fail coplanarity check: "
            f"out-of-plane angle {coplanar_angle_deg:.3f}° exceeds tolerance {coplanar_tol_deg}°"
        )

    # Gibbs auxiliary vectors N, D, S
    N = r1_norm * c23 + r2_norm * c31 + r3_norm * c12
    D = c12 + c23 + c31
    S = r1_vec * (r2_norm - r3_norm) + r2_vec * (r3_norm - r1_norm) + r3_vec * (r1_norm - r2_norm)

    N_norm = float(np.linalg.norm(N))
    D_norm = float(np.linalg.norm(D))

    if N_norm < 1e-9 or D_norm < 1e-9:
        raise ValueError("Degenerate geometry in Gibbs method (N or D norm near zero).")

    # Verify sequential order along orbit trajectory
    if np.dot(N, D) < 0.0:
        raise ValueError("Vectors N and D point in opposite directions; verify observation ordering.")

    # Velocity at point 2
    v2 = np.sqrt(mu / (N_norm * D_norm)) * ((np.cross(D, r2_vec) / r2_norm) + S)
    return v2


def herrick_gibbs_iod(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    mu: float = EARTH_MU,
) -> np.ndarray:
    """
    Herrick-Gibbs method of Initial Orbit Determination for closely-spaced position vectors.
    Uses 4th-order Taylor series with observation timestamps. Ideal for angular separations < 5°.

    Args:
        r1, r2, r3: Geocentric position vectors (km).
        t1, t2, t3: Observation timestamps (seconds).
        mu: Gravitational parameter (km^3/s^2).

    Returns:
        v2: Geocentric velocity vector at t2 (km/s).
    """
    r1_vec = np.asarray(r1, dtype=float).reshape(3,)
    r2_vec = np.asarray(r2, dtype=float).reshape(3,)
    r3_vec = np.asarray(r3, dtype=float).reshape(3,)

    dt32 = t3 - t2
    dt21 = t2 - t1
    dt31 = t3 - t1

    if abs(dt21) < 1e-6 or abs(dt32) < 1e-6 or abs(dt31) < 1e-6:
        raise ValueError("Time intervals between observations must be non-zero.")

    r1_norm = float(np.linalg.norm(r1_vec))
    r2_norm = float(np.linalg.norm(r2_vec))
    r3_norm = float(np.linalg.norm(r3_vec))

    term1 = -dt32 * (1.0 / (dt21 * dt31) + mu / (12.0 * (r1_norm**3))) * r1_vec
    term2 = (dt32 - dt21) * (1.0 / (dt21 * dt32) + mu / (12.0 * (r2_norm**3))) * r2_vec
    term3 = dt21 * (1.0 / (dt32 * dt31) + mu / (12.0 * (r3_norm**3))) * r3_vec

    return term1 + term2 + term3


def three_position_iod(
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    mu: float = EARTH_MU,
    method: str = "auto",
    coplanar_tol_deg: float = 3.0,
) -> np.ndarray:
    """
    Determine velocity v2 from three geocentric position vectors.
    In 'auto' mode, automatically selects Herrick-Gibbs for angular separations < 2.5°
    and Gibbs for wider separations.
    """
    ang_sep_13 = angular_separation_deg(r1, r3)

    if method == "auto":
        if ang_sep_13 < 2.5:
            return herrick_gibbs_iod(r1, r2, r3, t1, t2, t3, mu=mu)
        else:
            return gibbs_iod(r1, r2, r3, mu=mu, coplanar_tol_deg=coplanar_tol_deg)
    elif method == "gibbs":
        return gibbs_iod(r1, r2, r3, mu=mu, coplanar_tol_deg=coplanar_tol_deg)
    elif method == "herrick_gibbs":
        return herrick_gibbs_iod(r1, r2, r3, t1, t2, t3, mu=mu)
    else:
        raise ValueError(f"Unknown three-position IOD method: '{method}'")


# ---------------------------------------------------------------------------
# Gauss Angles-Only Initial Orbit Determination
# ---------------------------------------------------------------------------

def gauss_iod(
    obs1: OpticalObservation,
    obs2: OpticalObservation,
    obs3: OpticalObservation,
    mu: float = EARTH_MU,
    iterate: bool = True,
    max_iter: int = 50,
    tol: float = 1e-8,
    light_time_correction: bool = False,
    min_radius_km: float = EARTH_RADIUS_KM,
) -> IODResult:
    """
    Gauss Initial Orbit Determination from three optical angles-only observations
    (Right Ascension and Declination triplets).

    Solves the fundamental 8th-degree polynomial for the middle geocentric radius r2,
    determines slant ranges rho1, rho2, rho3, and computes Cartesian state (r2, v2).
    Optionally refines using Universal Keplerian propagation and light-time compensation.

    Args:
        obs1, obs2, obs3: Optical observations in chronological order (t1 < t2 < t3).
        mu: Gravitational parameter (km^3/s^2).
        iterate: If True, iteratively refine f and g coefficients via universal anomaly.
        max_iter: Maximum iterations for refinement.
        tol: Convergence tolerance on position norm change (km).
        light_time_correction: If True, correct timestamps for light travel time (rho / c).
        min_radius_km: Minimum acceptable physical geocentric radius (default: Earth radius).

    Returns:
        IODResult object with estimated state, Keplerian elements, and residuals.
    """
    # Working timestamps
    t1 = obs1.timestamp_s
    t2 = obs2.timestamp_s
    t3 = obs3.timestamp_s

    if not (t1 < t2 < t3):
        raise ValueError(f"Observations must be strictly chronological: t1={t1}, t2={t2}, t3={t3}")

    L1 = obs1.los_vector
    L2 = obs2.los_vector
    L3 = obs3.los_vector

    R1 = obs1.site_eci_km
    R2 = obs2.site_eci_km
    R3 = obs3.site_eci_km

    # Matrix M = [L1, L2, L3]
    M = np.column_stack([L1, L2, L3])
    det_M = float(np.linalg.det(M))
    if abs(det_M) < 1e-8:
        raise ValueError(
            f"Line-of-sight vectors are near-coplanar (det(M) = {det_M:.2e}). "
            "Gauss IOD requires non-coplanar sightlines."
        )

    M_inv = np.linalg.inv(M)
    m1 = M_inv[0, :]
    m2 = M_inv[1, :]
    m3 = M_inv[2, :]

    # Slant range initial estimate (for light time correction)
    rho_est = np.array([1000.0, 1000.0, 1000.0])

    # Outer loop for light-time correction (typically 1-3 iterations)
    outer_loops = 3 if light_time_correction else 1
    iterations_performed = 0
    converged = False

    r1_vec = np.zeros(3)
    r2_vec = np.zeros(3)
    r3_vec = np.zeros(3)
    v2_vec = np.zeros(3)
    rho1, rho2, rho3 = 0.0, 0.0, 0.0

    for _ in range(outer_loops):
        if light_time_correction:
            # Correct observation times for light travel delay
            t1_eff = t1 - (rho_est[0] / SPEED_OF_LIGHT_KM_S)
            t2_eff = t2 - (rho_est[1] / SPEED_OF_LIGHT_KM_S)
            t3_eff = t3 - (rho_est[2] / SPEED_OF_LIGHT_KM_S)
        else:
            t1_eff, t2_eff, t3_eff = t1, t2, t3

        tau1 = t1_eff - t2_eff
        tau3 = t3_eff - t2_eff
        tau = t3_eff - t1_eff

        # Initial series approximation for f and g
        a1 = tau3 / tau
        a3 = -tau1 / tau
        b1 = (a1 / 6.0) * (tau**2 - tau3**2)
        b3 = (a3 / 6.0) * (tau**2 - tau1**2)

        A = a1 * R1 - R2 + a3 * R3
        B = mu * (b1 * R1 + b3 * R3)

        d1 = float(np.dot(m2, A))
        d2 = float(np.dot(m2, B))

        C = float(np.dot(L2, R2))
        R2_sq = float(np.dot(R2, R2))

        # 8th-degree polynomial: x^8 - (d1^2 + 2*C*d1 + R2^2)*x^6 - 2*d2*(d1 + C)*x^3 - d2^2 = 0
        poly_coeffs = [
            1.0,
            0.0,
            -(d1**2 + 2.0 * C * d1 + R2_sq),
            0.0,
            0.0,
            -2.0 * d2 * (d1 + C),
            0.0,
            0.0,
            -(d2**2),
        ]

        all_roots = np.roots(poly_coeffs)
        real_pos_roots: List[Tuple[float, float]] = []
        for root in all_roots:
            if abs(root.imag) < 1e-6 and root.real > 0.0:
                val = float(root.real)
                rho2_cand = d1 + d2 / (val**3)
                if rho2_cand > 0.0:
                    real_pos_roots.append((val, rho2_cand))

        if not real_pos_roots:
            raise ValueError("No physical positive real roots found for Gauss 8th-order polynomial.")

        # Filter roots satisfying minimum physical orbit radius
        orbit_roots = [c for c in real_pos_roots if c[0] >= min_radius_km]
        if orbit_roots:
            # Pick the root closest to Earth orbital realm
            r2_mag, rho2 = orbit_roots[0]
        else:
            r2_mag, rho2 = real_pos_roots[0]

        u = mu / (r2_mag**3)
        c1 = a1 + b1 * u
        c3 = a3 + b3 * u

        rhs = c1 * R1 - R2 + c3 * R3
        rho1 = float(-(1.0 / c1) * np.dot(m1, rhs))
        rho2 = float(np.dot(m2, rhs))
        rho3 = float(-(1.0 / c3) * np.dot(m3, rhs))

        r1_vec = R1 + rho1 * L1
        r2_vec = R2 + rho2 * L2
        r3_vec = R3 + rho3 * L3

        # Compute initial velocity via f and g series or Herrick-Gibbs
        f1_ser = 1.0 - 0.5 * u * (tau1**2)
        f3_ser = 1.0 - 0.5 * u * (tau3**2)
        g1_ser = tau1 - (1.0 / 6.0) * u * (tau1**3)
        g3_ser = tau3 - (1.0 / 6.0) * u * (tau3**3)
        denom_ser = f1_ser * g3_ser - f3_ser * g1_ser
        v2_vec = (f1_ser * r3_vec - f3_ser * r1_vec) / denom_ser

        iterations_performed = 1
        converged = True

        # Iterative refinement of f and g coefficients using universal Kepler propagation
        if iterate:
            converged = False
            for it in range(1, max_iter + 1):
                iterations_performed = it

                # Propagate from r2, v2 by tau1 and tau3
                _, _, f1_ex, g1_ex, _, _ = kepler_universal_propagate(r2_vec, v2_vec, tau1, mu=mu)
                _, _, f3_ex, g3_ex, _, _ = kepler_universal_propagate(r2_vec, v2_vec, tau3, mu=mu)

                denom_ex = f1_ex * g3_ex - f3_ex * g1_ex
                if abs(denom_ex) < 1e-12:
                    break

                c1_new = g3_ex / denom_ex
                c3_new = -g1_ex / denom_ex

                rhs_new = c1_new * R1 - R2 + c3_new * R3
                rho1_new = float(-(1.0 / c1_new) * np.dot(m1, rhs_new))
                rho2_new = float(np.dot(m2, rhs_new))
                rho3_new = float(-(1.0 / c3_new) * np.dot(m3, rhs_new))

                r1_new = R1 + rho1_new * L1
                r2_new = R2 + rho2_new * L2
                r3_new = R3 + rho3_new * L3

                v2_new = (f1_ex * r3_new - f3_ex * r1_new) / denom_ex

                diff_r = float(np.linalg.norm(r2_new - r2_vec))
                r1_vec = r1_new
                r2_vec = r2_new
                r3_vec = r3_new
                v2_vec = v2_new
                rho1, rho2, rho3 = rho1_new, rho2_new, rho3_new

                if diff_r < tol:
                    converged = True
                    break

        rho_est = np.array([rho1, rho2, rho3])

    # Convert state to Keplerian orbital elements
    elements = cartesian_to_kepler(r2_vec, v2_vec, mu=mu)

    # Compute RMS angular residuals across the three observations
    # Propagate r2, v2 to t1 and t3 to evaluate optical residuals
    dt1 = obs1.timestamp_s - obs2.timestamp_s
    dt3 = obs3.timestamp_s - obs2.timestamp_s
    r1_eval, _, _, _, _, _ = kepler_universal_propagate(r2_vec, v2_vec, dt1, mu=mu)
    r3_eval, _, _, _, _, _ = kepler_universal_propagate(r2_vec, v2_vec, dt3, mu=mu)

    _, _, res1 = compute_optical_residuals(r1_eval, obs1)
    _, _, res2 = compute_optical_residuals(r2_vec, obs2)
    _, _, res3 = compute_optical_residuals(r3_eval, obs3)
    rms_arcsec = float(np.sqrt((res1**2 + res2**2 + res3**2) / 3.0))

    method_str = "gauss_iterative" if iterate else "gauss"

    return IODResult(
        r2_eci_km=r2_vec,
        v2_eci_km=v2_vec,
        epoch_s=obs2.timestamp_s,
        slant_ranges_km=(rho1, rho2, rho3),
        elements=elements,
        method=method_str,
        residual_rms_arcsec=rms_arcsec,
        converged=converged,
        iterations=iterations_performed,
        light_time_corrected=light_time_correction,
    )


# ---------------------------------------------------------------------------
# Laplace Initial Orbit Determination
# ---------------------------------------------------------------------------

def laplace_iod(
    obs1: OpticalObservation,
    obs2: OpticalObservation,
    obs3: OpticalObservation,
    mu: float = EARTH_MU,
    obs_site_velocities: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    obs_site_accelerations: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    min_radius_km: float = EARTH_RADIUS_KM,
) -> IODResult:
    """
    Laplace Initial Orbit Determination from three optical angles-only observations.

    Uses Lagrange polynomial differentiation to compute first and second derivatives
    of the line-of-sight vector L and observer site vector R at intermediate epoch t2.
    Solves the Laplace determinant system and 8th-order polynomial for geocentric radius r,
    slant range rho, and range rate rho_dot.

    Args:
        obs1, obs2, obs3: Optical observations in chronological order.
        mu: Gravitational parameter (km^3/s^2).
        obs_site_velocities: Optional (R_dot1, R_dot2, R_dot3) site velocities (km/s).
        obs_site_accelerations: Optional (R_ddot1, R_ddot2, R_ddot3) site accelerations (km/s^2).
        min_radius_km: Minimum acceptable physical geocentric radius (km).

    Returns:
        IODResult object.
    """
    t1 = obs1.timestamp_s
    t2 = obs2.timestamp_s
    t3 = obs3.timestamp_s

    if not (t1 < t2 < t3):
        raise ValueError(f"Observations must be strictly chronological: t1={t1}, t2={t2}, t3={t3}")

    L1 = obs1.los_vector
    L2 = obs2.los_vector
    L3 = obs3.los_vector

    R1 = obs1.site_eci_km
    R2 = obs2.site_eci_km
    R3 = obs3.site_eci_km

    tau1 = t1 - t2
    tau3 = t3 - t2

    # 1st and 2nd time derivatives of Lagrange interpolating polynomials evaluated at t = t2
    dL1 = -tau3 / (tau1 * (tau1 - tau3))
    dL2 = -(tau1 + tau3) / (tau1 * tau3)
    dL3 = tau1 / (tau3 * (tau1 - tau3))

    d2L1 = 2.0 / (tau1 * (tau1 - tau3))
    d2L2 = 2.0 / (tau1 * tau3)
    d2L3 = 2.0 / (tau3 * (tau3 - tau1))

    L = L2
    L_dot = dL1 * L1 + dL2 * L2 + dL3 * L3
    L_ddot = d2L1 * L1 + d2L2 * L2 + d2L3 * L3

    R = R2
    if obs_site_velocities is not None:
        R_dot = np.asarray(obs_site_velocities[1], dtype=float)
    else:
        R_dot = dL1 * R1 + dL2 * R2 + dL3 * R3

    if obs_site_accelerations is not None:
        R_ddot = np.asarray(obs_site_accelerations[1], dtype=float)
    else:
        R_ddot = d2L1 * R1 + d2L2 * R2 + d2L3 * R3

    # System determinant D = det([L, L_dot, L_ddot])
    D = float(np.linalg.det(np.column_stack([L, L_dot, L_ddot])))
    if abs(D) < 1e-12:
        raise ValueError(
            f"Singular geometry in Laplace IOD: determinant D = {D:.2e} is near zero."
        )

    # Cramer's determinants
    D1 = float(np.linalg.det(np.column_stack([L, L_dot, R_ddot])))
    D2 = float(np.linalg.det(np.column_stack([L, L_dot, R])))
    D3 = float(np.linalg.det(np.column_stack([L, R_ddot, L_ddot])))
    D4 = float(np.linalg.det(np.column_stack([L, R, L_ddot])))

    # Slant range relation: rho = A + B / r^3
    A = -D1 / D
    B = -mu * D2 / D

    C = float(np.dot(L, R))
    R_sq = float(np.dot(R, R))

    # 8th degree polynomial in r
    poly_coeffs = [
        1.0,
        0.0,
        -(A**2 + 2.0 * A * C + R_sq),
        0.0,
        0.0,
        -2.0 * B * (A + C),
        0.0,
        0.0,
        -(B**2),
    ]

    all_roots = np.roots(poly_coeffs)
    real_pos_roots: List[Tuple[float, float]] = []
    for root in all_roots:
        if abs(root.imag) < 1e-6 and root.real > 0.0:
            val = float(root.real)
            rho_cand = A + B / (val**3)
            if rho_cand > 0.0:
                real_pos_roots.append((val, rho_cand))

    if not real_pos_roots:
        raise ValueError("No physical positive real roots found in Laplace 8th-order polynomial.")

    orbit_roots = [c for c in real_pos_roots if c[0] >= min_radius_km]
    if orbit_roots:
        r_mag, rho2 = orbit_roots[0]
    else:
        r_mag, rho2 = real_pos_roots[0]

    # Range rate rho_dot from Cramer's rule on column 2:
    rho_dot2 = - (D3 + (mu / (r_mag**3)) * D4) / (2.0 * D)

    # State vector at epoch t2
    r2_vec = R + rho2 * L
    v2_vec = R_dot + rho_dot2 * L + rho2 * L_dot

    elements = cartesian_to_kepler(r2_vec, v2_vec, mu=mu)

    # Propagate to t1 and t3 to obtain rho1 and rho3 and compute residuals
    dt1 = obs1.timestamp_s - obs2.timestamp_s
    dt3 = obs3.timestamp_s - obs2.timestamp_s
    r1_eval, _, _, _, _, _ = kepler_universal_propagate(r2_vec, v2_vec, dt1, mu=mu)
    r3_eval, _, _, _, _, _ = kepler_universal_propagate(r2_vec, v2_vec, dt3, mu=mu)

    rho1 = float(np.linalg.norm(r1_eval - R1))
    rho3 = float(np.linalg.norm(r3_eval - R3))

    _, _, res1 = compute_optical_residuals(r1_eval, obs1)
    _, _, res2 = compute_optical_residuals(r2_vec, obs2)
    _, _, res3 = compute_optical_residuals(r3_eval, obs3)
    rms_arcsec = float(np.sqrt((res1**2 + res2**2 + res3**2) / 3.0))

    return IODResult(
        r2_eci_km=r2_vec,
        v2_eci_km=v2_vec,
        epoch_s=obs2.timestamp_s,
        slant_ranges_km=(rho1, rho2, rho3),
        elements=elements,
        method="laplace",
        residual_rms_arcsec=rms_arcsec,
        converged=True,
        iterations=1,
        light_time_corrected=False,
    )


# ---------------------------------------------------------------------------
# Unified IOD Dispatcher
# ---------------------------------------------------------------------------

def initial_orbit_determination(
    obs1: Optional[OpticalObservation] = None,
    obs2: Optional[OpticalObservation] = None,
    obs3: Optional[OpticalObservation] = None,
    method: str = "auto",
    mu: float = EARTH_MU,
    observations: Optional[List[OpticalObservation]] = None,
    **kwargs: Any,
) -> IODResult:
    """
    Unified Initial Orbit Determination entry point.
    """
    if observations is not None and len(observations) >= 3:
        obs1, obs2, obs3 = observations[0], observations[1], observations[2]
    if obs1 is None or obs2 is None or obs3 is None:
        raise ValueError("Three optical observations (obs1, obs2, obs3) or observations list required.")

    method_lower = method.lower()
    if method_lower in ("auto", "gauss_iterative"):
        return gauss_iod(obs1, obs2, obs3, mu=mu, iterate=True, **kwargs)
    elif method_lower == "gauss":
        return gauss_iod(obs1, obs2, obs3, mu=mu, iterate=False, **kwargs)
    elif method_lower == "laplace":
        return laplace_iod(obs1, obs2, obs3, mu=mu, **kwargs)
    else:
        raise ValueError(f"Unknown IOD method '{method}'. Supported: 'auto', 'gauss', 'gauss_iterative', 'laplace'.")


# ---------------------------------------------------------------------------
# Synthetic Optical Observation Generator
# ---------------------------------------------------------------------------

def generate_synthetic_optical_observations(
    r0: Optional[np.ndarray] = None,
    v0: Optional[np.ndarray] = None,
    site_lat_deg: float = 35.0,
    site_lon_deg: float = -106.0,
    site_alt_km: float = 2.0,
    observation_times_s: Optional[List[float]] = None,
    gmst0_rad: float = 0.0,
    noise_arcsec: float = 0.0,
    light_time_delay: bool = False,
    seed: Optional[int] = None,
    mu: float = EARTH_MU,
    altitude_km: Optional[float] = None,
    inclination_deg: Optional[float] = None,
    tracking_interval_s: Optional[float] = None,
) -> List[OpticalObservation]:
    """
    Generate synthetic optical angles-only observations (RA and Dec) from a ground station
    tracking a satellite with an initial orbital state (r0, v0).

    Args:
        r0: Initial position vector at t=0 (km).
        v0: Initial velocity vector at t=0 (km/s).
        site_lat_deg: Ground station geodetic latitude (deg).
        site_lon_deg: Ground station geodetic longitude (deg).
        site_alt_km: Ground station altitude above ellipsoid (km).
        observation_times_s: List of observation timestamps (seconds from t=0).
        gmst0_rad: Initial Greenwich Mean Sidereal Time at t=0 (radians).
        noise_arcsec: 1-sigma astrometric Gaussian angular noise (arcseconds).
        light_time_delay: If True, model finite speed of light travel delay (photon emission time).
        seed: Optional random seed for reproducible noise.
        mu: Gravitational parameter (km^3/s^2).

    Returns:
        List of OpticalObservation objects.
    """
    if altitude_km is not None:
        r_mag = EARTH_RADIUS_KM + float(altitude_km)
        inc_rad = math.radians(float(inclination_deg if inclination_deg is not None else 53.0))
        v_mag = math.sqrt(mu / r_mag)
        r0 = np.array([r_mag, 0.0, 0.0])
        v0 = np.array([0.0, v_mag * math.cos(inc_rad), v_mag * math.sin(inc_rad)])
        dt = tracking_interval_s if tracking_interval_s is not None else 120.0
        observation_times_s = [0.0, dt, 2.0 * dt]

    if r0 is None or v0 is None:
        r0 = np.array([EARTH_RADIUS_KM + 550.0, 0.0, 0.0])
        v_mag = math.sqrt(mu / (EARTH_RADIUS_KM + 550.0))
        v0 = np.array([0.0, v_mag * math.cos(math.radians(53.0)), v_mag * math.sin(math.radians(53.0))])

    if observation_times_s is None:
        observation_times_s = [0.0, 120.0, 240.0]
    rng = np.random.default_rng(seed)
    observations: List[OpticalObservation] = []

    for t in sorted(observation_times_s):
        # Ground station ECI position at reception epoch t
        gmst = gmst0_rad + EARTH_ROTATION_RATE_RAD_S * float(t)
        site_eci = ground_station_eci(site_lat_deg, site_lon_deg, site_alt_km, gmst)

        # Propagate satellite state to observation epoch t (or emission epoch if light-time enabled)
        t_emit = float(t)
        if light_time_delay:
            # Iteratively solve t_emit = t - ||r(t_emit) - site(t)|| / c
            for _ in range(5):
                r_cand, _, _, _, _, _ = kepler_universal_propagate(r0, v0, t_emit, mu=mu)
                rho_cand = float(np.linalg.norm(r_cand - site_eci))
                t_emit = float(t) - (rho_cand / SPEED_OF_LIGHT_KM_S)

        r_sat, _, _, _, _, _ = kepler_universal_propagate(r0, v0, t_emit, mu=mu)

        # True apparent RA and Dec
        ra_deg, dec_deg, _ = apparent_ra_dec(r_sat, site_eci)

        # Add Gaussian astrometric noise if requested
        if noise_arcsec > 0.0:
            noise_deg = noise_arcsec / 3600.0
            # RA noise scaled by 1/cos(dec) to preserve isotropic angular error
            cos_dec = max(np.cos(np.radians(dec_deg)), 1e-4)
            d_ra = float(rng.normal(0.0, noise_deg / cos_dec))
            d_dec = float(rng.normal(0.0, noise_deg))
            ra_deg = (ra_deg + d_ra) % 360.0
            dec_deg = np.clip(dec_deg + d_dec, -90.0, 90.0)

        obs = OpticalObservation(
            timestamp_s=float(t),
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            site_eci_km=site_eci,
        )
        observations.append(obs)

    return observations


def batch_optical_screening(
    observations: List[OpticalObservation],
    window_size: int = 3,
    method: str = "gauss_iterative",
    mu: float = EARTH_MU,
) -> List[IODResult]:
    """
    Perform sliding-window Initial Orbit Determination across a sequence of N >= 3
    optical tracking observations.

    Args:
        observations: Chronological list of OpticalObservation objects.
        window_size: Number of observations per triplet window (default 3).
        method: IOD algorithm ('gauss', 'gauss_iterative', 'laplace').
        mu: Gravitational parameter (km^3/s^2).

    Returns:
        List of IODResult objects for each valid observation triplet.
    """
    if len(observations) < 3:
        raise ValueError(f"At least 3 observations required for optical IOD, got {len(observations)}")

    results: List[IODResult] = []
    n_obs = len(observations)

    for i in range(0, n_obs - 2):
        obs1 = observations[i]
        obs2 = observations[i + 1]
        obs3 = observations[i + 2]
        try:
            res = initial_orbit_determination(obs1, obs2, obs3, method=method, mu=mu)
            results.append(res)
        except Exception:
            # Skip degenerate triplet windows
            continue

    return results
