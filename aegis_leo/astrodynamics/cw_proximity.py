"""
AEGIS-LEO: Clohessy-Wiltshire & Yamanaka-Ankersen Relative Motion Engine
Implements Hill-Clohessy-Wiltshire (HCW) circular relative motion dynamics,
Yamanaka-Ankersen (YA) elliptical relative motion dynamics, approach corridor
safety boundaries (V-bar / R-bar glideslope limits, Keep-Out Zones), and
multi-impulse (2-impulse and 4-impulse) Active Debris Removal (ADR) rendezvous planning.

References:
    - Clohessy, W. H., and Wiltshire, R. S., "Terminal Guidance System for Satellite
      Rendezvous," Journal of the Aerospace Sciences, Vol. 27, No. 9, 1960, pp. 653-658.
    - Hill, G. W., "Researches in the Lunar Theory," American Journal of Mathematics,
      Vol. 1, No. 1, 1878, pp. 5-26.
    - Yamanaka, K., and Ankersen, F., "New State Transition Matrix for Relative
      Motion on an Arbitrary Elliptical Orbit," Journal of Guidance, Control, and
      Dynamics, Vol. 25, No. 1, 2002, pp. 60-66.
    - Fehse, W., "Automated Rendezvous and Docking of Spacecraft," Cambridge University
      Press, 2003.
    - Hablani, H. B., et al., "Guidance and Relative Navigation for Autonomous
      Rendezvous in a Circular Orbit," Journal of Guidance, Control, and Dynamics, 2002.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from aegis_leo.constants import (
    EARTH_MU,
    EARTH_MU_SI,
    EARTH_RADIUS_M,
    STANDARD_GRAVITY,
)


# ==============================================================================
# ENUMS & DATA STRUCTURES
# ==============================================================================

class CorridorType(str, Enum):
    """Approach corridor orientation in Hill / LVLH frame."""
    V_BAR = "V-bar"          # Along-track (+/- y direction, velocity axis)
    V_BAR_POS = "+V-bar"     # Approach from behind (+y toward 0)
    V_BAR_NEG = "-V-bar"     # Approach from ahead (-y toward 0)
    R_BAR = "R-bar"          # Radial (+/- x direction, Earth center / zenith axis)
    R_BAR_POS = "+R-bar"     # Approach from below (+x toward 0, Earth side)
    R_BAR_NEG = "-R-bar"     # Approach from above (-x toward 0, zenith side)


@dataclass
class CorridorViolation:
    """Record of a single safety corridor violation during proximity operations."""
    timestamp_s: float
    state: np.ndarray             # [x, y, z, xdot, ydot, zdot] in meters and m/s
    violation_type: str          # "CONE", "VELOCITY_LIMIT", "VELOCITY_DIRECTION", "KOZ"
    magnitude: float             # Value of violated metric
    threshold: float             # Allowable limit
    description: str


@dataclass
class ApproachSafetyReport:
    """Comprehensive safety validation report for proximity operations trajectory."""
    is_safe: bool
    corridor_type: str
    cone_violations: List[CorridorViolation] = field(default_factory=list)
    velocity_violations: List[CorridorViolation] = field(default_factory=list)
    koz_violations: List[CorridorViolation] = field(default_factory=list)
    max_lateral_deviation_m: float = 0.0
    min_distance_to_target_m: float = float("inf")
    hold_points_detected: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""


@dataclass
class RelativeOrbitGeometry:
    """Geometric parameters of a closed or drifting relative orbit."""
    center_x_m: float
    center_y_m: float
    semi_major_axis_m: float     # Along-track amplitude
    semi_minor_axis_m: float     # Radial amplitude (semi_major / 2 for closed HCW)
    out_of_plane_amplitude_m: float
    period_s: float
    secular_drift_rate_m_s: float
    is_closed: bool


@dataclass
class RendezvousPlan:
    """Multi-impulse trajectory plan for ADR rendezvous and proximity operations."""
    maneuver_type: str           # "2-impulse" or "4-impulse"
    delta_v_1: np.ndarray        # First impulse vector (m/s)
    delta_v_2: np.ndarray        # Second impulse vector (m/s)
    delta_v_3: Optional[np.ndarray] = None  # Third impulse (for 4-impulse)
    delta_v_4: Optional[np.ndarray] = None  # Fourth impulse (for 4-impulse)
    total_delta_v: float = 0.0
    transfer_time_s: float = 0.0
    leg_times_s: List[float] = field(default_factory=list)
    propellant_used_kg: float = 0.0
    trajectory_times: np.ndarray = field(default_factory=lambda: np.array([]))
    trajectory_states: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    safety_report: Optional[ApproachSafetyReport] = None
    success: bool = True
    message: str = "Rendezvous plan successfully generated."


# ==============================================================================
# KEPLERIAN ANOMALY SOLVERS
# ==============================================================================

def mean_to_eccentric_anomaly(mean_anomaly: float, eccentricity: float,
                              tol: float = 1e-13, max_iter: int = 100) -> float:
    """Solve Kepler's equation M = E - e*sin(E) for Eccentric Anomaly using Newton-Raphson.

    Parameters:
        mean_anomaly: Mean anomaly M in radians.
        eccentricity: Orbit eccentricity e in [0, 1).
        tol: Convergence tolerance on residual.
        max_iter: Maximum iterations.

    Returns:
        Eccentric anomaly E in radians.
    """
    if eccentricity < 0.0 or eccentricity >= 1.0:
        raise ValueError(f"Eccentricity must be in [0, 1), got {eccentricity}")

    if eccentricity == 0.0:
        return mean_anomaly % (2.0 * math.pi)

    # Normalize mean anomaly to [0, 2*pi)
    m = mean_anomaly % (2.0 * math.pi)

    # Initial guess
    if eccentricity < 0.8:
        e_curr = m
    else:
        e_curr = math.pi if m < math.pi else 2.0 * math.pi

    for _ in range(max_iter):
        f = e_curr - eccentricity * math.sin(e_curr) - m
        f_prime = 1.0 - eccentricity * math.cos(e_curr)
        # Halley's correction step for quadratic convergence
        f_double_prime = eccentricity * math.sin(e_curr)
        delta_e = f / (f_prime - 0.5 * f * f_double_prime / f_prime)
        e_curr -= delta_e
        if abs(delta_e) < tol:
            return e_curr

    return e_curr


def eccentric_to_true_anomaly(eccentric_anomaly: float, eccentricity: float) -> float:
    """Convert eccentric anomaly E to true anomaly nu in radians."""
    if eccentricity < 0.0 or eccentricity >= 1.0:
        raise ValueError(f"Eccentricity must be in [0, 1), got {eccentricity}")

    if eccentricity == 0.0:
        return eccentric_anomaly % (2.0 * math.pi)

    beta = eccentricity / (1.0 + math.sqrt(1.0 - eccentricity**2))
    nu = eccentric_anomaly + 2.0 * math.atan2(
        beta * math.sin(eccentric_anomaly),
        1.0 - beta * math.cos(eccentric_anomaly)
    )
    return nu % (2.0 * math.pi)


def true_to_eccentric_anomaly(true_anomaly: float, eccentricity: float) -> float:
    """Convert true anomaly nu to eccentric anomaly E in radians."""
    if eccentricity < 0.0 or eccentricity >= 1.0:
        raise ValueError(f"Eccentricity must be in [0, 1), got {eccentricity}")

    if eccentricity == 0.0:
        return true_anomaly % (2.0 * math.pi)

    sin_nu = math.sin(true_anomaly)
    cos_nu = math.cos(true_anomaly)
    sqrt_1_minus_e2 = math.sqrt(1.0 - eccentricity**2)

    sin_e = (sqrt_1_minus_e2 * sin_nu) / (1.0 + eccentricity * cos_nu)
    cos_e = (eccentricity + cos_nu) / (1.0 + eccentricity * cos_nu)

    return math.atan2(sin_e, cos_e) % (2.0 * math.pi)


def true_to_mean_anomaly(true_anomaly: float, eccentricity: float) -> float:
    """Convert true anomaly nu to mean anomaly M in radians."""
    e_anom = true_to_eccentric_anomaly(true_anomaly, eccentricity)
    m = e_anom - eccentricity * math.sin(e_anom)
    return m % (2.0 * math.pi)


def propagate_kepler_orbit(semi_major_axis_m: float, eccentricity: float,
                           true_anomaly_0: float, delta_time_s: float,
                           mu: float = EARTH_MU_SI) -> Tuple[float, float, float]:
    """Propagate a Keplerian orbit over delta_time_s.

    Returns:
        (true_anomaly_f, eccentric_anomaly_f, mean_anomaly_f) in radians.
    """
    n = math.sqrt(mu / (semi_major_axis_m**3))
    m0 = true_to_mean_anomaly(true_anomaly_0, eccentricity)
    mf = (m0 + n * delta_time_s) % (2.0 * math.pi)
    ef = mean_to_eccentric_anomaly(mf, eccentricity)
    nuf = eccentric_to_true_anomaly(ef, eccentricity)
    return nuf, ef, mf


# ==============================================================================
# CLOHESSY-WILTSHIRE (HILL-CLOHESSY-WILTSHIRE / HCW) EQUATIONS
# ==============================================================================

def compute_hcw_continuous_matrix(mean_motion: float) -> np.ndarray:
    """Compute continuous-time system matrix A for Clohessy-Wiltshire equations.

    State: [x, y, z, xdot, ydot, zdot]^T in Hill / LVLH frame:
        x: radial (positive outward from Earth center)
        y: along-track (positive in velocity direction)
        z: cross-track (out-of-plane, angular momentum direction)

    Equations of motion:
        x_ddot - 2*n*y_dot - 3*n^2*x = 0
        y_ddot + 2*n*x_dot = 0
        z_ddot + n^2*z = 0
    """
    n = mean_motion
    a = np.zeros((6, 6), dtype=float)
    a[0, 3] = 1.0
    a[1, 4] = 1.0
    a[2, 5] = 1.0
    a[3, 0] = 3.0 * n**2
    a[3, 4] = 2.0 * n
    a[4, 3] = -2.0 * n
    a[5, 2] = -n**2
    return a


def compute_hcw_stm(delta_time_s: float, mean_motion: float) -> np.ndarray:
    """Compute the 6x6 analytical State Transition Matrix (STM) Phi(t) for HCW equations.

    Maps state at t0 to state at t0 + delta_time_s:
        X(t) = Phi(delta_time_s) @ X(0)

    Partitioned form:
        Phi = [[Phi_rr, Phi_rv],
               [Phi_vr, Phi_vv]]
    """
    n = mean_motion
    t = delta_time_s
    nt = n * t
    s = math.sin(nt)
    c = math.cos(nt)

    phi = np.zeros((6, 6), dtype=float)

    # Position from Position (Phi_rr)
    phi[0, 0] = 4.0 - 3.0 * c
    phi[0, 1] = 0.0
    phi[0, 2] = 0.0

    phi[1, 0] = 6.0 * (s - nt)
    phi[1, 1] = 1.0
    phi[1, 2] = 0.0

    phi[2, 0] = 0.0
    phi[2, 1] = 0.0
    phi[2, 2] = c

    # Position from Velocity (Phi_rv)
    phi[0, 3] = s / n
    phi[0, 4] = (2.0 / n) * (1.0 - c)
    phi[0, 5] = 0.0

    phi[1, 3] = -(2.0 / n) * (1.0 - c)
    phi[1, 4] = (4.0 / n) * s - 3.0 * t
    phi[1, 5] = 0.0

    phi[2, 3] = 0.0
    phi[2, 4] = 0.0
    phi[2, 5] = s / n

    # Velocity from Position (Phi_vr)
    phi[3, 0] = 3.0 * n * s
    phi[3, 1] = 0.0
    phi[3, 2] = 0.0

    phi[4, 0] = 6.0 * n * (c - 1.0)
    phi[4, 1] = 0.0
    phi[4, 2] = 0.0

    phi[5, 0] = 0.0
    phi[5, 1] = 0.0
    phi[5, 2] = -n * s

    # Velocity from Velocity (Phi_vv)
    phi[3, 3] = c
    phi[3, 4] = 2.0 * s
    phi[3, 5] = 0.0

    phi[4, 3] = -2.0 * s
    phi[4, 4] = 4.0 * c - 3.0
    phi[4, 5] = 0.0

    phi[5, 3] = 0.0
    phi[5, 4] = 0.0
    phi[5, 5] = c

    return phi


def propagate_hcw(initial_state: np.ndarray, time_span: np.ndarray,
                  mean_motion: float) -> np.ndarray:
    """Propagate relative state vector across array of timestamps using HCW STM.

    Parameters:
        initial_state: 6-element array [x0, y0, z0, xdot0, ydot0, zdot0] in meters and m/s.
        time_span: 1D array of timestamps (seconds) where time_span[0] is initial epoch.
        mean_motion: Chief orbit mean motion n (rad/s).

    Returns:
        (N, 6) trajectory state array in meters and m/s.
    """
    state_0 = np.asarray(initial_state, dtype=float).flatten()
    if state_0.shape != (6,):
        raise ValueError(f"initial_state must have length 6, got shape {state_0.shape}")

    times = np.asarray(time_span, dtype=float).flatten()
    t0 = times[0]
    n_pts = len(times)
    trajectory = np.zeros((n_pts, 6), dtype=float)

    for i, t in enumerate(times):
        dt = t - t0
        phi = compute_hcw_stm(dt, mean_motion)
        trajectory[i, :] = phi @ state_0

    return trajectory


def compute_closed_orbit_velocity(radial_offset_x0: float, mean_motion: float) -> float:
    """Compute along-track velocity ydot0 required for a closed (drift-free) relative orbit.

    The secular along-track drift term in the HCW equations is:
        drift_rate = -(6 * n * x0 + 3 * ydot0)
    For zero secular drift:
        ydot0 = -2 * n * x0
    """
    return -2.0 * mean_motion * radial_offset_x0


def compute_relative_orbit_geometry(initial_state: np.ndarray,
                                    mean_motion: float) -> RelativeOrbitGeometry:
    """Analyze the geometry of a Clohessy-Wiltshire relative orbit.

    Computes center of motion, ellipse semi-axes, out-of-plane amplitude,
    secular drift rate, and whether the orbit is closed.
    """
    x0, y0, z0, xdot0, ydot0, zdot0 = np.asarray(initial_state, dtype=float).flatten()
    n = mean_motion
    period = 2.0 * math.pi / n

    # Secular along-track drift rate (m/s)
    drift_rate = float(-(6.0 * n * x0 + 3.0 * ydot0))
    is_closed = bool(abs(drift_rate) < 1e-9)

    # Center of relative motion
    center_x = float(4.0 * x0 + (2.0 * ydot0 / n))
    center_y = float(y0 - (2.0 * xdot0 / n))

    # In-plane oscillatory amplitude
    c_rad = (3.0 * x0 + 2.0 * ydot0 / n)
    s_rad = xdot0 / n
    radial_amplitude = float(math.sqrt(c_rad**2 + s_rad**2))

    # In a closed HCW orbit, along-track amplitude is exactly 2 * radial amplitude
    along_track_amplitude = float(2.0 * radial_amplitude)

    # Out-of-plane amplitude
    out_of_plane_amplitude = float(math.sqrt(z0**2 + (zdot0 / n)**2))

    return RelativeOrbitGeometry(
        center_x_m=center_x,
        center_y_m=center_y,
        semi_major_axis_m=along_track_amplitude,
        semi_minor_axis_m=radial_amplitude,
        out_of_plane_amplitude_m=out_of_plane_amplitude,
        period_s=float(period),
        secular_drift_rate_m_s=drift_rate,
        is_closed=is_closed,
    )


# ==============================================================================
# YAMANAKA-ANKERSEN (YA) ELLIPTICAL RELATIVE MOTION
# ==============================================================================

def compute_yamanaka_ankersen_psi(true_anomaly: float, eccentricity: float,
                                  time_integral_j: float) -> np.ndarray:
    """Construct the 6x6 normalized fundamental matrix Psi(nu) of Yamanaka & Ankersen (2002).

    Solves the normalized Tschauner-Hempel equations:
        x'' - 2*y' - (3/rho)*x = 0
        y'' + 2*x' = 0
        z'' + z = 0
    where prime denotes d/dnu, and rho = 1 + e*cos(nu).

    Parameters:
        true_anomaly: Target true anomaly nu in radians.
        eccentricity: Reference orbit eccentricity e in [0, 1).
        time_integral_j: Auxiliary integral J = int_nu0^nu dtau / rho(tau)^2.
                         Evaluated via J = n*(t - t0) / (1 - e^2)^(3/2).

    Returns:
        6x6 fundamental matrix Psi(nu).
    """
    nu = true_anomaly
    e = eccentricity
    j = time_integral_j

    s = math.sin(nu)
    c = math.cos(nu)
    rho = 1.0 + e * c

    # Mode 1 (c1): x1 = c*rho, y1 = -s*(2 + e*c)
    x1 = c * rho
    y1 = -s * (2.0 + e * c)
    x1_p = -s * (1.0 + 2.0 * e * c)
    y1_p = e - 2.0 * c * rho

    # Mode 2 (c2): x2 = s*rho, y2 = c*(2 + e*c)
    x2 = s * rho
    y2 = c * (2.0 + e * c)
    x2_p = c + e * (2.0 * c**2 - 1.0)  # c + e*cos(2*nu)
    y2_p = -2.0 * s * rho

    # Mode 3 (c3): Energy / secular mode, x3 = 2 - 3*e*s*rho*J, y3 = -3*rho^2*J
    x3 = 2.0 - 3.0 * e * s * rho * j
    y3 = -3.0 * (rho**2) * j
    x3_p = -3.0 * e * (c + e * (2.0 * c**2 - 1.0)) * j - (3.0 * e * s / rho)
    y3_p = 6.0 * e * s * rho * j - 3.0

    # Mode 4 (c4): Along-track bias, x4 = 0, y4 = 1
    x4 = 0.0
    y4 = 1.0
    x4_p = 0.0
    y4_p = 0.0

    # Mode 5 & 6 (c5, c6): Out-of-plane harmonic oscillator
    z5 = c
    z5_p = -s

    z6 = s
    z6_p = c

    psi = np.zeros((6, 6), dtype=float)
    # Positions
    psi[0, :] = [x1, x2, x3, x4, 0.0, 0.0]
    psi[1, :] = [y1, y2, y3, y4, 0.0, 0.0]
    psi[2, :] = [0.0, 0.0, 0.0, 0.0, z5, z6]

    # Derivatives with respect to true anomaly (d/dnu)
    psi[3, :] = [x1_p, x2_p, x3_p, x4_p, 0.0, 0.0]
    psi[4, :] = [y1_p, y2_p, y3_p, y4_p, 0.0, 0.0]
    psi[5, :] = [0.0, 0.0, 0.0, 0.0, z5_p, z6_p]

    return psi


def compute_pulsating_frame_transform(true_anomaly: float, eccentricity: float,
                                      semi_major_axis_m: float,
                                      mu: float = EARTH_MU_SI) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the kinematic transformation matrices T and T_inv between physical and normalized states.

    Physical state:   X_phys = [x, y, z, xdot, ydot, zdot]^T  (meters, m/s)
    Normalized state: X_norm = [xtilde, ytilde, ztilde, xtilde', ytilde', ztilde']^T

    Transformations:
        X_phys = T @ X_norm
        X_norm = T_inv @ X_phys
    """
    a = semi_major_axis_m
    e = eccentricity
    p = a * (1.0 - e**2)
    c = math.cos(true_anomaly)
    s = math.sin(true_anomaly)
    rho = 1.0 + e * c

    r = p / rho
    sqrt_mu_over_p = math.sqrt(mu / p)
    rdot = sqrt_mu_over_p * e * s
    r_nudot = sqrt_mu_over_p * rho

    # T matrix (6x6)
    t_mat = np.zeros((6, 6), dtype=float)
    t_mat[0:3, 0:3] = r * np.eye(3)
    t_mat[3:6, 0:3] = rdot * np.eye(3)
    t_mat[3:6, 3:6] = r_nudot * np.eye(3)

    # Analytical inverse of T
    t_inv = np.zeros((6, 6), dtype=float)
    t_inv[0:3, 0:3] = (1.0 / r) * np.eye(3)
    t_inv[3:6, 0:3] = (-rdot / (r * r_nudot)) * np.eye(3)
    t_inv[3:6, 3:6] = (1.0 / r_nudot) * np.eye(3)

    return t_mat, t_inv


def compute_yamanaka_ankersen_stm(true_anomaly_0: float, true_anomaly: float,
                                  eccentricity: float, semi_major_axis_m: float,
                                  delta_time_s: Optional[float] = None,
                                  mu: float = EARTH_MU_SI,
                                  normalized: bool = False) -> np.ndarray:
    """Compute the 6x6 Yamanaka-Ankersen State Transition Matrix Phi(nu, nu0).

    Valid for any arbitrary elliptical Keplerian orbit with 0 <= e < 1.
    Smoothly reduces to Clohessy-Wiltshire STM as e -> 0.

    Parameters:
        true_anomaly_0: Initial true anomaly nu0 in radians.
        true_anomaly: Target true anomaly nu in radians.
        eccentricity: Reference orbit eccentricity e in [0, 1).
        semi_major_axis_m: Reference orbit semi-major axis a in meters.
        delta_time_s: Time elapsed in seconds. If None, derived from Keplerian anomalies.
        mu: Gravitational parameter (m^3/s^2).
        normalized: If True, returns STM mapping normalized states [xtilde, xtilde'].
                    If False, returns STM mapping physical states [x (m), xdot (m/s)].
    """
    e = eccentricity
    a = semi_major_axis_m
    nu0 = true_anomaly_0
    nu = true_anomaly

    if e < 0.0 or e >= 1.0:
        raise ValueError(f"Eccentricity must be in [0, 1), got {e}")

    # Compute time elapsed if not provided
    n = math.sqrt(mu / (a**3))
    eta = math.sqrt(1.0 - e**2)

    if delta_time_s is None:
        m0 = true_to_mean_anomaly(nu0, e)
        m = true_to_mean_anomaly(nu, e)
        # Unwrap mean anomaly difference
        delta_m = (m - m0) % (2.0 * math.pi)
        delta_t = delta_m / n
    else:
        delta_t = delta_time_s

    # Yamanaka-Ankersen core J integral: J = int_nu0^nu dtau / rho^2 = n*dt / (1 - e^2)^(3/2)
    j = (n * delta_t) / (eta**3)

    # Initial fundamental matrix at nu0 with J=0
    psi0 = compute_yamanaka_ankersen_psi(nu0, e, 0.0)
    # Current fundamental matrix at nu with elapsed J
    psi = compute_yamanaka_ankersen_psi(nu, e, j)

    # Normalized STM: Phi_norm(nu, nu0) = Psi(nu) @ inv(Psi(nu0))
    # Solve linear system rather than direct matrix inversion for maximum numerical stability
    phi_norm = psi @ np.linalg.inv(psi0)

    if normalized:
        return phi_norm

    # Transform to physical coordinates
    t0, t0_inv = compute_pulsating_frame_transform(nu0, e, a, mu)
    t1, _ = compute_pulsating_frame_transform(nu, e, a, mu)

    phi_phys = t1 @ phi_norm @ t0_inv
    return phi_phys


def propagate_yamanaka_ankersen(initial_physical_state: np.ndarray,
                                time_span: np.ndarray,
                                semi_major_axis_m: float,
                                eccentricity: float,
                                true_anomaly_0: float = 0.0,
                                mu: float = EARTH_MU_SI) -> np.ndarray:
    """Propagate relative physical state across time_span on an elliptical orbit.

    Parameters:
        initial_physical_state: 6-element array [x0, y0, z0, xdot0, ydot0, zdot0] in meters and m/s.
        time_span: 1D array of timestamps (seconds) where time_span[0] is initial epoch.
        semi_major_axis_m: Reference orbit semi-major axis (meters).
        eccentricity: Reference orbit eccentricity in [0, 1).
        true_anomaly_0: True anomaly of target at initial epoch (radians).
        mu: Gravitational parameter (m^3/s^2).

    Returns:
        (N, 6) trajectory physical state array in meters and m/s.
    """
    state_0 = np.asarray(initial_physical_state, dtype=float).flatten()
    if state_0.shape != (6,):
        raise ValueError(f"initial_physical_state must have length 6, got shape {state_0.shape}")

    times = np.asarray(time_span, dtype=float).flatten()
    t0 = times[0]
    n_pts = len(times)
    trajectory = np.zeros((n_pts, 6), dtype=float)

    for i, t in enumerate(times):
        dt = t - t0
        # Compute instantaneous true anomaly
        nu_curr, _, _ = propagate_kepler_orbit(semi_major_axis_m, eccentricity,
                                              true_anomaly_0, dt, mu)
        phi_phys = compute_yamanaka_ankersen_stm(
            true_anomaly_0=true_anomaly_0,
            true_anomaly=nu_curr,
            eccentricity=eccentricity,
            semi_major_axis_m=semi_major_axis_m,
            delta_time_s=dt,
            mu=mu,
            normalized=False,
        )
        trajectory[i, :] = phi_phys @ state_0

    return trajectory


# ==============================================================================
# APPROACH CORRIDOR SAFETY LIMITS & VALIDATION
# ==============================================================================

def check_approach_corridor(relative_state: np.ndarray,
                            corridor_type: Union[str, CorridorType] = CorridorType.V_BAR,
                            cone_half_angle_deg: float = 15.0,
                            glideslope_slope: float = 0.002,
                            min_closing_vel_m_s: float = 0.01,
                            max_closing_vel_m_s: float = 5.0,
                            koz_radius_m: float = 200.0) -> Tuple[bool, List[str]]:
    """Validate instantaneous relative state against approach corridor and safety limits.

    Checks:
        1. Approach cone corridor: lateral offset <= along-axis distance * tan(cone_half_angle)
        2. Glideslope velocity: closing velocity <= glideslope_slope * distance + min_closing_vel
        3. Velocity direction: closing vector directed toward target
        4. Keep-Out Zone (KOZ): distance outside KOZ unless inside corridor cone

    Parameters:
        relative_state: [x, y, z, xdot, ydot, zdot] in meters and m/s.
        corridor_type: V-bar or R-bar approach corridor.
        cone_half_angle_deg: Corridor cone half-angle (degrees).
        glideslope_slope: Velocity scaling slope (1/s), e.g. 0.002 = 2 m/s per 1000 m.
        min_closing_vel_m_s: Minimum permitted closing speed for sensor tracking.
        max_closing_vel_m_s: Hard upper limit on approach speed.
        koz_radius_m: Safety Keep-Out Zone sphere radius in meters.

    Returns:
        (is_safe, list_of_violation_reasons)
    """
    x, y, z, xdot, ydot, zdot = np.asarray(relative_state, dtype=float).flatten()
    tan_alpha = math.tan(math.radians(cone_half_angle_deg))
    distance_total = math.sqrt(x**2 + y**2 + z**2)
    violations: List[str] = []

    c_type = CorridorType(corridor_type)

    if c_type in (CorridorType.V_BAR, CorridorType.V_BAR_POS, CorridorType.V_BAR_NEG):
        # Along-track approach: primary axis is y, lateral is sqrt(x^2 + z^2)
        primary_dist = abs(y)
        lateral_dist = math.sqrt(x**2 + z**2)
        sgn_y = 1.0 if y >= 0.0 else -1.0
        closing_vel = -sgn_y * ydot  # positive if moving toward y=0

        # Check cone confinement
        if primary_dist > 1e-3 and lateral_dist > primary_dist * tan_alpha:
            violations.append(
                f"Cone violation: lateral offset {lateral_dist:.2f} m exceeds "
                f"limit {primary_dist * tan_alpha:.2f} m for y={y:.2f} m"
            )

        # Check velocity direction
        if primary_dist > 1.0 and closing_vel < -1e-3:
            violations.append(
                f"Negative closing velocity: spacecraft drifting away from target ({ydot:.3f} m/s)"
            )

        # Check glideslope closing speed
        allowable_speed = min(max_closing_vel_m_s,
                              glideslope_slope * primary_dist + min_closing_vel_m_s)
        if closing_vel > allowable_speed:
            violations.append(
                f"Glideslope velocity exceedance: closing speed {closing_vel:.3f} m/s "
                f"exceeds allowable {allowable_speed:.3f} m/s"
            )

    elif c_type in (CorridorType.R_BAR, CorridorType.R_BAR_POS, CorridorType.R_BAR_NEG):
        # Radial approach: primary axis is x, lateral is sqrt(y^2 + z^2)
        primary_dist = abs(x)
        lateral_dist = math.sqrt(y**2 + z**2)
        sgn_x = 1.0 if x >= 0.0 else -1.0
        closing_vel = -sgn_x * xdot  # positive if moving toward x=0

        # Check cone confinement
        if primary_dist > 1e-3 and lateral_dist > primary_dist * tan_alpha:
            violations.append(
                f"Cone violation: lateral offset {lateral_dist:.2f} m exceeds "
                f"limit {primary_dist * tan_alpha:.2f} m for x={x:.2f} m"
            )

        # Check velocity direction
        if primary_dist > 1.0 and closing_vel < -1e-3:
            violations.append(
                f"Negative closing velocity: spacecraft drifting away radially ({xdot:.3f} m/s)"
            )

        # Check glideslope closing speed
        allowable_speed = min(max_closing_vel_m_s,
                              glideslope_slope * primary_dist + min_closing_vel_m_s)
        if closing_vel > allowable_speed:
            violations.append(
                f"Glideslope velocity exceedance: closing speed {closing_vel:.3f} m/s "
                f"exceeds allowable {allowable_speed:.3f} m/s"
            )

    # Check Keep-Out Zone (KOZ) violation outside authorized approach cone
    if distance_total < koz_radius_m:
        # Inside KOZ: must strictly satisfy cone corridor
        if violations:
            violations.append(
                f"KOZ breach with corridor violation: total distance {distance_total:.2f} m "
                f"< KOZ radius {koz_radius_m:.2f} m"
            )

    is_safe = len(violations) == 0
    return is_safe, violations


def validate_approach_trajectory(trajectory_states: np.ndarray,
                                 timestamps: np.ndarray,
                                 corridor_type: Union[str, CorridorType] = CorridorType.V_BAR,
                                 cone_half_angle_deg: float = 15.0,
                                 glideslope_slope: float = 0.002,
                                 min_closing_vel_m_s: float = 0.01,
                                 max_closing_vel_m_s: float = 5.0,
                                 koz_radius_m: float = 200.0,
                                 capture_radius_m: float = 5.0) -> ApproachSafetyReport:
    """Validate a complete time-series trajectory against all approach corridor safety limits.

    Returns:
        Structured ApproachSafetyReport detailing all violations, hold points, and safety margins.
    """
    states = np.asarray(trajectory_states, dtype=float)
    times = np.asarray(timestamps, dtype=float)

    if states.ndim != 2 or states.shape[1] != 6:
        raise ValueError(f"trajectory_states must have shape (N, 6), got {states.shape}")
    if len(times) != states.shape[0]:
        raise ValueError("timestamps length must match trajectory_states rows")

    c_type = CorridorType(corridor_type)
    tan_alpha = math.tan(math.radians(cone_half_angle_deg))

    cone_violations: List[CorridorViolation] = []
    vel_violations: List[CorridorViolation] = []
    koz_violations: List[CorridorViolation] = []
    hold_points: List[Dict[str, Any]] = []

    max_lateral_dev = 0.0
    min_dist_to_target = float("inf")

    is_vbar = c_type in (CorridorType.V_BAR, CorridorType.V_BAR_POS, CorridorType.V_BAR_NEG)

    for i in range(len(times)):
        t = times[i]
        state = states[i, :]
        x, y, z, xdot, ydot, zdot = state
        dist = math.sqrt(x**2 + y**2 + z**2)
        speed = math.sqrt(xdot**2 + ydot**2 + zdot**2)

        min_dist_to_target = min(min_dist_to_target, dist)

        # Detect hold points (speed < 0.05 m/s for >= 2 consecutive steps)
        if speed < 0.05 and dist > capture_radius_m:
            if not hold_points or (t - hold_points[-1]["end_time_s"] > 10.0):
                hold_points.append({
                    "start_time_s": t,
                    "end_time_s": t,
                    "distance_m": dist,
                    "state": state.copy(),
                })
            else:
                hold_points[-1]["end_time_s"] = t

        if is_vbar:
            p_dist = abs(y)
            lat_dist = math.sqrt(x**2 + z**2)
            sgn_y = 1.0 if y >= 0.0 else -1.0
            c_vel = -sgn_y * ydot
        else:
            p_dist = abs(x)
            lat_dist = math.sqrt(y**2 + z**2)
            sgn_x = 1.0 if x >= 0.0 else -1.0
            c_vel = -sgn_x * xdot

        max_lateral_dev = max(max_lateral_dev, lat_dist)

        # Check cone (skip if within capture radius)
        if dist > capture_radius_m and p_dist > 1e-3:
            cone_limit = p_dist * tan_alpha
            if lat_dist > cone_limit:
                cone_violations.append(CorridorViolation(
                    timestamp_s=t,
                    state=state,
                    violation_type="CONE",
                    magnitude=lat_dist,
                    threshold=cone_limit,
                    description=f"Lateral offset {lat_dist:.2f}m > limit {cone_limit:.2f}m",
                ))

        # Check glideslope speed
        allowable_speed = min(max_closing_vel_m_s, glideslope_slope * p_dist + min_closing_vel_m_s)
        if dist > capture_radius_m and c_vel > allowable_speed:
            vel_violations.append(CorridorViolation(
                timestamp_s=t,
                state=state,
                violation_type="VELOCITY_LIMIT",
                magnitude=c_vel,
                threshold=allowable_speed,
                description=f"Closing speed {c_vel:.3f}m/s > limit {allowable_speed:.3f}m/s",
            ))

        # Check KOZ breach with cone violation
        if dist < koz_radius_m and lat_dist > (p_dist * tan_alpha):
            koz_violations.append(CorridorViolation(
                timestamp_s=t,
                state=state,
                violation_type="KOZ",
                magnitude=dist,
                threshold=koz_radius_m,
                description=f"Breached KOZ ({dist:.2f}m < {koz_radius_m:.2f}m) outside approach cone",
            ))

    total_violations = len(cone_violations) + len(vel_violations) + len(koz_violations)
    is_safe = total_violations == 0

    summary = (
        f"Approach Trajectory Safety Summary ({c_type.value}):\n"
        f"  Status: {'PASSED (Nominal & Safe)' if is_safe else 'FAILED (Safety Violations Detected)'}\n"
        f"  Total steps evaluated: {len(times)}\n"
        f"  Cone violations: {len(cone_violations)}\n"
        f"  Velocity exceedances: {len(vel_violations)}\n"
        f"  KOZ violations: {len(koz_violations)}\n"
        f"  Maximum lateral deviation: {max_lateral_dev:.3f} m\n"
        f"  Closest approach distance: {min_dist_to_target:.3f} m\n"
        f"  Hold points detected: {len(hold_points)}"
    )

    return ApproachSafetyReport(
        is_safe=is_safe,
        corridor_type=c_type.value,
        cone_violations=cone_violations,
        velocity_violations=vel_violations,
        koz_violations=koz_violations,
        max_lateral_deviation_m=max_lateral_dev,
        min_distance_to_target_m=min_dist_to_target,
        hold_points_detected=hold_points,
        summary=summary,
    )


def check_passive_collision_safety(abort_state: np.ndarray,
                                   time_span: np.ndarray,
                                   mean_motion: float,
                                   koz_radius_m: float = 200.0) -> Tuple[bool, float]:
    """Verify passive collision avoidance (CAM) if propulsion ceases.

    Simulates natural drift trajectory and verifies that the chaser never penetrates
    the Keep-Out Zone (KOZ) during subsequent orbits.

    Returns:
        (is_passively_safe, minimum_miss_distance_m)
    """
    traj = propagate_hcw(abort_state, time_span, mean_motion)
    distances = np.linalg.norm(traj[:, 0:3], axis=1)
    min_dist = float(np.min(distances))
    is_safe = min_dist >= koz_radius_m
    return is_safe, min_dist


# ==============================================================================
# RENDEZVOUS MANEUVER PLANNING (2-IMPULSE & 4-IMPULSE)
# ==============================================================================

def plan_2_impulse_rendezvous(initial_relative_state: np.ndarray,
                              final_relative_state: np.ndarray,
                              transfer_time_s: float,
                              orbit_type: str = "circular",
                              semi_major_axis_m: float = 7000e3,
                              eccentricity: float = 0.0,
                              true_anomaly_0: float = 0.0,
                              mu: float = EARTH_MU_SI,
                              dry_mass_kg: float = 500.0,
                              isp_s: float = 300.0,
                              num_trajectory_steps: int = 100,
                              corridor_type: Optional[Union[str, CorridorType]] = None) -> RendezvousPlan:
    """Plan a 2-impulse relative orbital transfer (Lambert / HCW-YA targeting).

    Solves:
        rf = Phi_rr * r0 + Phi_rv * v0_plus
        => v0_plus = inv(Phi_rv) * (rf - Phi_rr * r0)
        Delta_v1 = v0_plus - v0
        vf_minus = Phi_vr * r0 + Phi_vv * v0_plus
        Delta_v2 = vf - vf_minus

    Parameters:
        initial_relative_state: [x0, y0, z0, xdot0, ydot0, zdot0] in meters and m/s.
        final_relative_state: [xf, yf, zf, xdotf, ydotf, zdotf] in meters and m/s.
        transfer_time_s: Total transfer duration in seconds.
        orbit_type: "circular" (uses HCW) or "elliptical" (uses Yamanaka-Ankersen).
        semi_major_axis_m: Chief semi-major axis (meters).
        eccentricity: Chief orbit eccentricity e.
        true_anomaly_0: Chief true anomaly at departure (radians).
        mu: Gravitational parameter (m^3/s^2).
        dry_mass_kg: Spacecraft dry mass for Tsiolkovsky propellant calculation.
        isp_s: Thruster specific impulse (seconds).
        num_trajectory_steps: Number of intermediate points for trajectory generation.
        corridor_type: Optional corridor to validate ("V-bar" or "R-bar").

    Returns:
        RendezvousPlan dataclass containing delta-Vs, propellant, and trajectory states.
    """
    state_0 = np.asarray(initial_relative_state, dtype=float).flatten()
    state_f = np.asarray(final_relative_state, dtype=float).flatten()

    r0 = state_0[0:3]
    v0 = state_0[3:6]
    rf = state_f[0:3]
    vf = state_f[3:6]

    n = math.sqrt(mu / (semi_major_axis_m**3))
    t_transfer = transfer_time_s

    # Compute State Transition Matrix
    if orbit_type.lower() == "circular" or eccentricity == 0.0:
        phi = compute_hcw_stm(t_transfer, n)
    else:
        nu_f, _, _ = propagate_kepler_orbit(semi_major_axis_m, eccentricity,
                                           true_anomaly_0, t_transfer, mu)
        phi = compute_yamanaka_ankersen_stm(
            true_anomaly_0=true_anomaly_0,
            true_anomaly=nu_f,
            eccentricity=eccentricity,
            semi_major_axis_m=semi_major_axis_m,
            delta_time_s=t_transfer,
            mu=mu,
            normalized=False,
        )

    phi_rr = phi[0:3, 0:3]
    phi_rv = phi[0:3, 3:6]
    phi_vr = phi[3:6, 0:3]
    phi_vv = phi[3:6, 3:6]

    # Check singularity of Phi_rv (e.g. transfer time = integer orbits where sin(nt)=0)
    cond = np.linalg.cond(phi_rv)
    if cond > 1e12 or abs(np.linalg.det(phi_rv)) < 1e-12:
        return RendezvousPlan(
            maneuver_type="2-impulse",
            delta_v_1=np.zeros(3),
            delta_v_2=np.zeros(3),
            total_delta_v=0.0,
            transfer_time_s=t_transfer,
            success=False,
            message=f"Singular transfer geometry: Phi_rv is ill-conditioned (cond={cond:.2e}). "
                    f"Avoid transfer times near full orbital periods.",
        )

    # Solve for required initial velocity v0_plus
    # rf = phi_rr @ r0 + phi_rv @ v0_plus
    target_pos_residual = rf - (phi_rr @ r0)
    v0_plus = np.linalg.solve(phi_rv, target_pos_residual)

    # First burn (departure)
    delta_v_1 = v0_plus - v0

    # Arrival velocity before second burn
    vf_minus = (phi_vr @ r0) + (phi_vv @ v0_plus)

    # Second burn (arrival matching)
    delta_v_2 = vf - vf_minus

    mag_dv1 = float(np.linalg.norm(delta_v_1))
    mag_dv2 = float(np.linalg.norm(delta_v_2))
    total_dv = mag_dv1 + mag_dv2

    # Tsiolkovsky propellant calculation: m_prop = m_wet - m_dry = m_dry * (exp(dv / (Isp * g0)) - 1)
    g0 = STANDARD_GRAVITY
    ve = isp_s * g0
    mass_ratio = math.exp(total_dv / ve)
    propellant_used = dry_mass_kg * (mass_ratio - 1.0)

    # Generate trajectory time-series
    time_pts = np.linspace(0.0, t_transfer, num_trajectory_steps)
    post_burn_state_0 = np.concatenate([r0, v0_plus])

    if orbit_type.lower() == "circular" or eccentricity == 0.0:
        traj_states = propagate_hcw(post_burn_state_0, time_pts, n)
    else:
        traj_states = propagate_yamanaka_ankersen(
            post_burn_state_0, time_pts, semi_major_axis_m, eccentricity, true_anomaly_0, mu
        )

    # Ensure exact final state matches
    traj_states[-1, 0:3] = rf
    traj_states[-1, 3:6] = vf

    # Validate approach safety corridor if requested
    safety_rep = None
    if corridor_type is not None:
        safety_rep = validate_approach_trajectory(
            traj_states, time_pts, corridor_type=corridor_type
        )

    return RendezvousPlan(
        maneuver_type="2-impulse",
        delta_v_1=delta_v_1,
        delta_v_2=delta_v_2,
        total_delta_v=total_dv,
        transfer_time_s=t_transfer,
        leg_times_s=[t_transfer],
        propellant_used_kg=propellant_used,
        trajectory_times=time_pts,
        trajectory_states=traj_states,
        safety_report=safety_rep,
        success=True,
        message="2-impulse rendezvous plan generated successfully.",
    )


def plan_4_impulse_rendezvous(initial_relative_state: np.ndarray,
                              final_relative_state: np.ndarray,
                              leg1_transfer_time_s: float,
                              leg2_transfer_time_s: float,
                              waypoint_relative_state: Optional[np.ndarray] = None,
                              hold_duration_s: float = 0.0,
                              orbit_type: str = "circular",
                              semi_major_axis_m: float = 7000e3,
                              eccentricity: float = 0.0,
                              true_anomaly_0: float = 0.0,
                              mu: float = EARTH_MU_SI,
                              dry_mass_kg: float = 500.0,
                              isp_s: float = 300.0,
                              num_trajectory_steps_per_leg: int = 60,
                              corridor_type: Optional[Union[str, CorridorType]] = None) -> RendezvousPlan:
    """Plan a 4-impulse rendezvous trajectory via an intermediate waypoint / hold-point.

    Ideal for ADR proximity operations:
        Leg 1 (Burns 1 & 2): Coarse transfer to V-bar stationkeeping hold point (e.g. S1 at 1000m)
        Stationkeeping hold: Chaser holds position for sensor calibration / debris tracking
        Leg 2 (Burns 3 & 4): Glideslope approach to final capture point (e.g. 5m)

    If waypoint_relative_state is None, automatically synthesizes a safe waypoint
    along the approach corridor at 50% along-axis distance.
    """
    state_0 = np.asarray(initial_relative_state, dtype=float).flatten()
    state_f = np.asarray(final_relative_state, dtype=float).flatten()

    # Automatic waypoint synthesis if not specified
    if waypoint_relative_state is None:
        # Midpoint in along-axis, zero lateral offset, zero velocity (stationkeeping)
        r_mid = 0.5 * (state_0[0:3] + state_f[0:3])
        waypoint_state = np.zeros(6, dtype=float)
        waypoint_state[0:3] = r_mid
        # Closed orbit velocity for hold point
        n = math.sqrt(mu / (semi_major_axis_m**3))
        waypoint_state[4] = compute_closed_orbit_velocity(r_mid[0], n)
    else:
        waypoint_state = np.asarray(waypoint_relative_state, dtype=float).flatten()

    # Plan Leg 1: state_0 -> waypoint_state
    plan_leg1 = plan_2_impulse_rendezvous(
        initial_relative_state=state_0,
        final_relative_state=waypoint_state,
        transfer_time_s=leg1_transfer_time_s,
        orbit_type=orbit_type,
        semi_major_axis_m=semi_major_axis_m,
        eccentricity=eccentricity,
        true_anomaly_0=true_anomaly_0,
        mu=mu,
        dry_mass_kg=dry_mass_kg,
        isp_s=isp_s,
        num_trajectory_steps=num_trajectory_steps_per_leg,
    )

    if not plan_leg1.success:
        return RendezvousPlan(
            maneuver_type="4-impulse",
            delta_v_1=np.zeros(3),
            delta_v_2=np.zeros(3),
            total_delta_v=0.0,
            success=False,
            message=f"Leg 1 planning failed: {plan_leg1.message}",
        )

    # Anomaly after Leg 1 + hold
    nu_wp, _, _ = propagate_kepler_orbit(
        semi_major_axis_m, eccentricity, true_anomaly_0,
        leg1_transfer_time_s + hold_duration_s, mu
    )

    # Plan Leg 2: waypoint_state -> state_f
    plan_leg2 = plan_2_impulse_rendezvous(
        initial_relative_state=waypoint_state,
        final_relative_state=state_f,
        transfer_time_s=leg2_transfer_time_s,
        orbit_type=orbit_type,
        semi_major_axis_m=semi_major_axis_m,
        eccentricity=eccentricity,
        true_anomaly_0=nu_wp,
        mu=mu,
        dry_mass_kg=dry_mass_kg - plan_leg1.propellant_used_kg,
        isp_s=isp_s,
        num_trajectory_steps=num_trajectory_steps_per_leg,
    )

    if not plan_leg2.success:
        return RendezvousPlan(
            maneuver_type="4-impulse",
            delta_v_1=plan_leg1.delta_v_1,
            delta_v_2=plan_leg1.delta_v_2,
            total_delta_v=plan_leg1.total_delta_v,
            success=False,
            message=f"Leg 2 planning failed: {plan_leg2.message}",
        )

    # Extract the 4 impulses
    dv1 = plan_leg1.delta_v_1
    dv2 = plan_leg1.delta_v_2
    dv3 = plan_leg2.delta_v_1
    dv4 = plan_leg2.delta_v_2

    total_dv = (float(np.linalg.norm(dv1)) + float(np.linalg.norm(dv2)) +
                float(np.linalg.norm(dv3)) + float(np.linalg.norm(dv4)))
    total_transfer_time = leg1_transfer_time_s + hold_duration_s + leg2_transfer_time_s

    # Propellant consumption
    g0 = STANDARD_GRAVITY
    ve = isp_s * g0
    mass_ratio = math.exp(total_dv / ve)
    total_propellant = dry_mass_kg * (mass_ratio - 1.0)

    # Stitch trajectory histories together
    t1_arr = plan_leg1.trajectory_times
    s1_arr = plan_leg1.trajectory_states

    if hold_duration_s > 0.0:
        # Add hold phase
        n_hold_pts = 10
        t_hold_arr = np.linspace(leg1_transfer_time_s,
                                 leg1_transfer_time_s + hold_duration_s,
                                 n_hold_pts)[1:]
        s_hold_arr = np.tile(waypoint_state, (len(t_hold_arr), 1))
    else:
        t_hold_arr = np.array([])
        s_hold_arr = np.zeros((0, 6))

    t2_offset = leg1_transfer_time_s + hold_duration_s
    t2_arr = plan_leg2.trajectory_times[1:] + t2_offset
    s2_arr = plan_leg2.trajectory_states[1:, :]

    all_times = np.concatenate([t1_arr, t_hold_arr, t2_arr])
    all_states = np.vstack([s1_arr, s_hold_arr, s2_arr])

    # Validate corridor safety if requested
    safety_rep = None
    if corridor_type is not None:
        safety_rep = validate_approach_trajectory(
            all_states, all_times, corridor_type=corridor_type
        )

    return RendezvousPlan(
        maneuver_type="4-impulse",
        delta_v_1=dv1,
        delta_v_2=dv2,
        delta_v_3=dv3,
        delta_v_4=dv4,
        total_delta_v=total_dv,
        transfer_time_s=total_transfer_time,
        leg_times_s=[leg1_transfer_time_s, hold_duration_s, leg2_transfer_time_s],
        propellant_used_kg=total_propellant,
        trajectory_times=all_times,
        trajectory_states=all_states,
        safety_report=safety_rep,
        success=True,
        message="4-impulse rendezvous plan generated successfully with waypoint hold.",
    )


# ==============================================================================
# OBJECT-ORIENTED PROPAGATORS & VALIDATORS
# ==============================================================================

class ClohessyWiltshirePropagator:
    """High-level class for Clohessy-Wiltshire relative orbit propagation and targeting."""

    def __init__(self, semi_major_axis_m: float = 7000e3, mu: float = EARTH_MU_SI):
        self.semi_major_axis_m = float(semi_major_axis_m)
        self.mu = float(mu)
        self.mean_motion = math.sqrt(self.mu / (self.semi_major_axis_m**3))
        self.period_s = 2.0 * math.pi / self.mean_motion

    def stm(self, delta_time_s: float) -> np.ndarray:
        """Compute the 6x6 HCW State Transition Matrix."""
        return compute_hcw_stm(delta_time_s, self.mean_motion)

    def propagate(self, initial_state: np.ndarray, time_span: np.ndarray) -> np.ndarray:
        """Propagate state across time_span."""
        return propagate_hcw(initial_state, time_span, self.mean_motion)

    def geometry(self, state: np.ndarray) -> RelativeOrbitGeometry:
        """Analyze relative orbit geometry."""
        return compute_relative_orbit_geometry(state, self.mean_motion)

    def closed_orbit_velocity(self, x0: float) -> float:
        """Compute along-track velocity for closed relative orbit."""
        return compute_closed_orbit_velocity(x0, self.mean_motion)

    def plan_2_impulse(self, state_0: np.ndarray, state_f: np.ndarray,
                       transfer_time_s: float, **kwargs) -> RendezvousPlan:
        """Plan a 2-impulse rendezvous trajectory."""
        return plan_2_impulse_rendezvous(
            initial_relative_state=state_0,
            final_relative_state=state_f,
            transfer_time_s=transfer_time_s,
            orbit_type="circular",
            semi_major_axis_m=self.semi_major_axis_m,
            mu=self.mu,
            **kwargs
        )


class YamanakaAnkersenPropagator:
    """High-level class for Yamanaka-Ankersen elliptical relative orbit propagation and targeting."""

    def __init__(self, semi_major_axis_m: float = 7000e3, eccentricity: float = 0.05,
                 true_anomaly_0: float = 0.0, mu: float = EARTH_MU_SI):
        if eccentricity < 0.0 or eccentricity >= 1.0:
            raise ValueError(f"Eccentricity must be in [0, 1), got {eccentricity}")
        self.semi_major_axis_m = float(semi_major_axis_m)
        self.eccentricity = float(eccentricity)
        self.true_anomaly_0 = float(true_anomaly_0)
        self.mu = float(mu)
        self.mean_motion = math.sqrt(self.mu / (self.semi_major_axis_m**3))
        self.period_s = 2.0 * math.pi / self.mean_motion

    def stm(self, true_anomaly: float, delta_time_s: Optional[float] = None,
            normalized: bool = False) -> np.ndarray:
        """Compute the 6x6 YA State Transition Matrix from true_anomaly_0 to true_anomaly."""
        return compute_yamanaka_ankersen_stm(
            true_anomaly_0=self.true_anomaly_0,
            true_anomaly=true_anomaly,
            eccentricity=self.eccentricity,
            semi_major_axis_m=self.semi_major_axis_m,
            delta_time_s=delta_time_s,
            mu=self.mu,
            normalized=normalized,
        )

    def propagate(self, initial_physical_state: np.ndarray,
                  time_span: np.ndarray) -> np.ndarray:
        """Propagate physical state across time_span."""
        return propagate_yamanaka_ankersen(
            initial_physical_state=initial_physical_state,
            time_span=time_span,
            semi_major_axis_m=self.semi_major_axis_m,
            eccentricity=self.eccentricity,
            true_anomaly_0=self.true_anomaly_0,
            mu=self.mu,
        )

    def plan_2_impulse(self, state_0: np.ndarray, state_f: np.ndarray,
                       transfer_time_s: float, **kwargs) -> RendezvousPlan:
        """Plan a 2-impulse rendezvous trajectory on elliptical orbit."""
        return plan_2_impulse_rendezvous(
            initial_relative_state=state_0,
            final_relative_state=state_f,
            transfer_time_s=transfer_time_s,
            orbit_type="elliptical",
            semi_major_axis_m=self.semi_major_axis_m,
            eccentricity=self.eccentricity,
            true_anomaly_0=self.true_anomaly_0,
            mu=self.mu,
            **kwargs
        )


class ApproachSafetyValidator:
    """Configurable validator for V-bar / R-bar approach corridor safety and Keep-Out Zones."""

    def __init__(self, corridor_type: Union[str, CorridorType] = CorridorType.V_BAR,
                 cone_half_angle_deg: float = 15.0,
                 glideslope_slope: float = 0.002,
                 min_closing_vel_m_s: float = 0.01,
                 max_closing_vel_m_s: float = 5.0,
                 koz_radius_m: float = 200.0,
                 capture_radius_m: float = 5.0):
        self.corridor_type = CorridorType(corridor_type)
        self.cone_half_angle_deg = float(cone_half_angle_deg)
        self.glideslope_slope = float(glideslope_slope)
        self.min_closing_vel_m_s = float(min_closing_vel_m_s)
        self.max_closing_vel_m_s = float(max_closing_vel_m_s)
        self.koz_radius_m = float(koz_radius_m)
        self.capture_radius_m = float(capture_radius_m)

    def check_state(self, state: np.ndarray) -> Tuple[bool, List[str]]:
        """Validate an instantaneous relative state."""
        return check_approach_corridor(
            relative_state=state,
            corridor_type=self.corridor_type,
            cone_half_angle_deg=self.cone_half_angle_deg,
            glideslope_slope=self.glideslope_slope,
            min_closing_vel_m_s=self.min_closing_vel_m_s,
            max_closing_vel_m_s=self.max_closing_vel_m_s,
            koz_radius_m=self.koz_radius_m,
        )

    def validate_trajectory(self, trajectory_states: np.ndarray,
                            timestamps: np.ndarray) -> ApproachSafetyReport:
        """Validate an entire trajectory history."""
        return validate_approach_trajectory(
            trajectory_states=trajectory_states,
            timestamps=timestamps,
            corridor_type=self.corridor_type,
            cone_half_angle_deg=self.cone_half_angle_deg,
            glideslope_slope=self.glideslope_slope,
            min_closing_vel_m_s=self.min_closing_vel_m_s,
            max_closing_vel_m_s=self.max_closing_vel_m_s,
            koz_radius_m=self.koz_radius_m,
            capture_radius_m=self.capture_radius_m,
        )
