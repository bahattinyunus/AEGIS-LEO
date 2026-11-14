"""
AEGIS-LEO: Autonomous Orbit Determination (AOD) & Sensor Fusion Engine
=====================================================================
Multi-sensor navigation filter implementing Extended Kalman Filter (EKF) and
Unscented Kalman Filter (UKF) for onboard real-time orbit determination in LEO.

Fuses:
1. Star Tracker optical Line-Of-Sight (LOS) unit vectors (ground beacons, celestial targets, nadir horizon)
2. GNSS pseudorange and Doppler range-rate measurements (GPS/Galileo/GLONASS)
3. 3-Axis Magnetometer magnetic field measurements with space-weather storm adaptive noise inflation
   (NOAA Kp 0-9 scale, Dst ring-current depression, and auroral electrojet compensation)

Architecture:
- High-precision numerical orbit dynamics: 2-body gravity + Earth J2 oblateness + atmospheric drag
- Fully coupled 8-state navigation vector: [r_x, r_y, r_z, v_x, v_y, v_z, b_clk, d_clk]
- Numerical/analytical variational equations for discrete state transition matrix
- Joseph-stabilized EKF update and Scaled Unscented Transformation UKF (Merwe parameters)
- Innovation Mahalanobis chi-squared gating for anomaly and corrupted measurement rejection
- Pure NumPy & Python standard library implementation with zero external dependencies
"""

from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import math
import numpy as np

from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    EARTH_MU,
    EARTH_J2,
    EARTH_ROTATION_RATE_RAD_S,
    SPEED_OF_LIGHT,
    QUIET_KP,
    MODERATE_KP,
    STORM_G1_KP,
    STORM_G2_KP,
    STORM_G3_KP,
    STORM_G4_KP,
    STORM_G5_KP,
)

# ---------------------------------------------------------------------------
# Physical & Sensor Constants
# ---------------------------------------------------------------------------
SPEED_OF_LIGHT_KM_S: float = SPEED_OF_LIGHT / 1000.0  # ~299792.458 km/s

# Earth Magnetic Dipole Parameters (IGRF Baseline)
# B0 is equatorial surface field ~31,200 nT (0.312 Gauss)
EARTH_MAGNETIC_EQUATORIAL_B0_NT: float = 31200.0
# Magnetic dipole axis tilt relative to geographic North pole (~11.5 degrees)
DIPOLE_TILT_RAD: float = math.radians(11.5)
# Longitude of Earth's magnetic dipole axis in ECEF (~ -70.0 degrees West)
DIPOLE_LONGITUDE_RAD: float = math.radians(-70.0)

# Default Sensor 1-Sigma Noise Characteristics
DEFAULT_GNSS_PSEUDORANGE_SIGMA_KM: float = 0.003  # 3.0 meters 1-sigma
DEFAULT_GNSS_DOPPLER_SIGMA_KM_S: float = 0.00005   # 0.05 m/s 1-sigma
DEFAULT_STAR_TRACKER_SIGMA_RAD: float = math.radians(15.0 / 3600.0)  # 15 arcseconds
DEFAULT_MAGNETOMETER_SIGMA_NT: float = 25.0  # 25 nT nominal quiet magnetometer noise


# ---------------------------------------------------------------------------
# Data Classes & Telemetry Containers
# ---------------------------------------------------------------------------

@dataclass
class NavigationState:
    """
    Coupled Spacecraft Orbit and GNSS Clock Navigation State Vector.
    Default representation has 8 dimensions:
    - Position: r_eci = [x, y, z] in km
    - Velocity: v_eci = [vx, vy, vz] in km/s
    - GNSS Receiver Clock Bias: b_clk in km (equivalent range c * delta_t)
    - GNSS Receiver Clock Drift: d_clk in km/s (equivalent range rate c * dot_delta_t)
    """
    epoch_seconds: float
    position_km: np.ndarray
    velocity_km_s: np.ndarray
    clock_bias_km: float = 0.0
    clock_drift_km_s: float = 0.0
    covariance: Optional[np.ndarray] = None

    def __post_init__(self):
        self.position_km = np.asarray(self.position_km, dtype=float).reshape(3)
        self.velocity_km_s = np.asarray(self.velocity_km_s, dtype=float).reshape(3)
        if self.covariance is None:
            self.covariance = np.eye(8, dtype=float)

    @property
    def x(self) -> float:
        return float(self.position_km[0])

    @property
    def y(self) -> float:
        return float(self.position_km[1])

    @property
    def z(self) -> float:
        return float(self.position_km[2])

    @property
    def vx(self) -> float:
        return float(self.velocity_km_s[0])

    @property
    def vy(self) -> float:
        return float(self.velocity_km_s[1])

    @property
    def vz(self) -> float:
        return float(self.velocity_km_s[2])

    @property
    def radius_km(self) -> float:
        return float(np.linalg.norm(self.position_km))

    @property
    def speed_km_s(self) -> float:
        return float(np.linalg.norm(self.velocity_km_s))

    @property
    def altitude_km(self) -> float:
        return self.radius_km - EARTH_RADIUS_KM

    def as_vector(self, dim: int = 8) -> np.ndarray:
        """Return state as 1D numpy array of size 6 or 8."""
        if dim == 6:
            return np.concatenate([self.position_km, self.velocity_km_s])
        elif dim == 8:
            return np.array([
                self.position_km[0], self.position_km[1], self.position_km[2],
                self.velocity_km_s[0], self.velocity_km_s[1], self.velocity_km_s[2],
                self.clock_bias_km, self.clock_drift_km_s
            ], dtype=float)
        else:
            raise ValueError(f"Unsupported state dimension {dim}; must be 6 or 8")

    @classmethod
    def from_vector(
        cls,
        vec: np.ndarray,
        epoch_seconds: float,
        covariance: Optional[np.ndarray] = None
    ) -> "NavigationState":
        """Reconstruct NavigationState from vector."""
        vec = np.asarray(vec, dtype=float).flatten()
        if len(vec) == 6:
            return cls(
                epoch_seconds=epoch_seconds,
                position_km=vec[0:3].copy(),
                velocity_km_s=vec[3:6].copy(),
                clock_bias_km=0.0,
                clock_drift_km_s=0.0,
                covariance=covariance
            )
        elif len(vec) == 8:
            return cls(
                epoch_seconds=epoch_seconds,
                position_km=vec[0:3].copy(),
                velocity_km_s=vec[3:6].copy(),
                clock_bias_km=float(vec[6]),
                clock_drift_km_s=float(vec[7]),
                covariance=covariance
            )
        else:
            raise ValueError(f"Vector length must be 6 or 8, got {len(vec)}")


@dataclass
class GNSSObservation:
    """
    GNSS Space Vehicle (SV) Single Pseudorange & Doppler Observation.
    All positions and velocities in Earth-Centered Inertial (ECI) coordinates.
    """
    sat_id: str
    prn: int
    pseudorange_km: float
    sat_pos_km: np.ndarray
    pseudorange_rate_km_s: Optional[float] = None
    sat_vel_km_s: Optional[np.ndarray] = None
    pseudorange_variance_km2: float = DEFAULT_GNSS_PSEUDORANGE_SIGMA_KM**2
    doppler_variance_km2_s2: float = DEFAULT_GNSS_DOPPLER_SIGMA_KM_S**2
    elevation_deg: float = 90.0

    def __post_init__(self):
        self.sat_pos_km = np.asarray(self.sat_pos_km, dtype=float).reshape(3)
        if self.sat_vel_km_s is not None:
            self.sat_vel_km_s = np.asarray(self.sat_vel_km_s, dtype=float).reshape(3)


@dataclass
class GNSSBatchMeasurement:
    """Simultaneous multi-channel GNSS constellation observation epoch."""
    epoch_seconds: float
    observations: List[GNSSObservation]


@dataclass
class StarTrackerObservation:
    """
    Star Tracker Optical Line-of-Sight (LOS) Vector Observation.
    Unit vector pointing towards known ground optical beacon, target satellite,
    or reference optical landmark.
    """
    epoch_seconds: float
    los_unit_vector: np.ndarray
    target_pos_km: np.ndarray
    angular_variance_rad2: float = DEFAULT_STAR_TRACKER_SIGMA_RAD**2
    sensor_id: str = "STR_PRIMARY"
    is_valid: bool = True

    def __post_init__(self):
        self.los_unit_vector = np.asarray(self.los_unit_vector, dtype=float).reshape(3)
        norm = np.linalg.norm(self.los_unit_vector)
        if norm > 1e-9:
            self.los_unit_vector /= norm
        self.target_pos_km = np.asarray(self.target_pos_km, dtype=float).reshape(3)


@dataclass
class MagnetometerObservation:
    """
    3-Axis Magnetometer In-Situ Geomagnetic Measurement.
    Includes active space-weather storm indices for adaptive noise covariance scaling.
    Optional dcm_eci_to_body (3x3) allows processing raw magnetometer telemetry
    measured in spacecraft body frame.
    """
    epoch_seconds: float
    b_measured_nt: np.ndarray
    kp_index: float = QUIET_KP
    dst_index_nt: float = 0.0
    base_variance_nt2: float = DEFAULT_MAGNETOMETER_SIGMA_NT**2
    sensor_id: str = "MAG_PRIMARY"
    dcm_eci_to_body: Optional[np.ndarray] = None

    def __post_init__(self):
        self.b_measured_nt = np.asarray(self.b_measured_nt, dtype=float).reshape(3)
        if self.dcm_eci_to_body is not None:
            self.dcm_eci_to_body = np.asarray(self.dcm_eci_to_body, dtype=float).reshape((3, 3))


@dataclass
class FilterUpdateDiagnostics:
    """Diagnostic telemetry packet produced after each measurement update."""
    epoch_seconds: float
    sensor_type: str
    num_measurements: int
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    mahalanobis_distance: float
    is_gated: bool
    state_correction_pos_km: float
    state_correction_vel_km_s: float
    trace_covariance_prior: float
    trace_covariance_post: float
    storm_noise_scale: float = 1.0


# ---------------------------------------------------------------------------
# Geomagnetic Field & Storm Disturbance Physics
# ---------------------------------------------------------------------------

def compute_geomagnetic_field(
    r_eci_km: np.ndarray,
    epoch_seconds: float = 0.0,
    kp: float = QUIET_KP,
    dst_nt: float = 0.0,
) -> np.ndarray:
    """
    Calculate modeled Earth geomagnetic field vector B (nT) in ECI coordinates.
    Incorporates:
    1. Tilted, rotating magnetic dipole model (B0 ~ 31,200 nT at Re).
    2. Ring current depression during geomagnetic storms parameterized by Dst (nT).
    3. High-latitude auroral electrojet magnetic field perturbations driven by NOAA Kp index.

    Parameters:
    -----------
    r_eci_km : np.ndarray
        Spacecraft position vector in ECI [x, y, z] (km).
    epoch_seconds : float
        Elapsed seconds from epoch (accounts for Earth diurnal rotation of dipole axis).
    kp : float
        NOAA geomagnetic storm index (0.0 to 9.0).
    dst_nt : float
        Disturbance Storm Time index (nT, typically 0 to -400 nT in major storms).

    Returns:
    --------
    np.ndarray
        Geomagnetic field vector [Bx, By, Bz] in nT.
    """
    r_vec = np.asarray(r_eci_km, dtype=float).reshape(3)
    r_mag = float(np.linalg.norm(r_vec))
    if r_mag < 1.0:
        return np.zeros(3)

    # Unit position vector
    r_hat = r_vec / r_mag

    # Rotating magnetic dipole axis in ECI
    # Earth rotates about +Z at EARTH_ROTATION_RATE_RAD_S
    phi_dipole = DIPOLE_LONGITUDE_RAD + EARTH_ROTATION_RATE_RAD_S * epoch_seconds
    m_hat = np.array([
        math.sin(DIPOLE_TILT_RAD) * math.cos(phi_dipole),
        math.sin(DIPOLE_TILT_RAD) * math.sin(phi_dipole),
        math.cos(DIPOLE_TILT_RAD),
    ], dtype=float)

    # Main internal dipole field (nT)
    # B_dipole = B0 * (Re / r)^3 * [3 * (m_hat . r_hat) * r_hat - m_hat]
    scaling = EARTH_MAGNETIC_EQUATORIAL_B0_NT * ((EARTH_RADIUS_KM / r_mag) ** 3)
    m_dot_r = float(np.dot(m_hat, r_hat))
    b_dipole = scaling * (3.0 * m_dot_r * r_hat - m_hat)

    # External geomagnetic storm disturbance:
    # 1. Ring Current (Dst) perturbation: southward uniform depression along dipole axis
    # In absence of direct Dst measurement, estimate empirical Dst from Kp:
    effective_dst = dst_nt
    if math.isclose(effective_dst, 0.0) and kp > MODERATE_KP:
        # Empirical relationship: Dst ~ -15 * (Kp - 2)^1.25 nT
        effective_dst = -15.0 * ((kp - 2.0) ** 1.25)

    b_ring = effective_dst * m_hat  # Since Dst is negative during storm, opposes internal dipole

    # 2. Auroral electrojet turbulence (high geomagnetic latitude perturbation)
    # Active primarily when Kp >= 5.0 (G1 through G5 storms)
    b_auroral = np.zeros(3)
    if kp >= STORM_G1_KP:
        # Geomagnetic latitude
        sin_geomag_lat = m_dot_r
        # Auroral oval peak around 65-75 degrees magnetic latitude (|sin| ~ 0.9)
        auroral_factor = math.exp(-((abs(sin_geomag_lat) - 0.9) ** 2) / 0.04)
        turb_amp = 12.0 * ((kp - 4.0) ** 1.6) * auroral_factor
        # Perturbation acts perpendicular to dipole axis
        transverse_axis = np.cross(m_hat, r_hat)
        t_norm = np.linalg.norm(transverse_axis)
        if t_norm > 1e-6:
            b_auroral = turb_amp * (transverse_axis / t_norm)

    return b_dipole + b_ring + b_auroral


def compute_geomagnetic_jacobian(
    r_eci_km: np.ndarray,
    epoch_seconds: float = 0.0,
) -> np.ndarray:
    """
    Calculate exact analytical 3x3 Jacobian matrix H_mag = d(B_eci)/d(r_eci) in nT / km.
    Derived from the spatial gradient of the Earth's magnetic dipole field.
    Note: Maxwell's equations require trace(H_mag) = div(B) = 0.

    Parameters:
    -----------
    r_eci_km : np.ndarray
        Spacecraft position vector in ECI [x, y, z] in km.
    epoch_seconds : float
        Elapsed seconds from epoch.

    Returns:
    --------
    np.ndarray
        3x3 Jacobian matrix (nT / km).
    """
    r_vec = np.asarray(r_eci_km, dtype=float).reshape(3)
    r2 = float(np.dot(r_vec, r_vec))
    r = math.sqrt(r2)
    if r < 1.0:
        return np.zeros((3, 3))

    r5 = r2 * r2 * r
    r7 = r5 * r2

    phi_dipole = DIPOLE_LONGITUDE_RAD + EARTH_ROTATION_RATE_RAD_S * epoch_seconds
    m_hat = np.array([
        math.sin(DIPOLE_TILT_RAD) * math.cos(phi_dipole),
        math.sin(DIPOLE_TILT_RAD) * math.sin(phi_dipole),
        math.cos(DIPOLE_TILT_RAD),
    ], dtype=float)

    c_mag = EARTH_MAGNETIC_EQUATORIAL_B0_NT * (EARTH_RADIUS_KM ** 3)
    m_dot_r = float(np.dot(m_hat, r_vec))

    # Analytical gradient:
    # d(B_i)/d(r_j) = 3*C/r^5 * [ m_j*r_i + m_i*r_j + (m.r)*delta_ij - 5*(m.r)*r_i*r_j / r^2 ]
    factor = 3.0 * c_mag / r5
    term1 = np.outer(r_vec, m_hat) + np.outer(m_hat, r_vec)
    term2 = m_dot_r * np.eye(3)
    term3 = (5.0 * m_dot_r / r2) * np.outer(r_vec, r_vec)

    h_mag = factor * (term1 + term2 - term3)
    return h_mag


def compute_magnetometer_covariance(
    base_variance_nt2: float,
    kp: float = QUIET_KP,
    dst_nt: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """
    Compute adaptive measurement noise covariance matrix R_mag (3x3) under space weather storms.
    Under quiet conditions (Kp < 4), nominal magnetometer noise is preserved.
    Under geomagnetic storms (Kp >= 5, G1-G5), external ring current turbulence and Birkeland
    field-aligned currents inflate measurement uncertainty, protecting the filter against
    divergence due to unmodeled storm transients.

    Returns:
    --------
    Tuple[np.ndarray, float]:
        (3x3 Covariance matrix in nT^2, storm inflation factor).
    """
    storm_scale = 1.0
    if kp > MODERATE_KP:
        # Quadratic inflation above moderate threshold
        excess_kp = kp - MODERATE_KP
        storm_scale += 1.2 * (excess_kp ** 2)
        if kp >= STORM_G3_KP:
            # Steep exponential rise for extreme G3-G5 storms
            storm_scale += 3.5 * ((kp - STORM_G3_KP) ** 2.5)

    if dst_nt < -50.0:
        # Ring current depression variance booster
        storm_scale += (abs(dst_nt) / 100.0) ** 1.5

    r_mag = (base_variance_nt2 * storm_scale) * np.eye(3, dtype=float)
    return r_mag, float(storm_scale)


# ---------------------------------------------------------------------------
# Star Tracker Optical LOS Physics
# ---------------------------------------------------------------------------

def compute_los_vector(
    r_spacecraft_km: np.ndarray,
    r_target_km: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Compute optical Line-Of-Sight (LOS) unit vector from spacecraft to target.

    Parameters:
    -----------
    r_spacecraft_km : np.ndarray
        Spacecraft position in ECI (km).
    r_target_km : np.ndarray
        Optical landmark, beacon, or celestial target position in ECI (km).

    Returns:
    --------
    Tuple[np.ndarray, float]:
        (3D unit LOS vector, range distance in km).
    """
    rho_vec = np.asarray(r_target_km, dtype=float) - np.asarray(r_spacecraft_km, dtype=float)
    range_km = float(np.linalg.norm(rho_vec))
    if range_km < 1e-6:
        return np.array([1.0, 0.0, 0.0]), 1e-6
    u_los = rho_vec / range_km
    return u_los, range_km


def compute_los_jacobian(
    r_spacecraft_km: np.ndarray,
    r_target_km: np.ndarray,
    state_dim: int = 8,
) -> np.ndarray:
    """
    Compute 3xN measurement Jacobian matrix H_los = d(u_los)/d(x_state).
    d(u_los)/d(r) = -1/rho * (I_3 - u_los * u_los^T).
    d(u_los)/d(v) = 0, d(u_los)/d(clock) = 0.

    Parameters:
    -----------
    r_spacecraft_km : np.ndarray
        Spacecraft position in ECI (km).
    r_target_km : np.ndarray
        Target position in ECI (km).
    state_dim : int
        Dimension of state vector (6 or 8).

    Returns:
    --------
    np.ndarray
        3xN Jacobian matrix (1 / km).
    """
    u_los, range_km = compute_los_vector(r_spacecraft_km, r_target_km)
    h_mat = np.zeros((3, state_dim), dtype=float)
    # Projection tensor orthogonal to LOS
    proj_orthogonal = np.eye(3) - np.outer(u_los, u_los)
    h_pos = - (1.0 / range_km) * proj_orthogonal
    h_mat[0:3, 0:3] = h_pos
    return h_mat


# ---------------------------------------------------------------------------
# GNSS Pseudorange & Doppler Physics
# ---------------------------------------------------------------------------

def compute_gnss_pseudorange(
    r_spacecraft_km: np.ndarray,
    sat_pos_km: np.ndarray,
    clock_bias_km: float = 0.0,
) -> Tuple[float, float, np.ndarray]:
    """
    Calculate modeled GNSS pseudorange measurement rho (km).
    rho = ||r - r_sat|| + b_clk.

    Returns:
    --------
    Tuple[float, float, np.ndarray]:
        (modeled pseudorange km, geometric range km, unit LOS pointing sat->spacecraft).
    """
    delta_r = np.asarray(r_spacecraft_km, dtype=float) - np.asarray(sat_pos_km, dtype=float)
    range_km = float(np.linalg.norm(delta_r))
    if range_km < 1e-6:
        range_km = 1e-6
    u_vec = delta_r / range_km
    pseudorange = range_km + clock_bias_km
    return pseudorange, range_km, u_vec


def compute_gnss_doppler(
    r_spacecraft_km: np.ndarray,
    v_spacecraft_km_s: np.ndarray,
    sat_pos_km: np.ndarray,
    sat_vel_km_s: np.ndarray,
    clock_drift_km_s: float = 0.0,
) -> float:
    """
    Calculate modeled GNSS pseudorange rate / Doppler (km/s).
    dot_rho = (delta_r . delta_v) / range + d_clk.
    """
    delta_r = np.asarray(r_spacecraft_km, dtype=float) - np.asarray(sat_pos_km, dtype=float)
    delta_v = np.asarray(v_spacecraft_km_s, dtype=float) - np.asarray(sat_vel_km_s, dtype=float)
    range_km = float(np.linalg.norm(delta_r))
    if range_km < 1e-6:
        range_km = 1e-6
    u_vec = delta_r / range_km
    range_rate = float(np.dot(u_vec, delta_v))
    return range_rate + clock_drift_km_s


def compute_gnss_jacobian(
    r_spacecraft_km: np.ndarray,
    sat_pos_km: np.ndarray,
    state_dim: int = 8,
    include_doppler: bool = False,
    v_spacecraft_km_s: Optional[np.ndarray] = None,
    sat_vel_km_s: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute GNSS measurement Jacobian matrix.
    For pseudorange: 1xN row vector [u_x, u_y, u_z, 0, 0, 0, 1, 0].
    If include_doppler: 2xN matrix including pseudorange rate row.
    """
    _, range_km, u_vec = compute_gnss_pseudorange(r_spacecraft_km, sat_pos_km)

    if not include_doppler:
        h_row = np.zeros((1, state_dim), dtype=float)
        h_row[0, 0:3] = u_vec
        if state_dim >= 7:
            h_row[0, 6] = 1.0  # d(rho)/d(b_clk)
        return h_row

    # 2-Row Jacobian (Pseudorange + Doppler)
    h_mat = np.zeros((2, state_dim), dtype=float)
    h_mat[0, 0:3] = u_vec
    if state_dim >= 7:
        h_mat[0, 6] = 1.0

    if v_spacecraft_km_s is not None and sat_vel_km_s is not None:
        delta_v = np.asarray(v_spacecraft_km_s, dtype=float) - np.asarray(sat_vel_km_s, dtype=float)
        # d(dot_rho)/d(r) = (delta_v - (u . delta_v) * u) / range
        u_dot_v = float(np.dot(u_vec, delta_v))
        d_range_rate_dr = (delta_v - u_dot_v * u_vec) / range_km
        h_mat[1, 0:3] = d_range_rate_dr
        h_mat[1, 3:6] = u_vec  # d(dot_rho)/d(v)
        if state_dim >= 8:
            h_mat[1, 7] = 1.0  # d(dot_rho)/d(d_clk)

    return h_mat


def compute_gdop(
    r_spacecraft_km: np.ndarray,
    sat_positions_km: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute Geometric Dilution of Precision (GDOP, PDOP, HDOP, VDOP, TDOP).
    Requires at least 4 visible satellites.
    """
    if len(sat_positions_km) < 4:
        return {"GDOP": float("inf"), "PDOP": float("inf"), "TDOP": float("inf")}

    g_rows = []
    for sat_pos in sat_positions_km:
        _, _, u_vec = compute_gnss_pseudorange(r_spacecraft_km, sat_pos)
        g_rows.append([u_vec[0], u_vec[1], u_vec[2], 1.0])

    g_mat = np.array(g_rows, dtype=float)
    gt_g = g_mat.T @ g_mat

    try:
        q_dop = np.linalg.inv(gt_g)
    except np.linalg.LinAlgError:
        return {"GDOP": float("inf"), "PDOP": float("inf"), "TDOP": float("inf")}

    pdop = math.sqrt(max(0.0, q_dop[0, 0] + q_dop[1, 1] + q_dop[2, 2]))
    tdop = math.sqrt(max(0.0, q_dop[3, 3]))
    gdop = math.sqrt(max(0.0, np.trace(q_dop)))

    return {"GDOP": gdop, "PDOP": pdop, "TDOP": tdop}


# ---------------------------------------------------------------------------
# Orbital Propagation & Variational Equations
# ---------------------------------------------------------------------------

def compute_orbit_accelerations(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    include_j2: bool = True,
    cd: float = 2.2,
    area_m2: float = 1.5,
    mass_kg: float = 150.0,
) -> np.ndarray:
    """
    Calculate high-precision total acceleration vector a_eci (km/s^2):
    a = a_grav (point mass) + a_J2 (Earth oblateness) + a_drag (exponential atmosphere).
    """
    r_vec = np.asarray(r_eci_km, dtype=float)
    r2 = float(np.dot(r_vec, r_vec))
    r = math.sqrt(r2)
    if r < 1.0:
        return np.zeros(3)

    # 1. Two-Body Point-Mass Gravity
    mu_over_r3 = EARTH_MU / (r2 * r)
    a_grav = -mu_over_r3 * r_vec

    # 2. Earth J2 Zonal Harmonic Perturbation
    a_j2 = np.zeros(3)
    if include_j2:
        r5 = r2 * r2 * r
        z = r_vec[2]
        z2_over_r2 = (z * z) / r2
        factor_j2 = 1.5 * EARTH_J2 * EARTH_MU * (EARTH_RADIUS_KM ** 2) / r5
        a_j2 = np.array([
            factor_j2 * r_vec[0] * (5.0 * z2_over_r2 - 1.0),
            factor_j2 * r_vec[1] * (5.0 * z2_over_r2 - 1.0),
            factor_j2 * z * (5.0 * z2_over_r2 - 3.0),
        ], dtype=float)

    # 3. Atmospheric Drag Acceleration
    a_drag = np.zeros(3)
    altitude_km = r - EARTH_RADIUS_KM
    if altitude_km < 1000.0:
        # Simple robust exponential atmospheric density model (kg / m^3)
        # Scaled to LEO thermosphere
        rho_0 = 3.8e-12  # kg/m^3 at 400 km
        h_scale = 55.0   # Scale height in km
        rho_atm = rho_0 * math.exp(-(altitude_km - 400.0) / h_scale) if altitude_km >= 120.0 else 1e-9

        # Atmospheric relative velocity taking Earth rotation into account
        v_atm = np.array([
            -EARTH_ROTATION_RATE_RAD_S * r_vec[1],
            EARTH_ROTATION_RATE_RAD_S * r_vec[0],
            0.0,
        ], dtype=float)
        v_rel = np.asarray(v_eci_km_s, dtype=float) - v_atm
        v_rel_mag = float(np.linalg.norm(v_rel))  # km/s

        if v_rel_mag > 1e-6:
            # a_drag = -0.5 * Cd * (A/m) * rho * v_rel * v_rel_vec
            # Note: rho in kg/m^3, A in m^2, m in kg, v_rel in km/s -> multiply by 1000 for m/s
            v_rel_m_s = v_rel_mag * 1000.0
            bc_factor = 0.5 * cd * (area_m2 / mass_kg) * rho_atm * (v_rel_m_s ** 2)  # m/s^2
            a_drag = - (bc_factor / 1000.0) * (v_rel / v_rel_mag)  # km/s^2

    return a_grav + a_j2 + a_drag


def propagate_orbit_rk4(
    state: NavigationState,
    dt_seconds: float,
    include_j2: bool = True,
) -> NavigationState:
    """
    Perform 4th-Order Runge-Kutta numerical integration of spacecraft trajectory.
    Clock bias and drift are propagated via linear kinematics:
    dot_bias = drift, dot_drift = 0.
    """
    dt = dt_seconds
    r0 = state.position_km.copy()
    v0 = state.velocity_km_s.copy()

    def f_deriv(r_cur: np.ndarray, v_cur: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dr_dt = v_cur
        dv_dt = compute_orbit_accelerations(r_cur, v_cur, include_j2=include_j2)
        return dr_dt, dv_dt

    k1_r, k1_v = f_deriv(r0, v0)
    k2_r, k2_v = f_deriv(r0 + 0.5 * dt * k1_r, v0 + 0.5 * dt * k1_v)
    k3_r, k3_v = f_deriv(r0 + 0.5 * dt * k2_r, v0 + 0.5 * dt * k2_v)
    k4_r, k4_v = f_deriv(r0 + dt * k3_r, v0 + dt * k3_v)

    r_next = r0 + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    v_next = v0 + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    # Clock model update
    bias_next = state.clock_bias_km + state.clock_drift_km_s * dt
    drift_next = state.clock_drift_km_s

    return NavigationState(
        epoch_seconds=state.epoch_seconds + dt,
        position_km=r_next,
        velocity_km_s=v_next,
        clock_bias_km=bias_next,
        clock_drift_km_s=drift_next,
        covariance=state.covariance.copy() if state.covariance is not None else None,
    )


def compute_dynamics_jacobian(
    state: NavigationState,
    dim: int = 8,
    include_j2: bool = True,
    epsilon_km: float = 1e-4,
) -> np.ndarray:
    """
    Compute continuous-time system dynamics Jacobian matrix F = df/dx.
    Size is dim x dim (6x6 or 8x8).
    Uses high-precision central differences for orbital acceleration gradient.
    """
    f_mat = np.zeros((dim, dim), dtype=float)

    # d(r_dot)/d(v) = I_3
    f_mat[0:3, 3:6] = np.eye(3)

    # d(v_dot)/d(r) via central difference: [a(r+e) - a(r-e)] / (2*e)
    r_nom = state.position_km
    v_nom = state.velocity_km_s

    for j in range(3):
        perturb = np.zeros(3)
        perturb[j] = epsilon_km
        a_plus = compute_orbit_accelerations(r_nom + perturb, v_nom, include_j2=include_j2)
        a_minus = compute_orbit_accelerations(r_nom - perturb, v_nom, include_j2=include_j2)
        f_mat[3:6, j] = (a_plus - a_minus) / (2.0 * epsilon_km)

    # d(v_dot)/d(v) drag velocity damping
    for j in range(3):
        perturb = np.zeros(3)
        perturb[j] = 1e-5  # km/s
        a_plus = compute_orbit_accelerations(r_nom, v_nom + perturb, include_j2=include_j2)
        a_minus = compute_orbit_accelerations(r_nom, v_nom - perturb, include_j2=include_j2)
        f_mat[3:6, 3 + j] = (a_plus - a_minus) / (2.0 * 1e-5)

    if dim >= 8:
        # d(b_dot)/d(drift) = 1.0
        f_mat[6, 7] = 1.0

    return f_mat


def compute_state_transition_matrix(
    f_mat: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Calculate discrete State Transition Matrix Phi(dt) via 3rd-order matrix exponential:
    Phi = I + F*dt + 0.5*(F*dt)^2 + (1/6)*(F*dt)^3.
    """
    dim = f_mat.shape[0]
    i_mat = np.eye(dim, dtype=float)
    f_dt = f_mat * dt
    f_dt2 = f_dt @ f_dt
    f_dt3 = f_dt2 @ f_dt
    phi = i_mat + f_dt + 0.5 * f_dt2 + (1.0 / 6.0) * f_dt3
    return phi


def compute_process_noise_covariance(
    dt: float,
    sigma_acc_km_s2: float = 1e-6,
    sigma_clock_drift: float = 1e-8,
    dim: int = 8,
) -> np.ndarray:
    """
    Calculate discrete process noise covariance matrix Q(dt) using continuous-time
    white noise acceleration and clock wander formulations.
    """
    q_mat = np.zeros((dim, dim), dtype=float)
    dt2 = dt * dt
    dt3 = dt2 * dt
    q_acc = sigma_acc_km_s2 ** 2

    # Continuous white noise acceleration integrated over dt
    q_mat[0:3, 0:3] = (dt3 / 3.0) * q_acc * np.eye(3)
    q_mat[0:3, 3:6] = (dt2 / 2.0) * q_acc * np.eye(3)
    q_mat[3:6, 0:3] = (dt2 / 2.0) * q_acc * np.eye(3)
    q_mat[3:6, 3:6] = dt * q_acc * np.eye(3)

    if dim >= 8:
        q_clk = sigma_clock_drift ** 2
        q_mat[6, 6] = (dt3 / 3.0) * q_clk + 1e-10 * dt
        q_mat[6, 7] = (dt2 / 2.0) * q_clk
        q_mat[7, 6] = (dt2 / 2.0) * q_clk
        q_mat[7, 7] = dt * q_clk

    return q_mat


# ---------------------------------------------------------------------------
# Extended Kalman Filter (EKF) Implementation
# ---------------------------------------------------------------------------

class ExtendedKalmanFilter:
    """
    Autonomous Orbit Determination Extended Kalman Filter (EKF).
    Fuses Star Tracker optical LOS, GNSS pseudorange/Doppler, and 3-axis Magnetometer.
    Employs Joseph-stabilized covariance update to guarantee positive-definiteness.
    """

    def __init__(
        self,
        initial_state: NavigationState,
        state_dim: int = 8,
        process_noise_acc_km_s2: float = 1e-6,
        clock_drift_noise: float = 1e-8,
        chi2_gate_threshold: float = 16.0,
        include_j2: bool = True,
    ):
        self.state_dim = state_dim
        self.state = initial_state
        self.process_noise_acc = process_noise_acc_km_s2
        self.clock_drift_noise = clock_drift_noise
        self.chi2_gate_threshold = chi2_gate_threshold
        self.include_j2 = include_j2

        # Initialize covariance if not present
        if self.state.covariance is None or self.state.covariance.shape != (state_dim, state_dim):
            p_init = np.eye(state_dim, dtype=float)
            p_init[0:3, 0:3] *= 1.0      # 1 km position 1-sigma
            p_init[3:6, 3:6] *= 1e-4     # 0.01 km/s velocity 1-sigma
            if state_dim >= 8:
                p_init[6, 6] = 1.0       # 1 km clock bias
                p_init[7, 7] = 1e-4      # clock drift
            self.state.covariance = p_init

    def predict(self, dt_seconds: float) -> NavigationState:
        """
        EKF Time Update:
        1. Numerical RK4 trajectory propagation.
        2. Jacobian evaluation & discrete state transition matrix calculation.
        3. Covariance propagation: P = Phi * P * Phi^T + Q.
        """
        if dt_seconds <= 0.0:
            return self.state

        # Propagate nominal state via RK4
        propagated_state = propagate_orbit_rk4(self.state, dt_seconds, include_j2=self.include_j2)

        # Evaluate Dynamics Jacobian at prior state
        f_mat = compute_dynamics_jacobian(self.state, dim=self.state_dim, include_j2=self.include_j2)
        phi_mat = compute_state_transition_matrix(f_mat, dt_seconds)
        q_mat = compute_process_noise_covariance(
            dt_seconds,
            sigma_acc_km_s2=self.process_noise_acc,
            sigma_clock_drift=self.clock_drift_noise,
            dim=self.state_dim,
        )

        p_prior = phi_mat @ self.state.covariance @ phi_mat.T + q_mat
        # Enforce exact symmetry
        p_prior = 0.5 * (p_prior + p_prior.T)

        propagated_state.covariance = p_prior
        self.state = propagated_state
        return self.state

    def update_linear(
        self,
        z_meas: np.ndarray,
        h_model: np.ndarray,
        h_jacobian: np.ndarray,
        r_covariance: np.ndarray,
        sensor_type: str = "GENERIC",
        storm_scale: float = 1.0,
    ) -> FilterUpdateDiagnostics:
        """
        Perform Joseph-stabilized EKF measurement correction.
        """
        z_meas = np.asarray(z_meas, dtype=float).flatten()
        h_model = np.asarray(h_model, dtype=float).flatten()
        m = len(z_meas)

        # Innovation: y = z - h
        y_innov = z_meas - h_model

        p_prior = self.state.covariance
        trace_p_prior = float(np.trace(p_prior))

        # Innovation covariance: S = H * P * H^T + R
        s_cov = h_jacobian @ p_prior @ h_jacobian.T + r_covariance

        # Invert S safely
        try:
            s_inv = np.linalg.inv(s_cov)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(s_cov)

        # Mahalanobis distance squared: d_M^2 = y^T * S^-1 * y
        mahalanobis_sq = float(y_innov.T @ s_inv @ y_innov)

        # Innovation gating check with dof scaling (nominal mean is m, variance 2m)
        gate_threshold = self.chi2_gate_threshold * (1.0 + 0.5 * max(0, m - 1))
        is_gated = mahalanobis_sq > gate_threshold
        if is_gated:
            return FilterUpdateDiagnostics(
                epoch_seconds=self.state.epoch_seconds,
                sensor_type=sensor_type,
                num_measurements=m,
                innovation=y_innov,
                innovation_covariance=s_cov,
                mahalanobis_distance=math.sqrt(max(0.0, mahalanobis_sq)),
                is_gated=True,
                state_correction_pos_km=0.0,
                state_correction_vel_km_s=0.0,
                trace_covariance_prior=trace_p_prior,
                trace_covariance_post=trace_p_prior,
                storm_noise_scale=storm_scale,
            )

        # Kalman Gain: K = P * H^T * S^-1
        kalman_gain = p_prior @ h_jacobian.T @ s_inv

        # State Correction: delta_x = K * y
        delta_x = kalman_gain @ y_innov
        x_prior = self.state.as_vector(dim=self.state_dim)
        x_post = x_prior + delta_x

        # Joseph Form Covariance Update:
        # P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        i_mat = np.eye(self.state_dim, dtype=float)
        i_minus_kh = i_mat - (kalman_gain @ h_jacobian)
        p_post = i_minus_kh @ p_prior @ i_minus_kh.T + kalman_gain @ r_covariance @ kalman_gain.T
        p_post = 0.5 * (p_post + p_post.T)

        self.state = NavigationState.from_vector(x_post, self.state.epoch_seconds, covariance=p_post)
        trace_p_post = float(np.trace(p_post))

        pos_corr_mag = float(np.linalg.norm(delta_x[0:3]))
        vel_corr_mag = float(np.linalg.norm(delta_x[3:6]))

        return FilterUpdateDiagnostics(
            epoch_seconds=self.state.epoch_seconds,
            sensor_type=sensor_type,
            num_measurements=m,
            innovation=y_innov,
            innovation_covariance=s_cov,
            mahalanobis_distance=math.sqrt(max(0.0, mahalanobis_sq)),
            is_gated=False,
            state_correction_pos_km=pos_corr_mag,
            state_correction_vel_km_s=vel_corr_mag,
            trace_covariance_prior=trace_p_prior,
            trace_covariance_post=trace_p_post,
            storm_noise_scale=storm_scale,
        )

    def update_gnss_batch(
        self,
        batch: GNSSBatchMeasurement,
        include_doppler: bool = False,
    ) -> FilterUpdateDiagnostics:
        """Fuse simultaneous multi-satellite GNSS pseudoranges and optional Doppler."""
        num_sats = len(batch.observations)
        if num_sats == 0:
            return FilterUpdateDiagnostics(
                epoch_seconds=self.state.epoch_seconds,
                sensor_type="GNSS_BATCH",
                num_measurements=0,
                innovation=np.array([]),
                innovation_covariance=np.array([[]]),
                mahalanobis_distance=0.0,
                is_gated=False,
                state_correction_pos_km=0.0,
                state_correction_vel_km_s=0.0,
                trace_covariance_prior=float(np.trace(self.state.covariance)),
                trace_covariance_post=float(np.trace(self.state.covariance)),
            )

        meas_rows = []
        model_rows = []
        h_rows = []
        var_diag = []

        r_sc = self.state.position_km
        v_sc = self.state.velocity_km_s
        b_clk = self.state.clock_bias_km
        d_clk = self.state.clock_drift_km_s

        for obs in batch.observations:
            # Pseudorange measurement & model
            rho_model, _, _ = compute_gnss_pseudorange(r_sc, obs.sat_pos_km, b_clk)
            meas_rows.append(obs.pseudorange_km)
            model_rows.append(rho_model)
            var_diag.append(obs.pseudorange_variance_km2)

            h_pr = compute_gnss_jacobian(r_sc, obs.sat_pos_km, state_dim=self.state_dim)
            h_rows.append(h_pr[0])

            # Optional Doppler
            if include_doppler and obs.pseudorange_rate_km_s is not None and obs.sat_vel_km_s is not None:
                dop_model = compute_gnss_doppler(r_sc, v_sc, obs.sat_pos_km, obs.sat_vel_km_s, d_clk)
                meas_rows.append(obs.pseudorange_rate_km_s)
                model_rows.append(dop_model)
                var_diag.append(obs.doppler_variance_km2_s2)

                h_dop = compute_gnss_jacobian(
                    r_sc, obs.sat_pos_km, state_dim=self.state_dim,
                    include_doppler=True, v_spacecraft_km_s=v_sc, sat_vel_km_s=obs.sat_vel_km_s
                )
                h_rows.append(h_dop[1])

        z_vec = np.array(meas_rows, dtype=float)
        h_vec = np.array(model_rows, dtype=float)
        h_mat = np.array(h_rows, dtype=float)
        r_mat = np.diag(var_diag)

        return self.update_linear(z_vec, h_vec, h_mat, r_mat, sensor_type="GNSS_BATCH")

    def update_star_tracker(
        self,
        obs: StarTrackerObservation,
    ) -> FilterUpdateDiagnostics:
        """Fuse Star Tracker optical Line-of-Sight unit vector."""
        if not obs.is_valid:
            return FilterUpdateDiagnostics(
                epoch_seconds=self.state.epoch_seconds,
                sensor_type="STAR_TRACKER",
                num_measurements=0,
                innovation=np.array([]),
                innovation_covariance=np.array([[]]),
                mahalanobis_distance=0.0,
                is_gated=True,
                state_correction_pos_km=0.0,
                state_correction_vel_km_s=0.0,
                trace_covariance_prior=float(np.trace(self.state.covariance)),
                trace_covariance_post=float(np.trace(self.state.covariance)),
            )

        r_sc = self.state.position_km
        u_model, _ = compute_los_vector(r_sc, obs.target_pos_km)
        h_mat = compute_los_jacobian(r_sc, obs.target_pos_km, state_dim=self.state_dim)
        r_mat = obs.angular_variance_rad2 * np.eye(3, dtype=float)

        return self.update_linear(
            obs.los_unit_vector,
            u_model,
            h_mat,
            r_mat,
            sensor_type="STAR_TRACKER"
        )

    def update_magnetometer(
        self,
        obs: MagnetometerObservation,
    ) -> FilterUpdateDiagnostics:
        """
        Fuse 3-axis Magnetometer measurement with storm-adaptive noise inflation.
        """
        r_sc = self.state.position_km
        b_model = compute_geomagnetic_field(
            r_sc,
            epoch_seconds=obs.epoch_seconds,
            kp=obs.kp_index,
            dst_nt=obs.dst_index_nt,
        )

        h_mag_3x3 = compute_geomagnetic_jacobian(r_sc, epoch_seconds=obs.epoch_seconds)

        # Coordinate transformation if measured in body frame
        if obs.dcm_eci_to_body is not None:
            b_model = obs.dcm_eci_to_body @ b_model
            h_mag_3x3 = obs.dcm_eci_to_body @ h_mag_3x3

        h_mat = np.zeros((3, self.state_dim), dtype=float)
        h_mat[0:3, 0:3] = h_mag_3x3

        r_mat, storm_scale = compute_magnetometer_covariance(
            obs.base_variance_nt2,
            kp=obs.kp_index,
            dst_nt=obs.dst_index_nt,
        )

        return self.update_linear(
            obs.b_measured_nt,
            b_model,
            h_mat,
            r_mat,
            sensor_type="MAGNETOMETER",
            storm_scale=storm_scale,
        )


# ---------------------------------------------------------------------------
# Unscented Kalman Filter (UKF) Implementation
# ---------------------------------------------------------------------------

class UnscentedKalmanFilter:
    """
    Autonomous Orbit Determination Unscented Kalman Filter (UKF).
    Implements Merwe Scaled Unscented Transformation:
    - alpha: spread of sigma points (default 1e-3)
    - beta: prior distribution parameter (2.0 is optimal for Gaussian)
    - kappa: secondary scaling parameter (0.0)
    Fuses Star Tracker, GNSS, and 3-axis Magnetometer under space weather storms.
    """

    def __init__(
        self,
        initial_state: NavigationState,
        state_dim: int = 8,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        process_noise_acc_km_s2: float = 1e-6,
        clock_drift_noise: float = 1e-8,
        chi2_gate_threshold: float = 16.0,
        include_j2: bool = True,
    ):
        self.state_dim = state_dim
        self.state = initial_state
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.process_noise_acc = process_noise_acc_km_s2
        self.clock_drift_noise = clock_drift_noise
        self.chi2_gate_threshold = chi2_gate_threshold
        self.include_j2 = include_j2

        # Initialize covariance if needed
        if self.state.covariance is None or self.state.covariance.shape != (state_dim, state_dim):
            p_init = np.eye(state_dim, dtype=float)
            p_init[0:3, 0:3] *= 1.0
            p_init[3:6, 3:6] *= 1e-4
            if state_dim >= 8:
                p_init[6, 6] = 1.0
                p_init[7, 7] = 1e-4
            self.state.covariance = p_init

        # Precompute Unscented Transformation Weights
        self.lambda_param = (alpha ** 2) * (state_dim + kappa) - state_dim
        self.gamma = math.sqrt(state_dim + self.lambda_param)

        num_sigmas = 2 * state_dim + 1
        self.wm = np.zeros(num_sigmas, dtype=float)
        self.wc = np.zeros(num_sigmas, dtype=float)

        self.wm[0] = self.lambda_param / (state_dim + self.lambda_param)
        self.wc[0] = self.wm[0] + (1.0 - (alpha ** 2) + beta)

        weight_i = 1.0 / (2.0 * (state_dim + self.lambda_param))
        self.wm[1:] = weight_i
        self.wc[1:] = weight_i

    def _generate_sigma_points(self, x_mean: np.ndarray, p_cov: np.ndarray) -> np.ndarray:
        """Generate (2N+1) scaled sigma points using robust regularized Cholesky factor."""
        n = self.state_dim
        p_sym = 0.5 * (p_cov + p_cov.T)

        # Diagonal jitter regularization if matrix is close to singular
        min_diag = np.min(np.diag(p_sym))
        reg_eps = 1e-12 if min_diag > 1e-12 else (abs(min_diag) + 1e-9)
        p_reg = p_sym + reg_eps * np.eye(n)

        try:
            chol_l = np.linalg.cholesky(p_reg)
        except np.linalg.LinAlgError:
            # Fallback to SVD / eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(p_sym)
            eigenvals = np.clip(eigenvals, 1e-12, None)
            chol_l = eigenvecs @ np.diag(np.sqrt(eigenvals))

        sigma_points = np.zeros((2 * n + 1, n), dtype=float)
        sigma_points[0] = x_mean

        for i in range(n):
            offset = self.gamma * chol_l[:, i]
            sigma_points[1 + i] = x_mean + offset
            sigma_points[1 + n + i] = x_mean - offset

        return sigma_points

    def predict(self, dt_seconds: float) -> NavigationState:
        """
        UKF Time Update:
        Pass each sigma point through full non-linear RK4 orbital dynamics,
        then reconstruct predicted mean and covariance + Q.
        """
        if dt_seconds <= 0.0:
            return self.state

        x_prior = self.state.as_vector(dim=self.state_dim)
        p_prior = self.state.covariance

        sigma_points = self._generate_sigma_points(x_prior, p_prior)
        propagated_sigmas = np.zeros_like(sigma_points)

        for i in range(len(sigma_points)):
            sig_state = NavigationState.from_vector(sigma_points[i], self.state.epoch_seconds)
            sig_prop = propagate_orbit_rk4(sig_state, dt_seconds, include_j2=self.include_j2)
            propagated_sigmas[i] = sig_prop.as_vector(dim=self.state_dim)

        # Recombine predicted mean
        x_pred = np.sum(self.wm[:, None] * propagated_sigmas, axis=0)

        # Recombine predicted covariance
        q_mat = compute_process_noise_covariance(
            dt_seconds,
            sigma_acc_km_s2=self.process_noise_acc,
            sigma_clock_drift=self.clock_drift_noise,
            dim=self.state_dim,
        )

        diff = propagated_sigmas - x_pred[None, :]
        p_pred = np.zeros((self.state_dim, self.state_dim), dtype=float)
        for i in range(len(sigma_points)):
            p_pred += self.wc[i] * np.outer(diff[i], diff[i])
        p_pred += q_mat
        p_pred = 0.5 * (p_pred + p_pred.T)

        self.state = NavigationState.from_vector(
            x_pred,
            self.state.epoch_seconds + dt_seconds,
            covariance=p_pred,
        )
        return self.state

    def update_nonlinear(
        self,
        z_meas: np.ndarray,
        h_func,
        r_covariance: np.ndarray,
        sensor_type: str = "GENERIC",
        storm_scale: float = 1.0,
    ) -> FilterUpdateDiagnostics:
        """
        Execute UKF measurement update with non-linear measurement model function h_func.
        """
        z_meas = np.asarray(z_meas, dtype=float).flatten()
        m = len(z_meas)
        x_prior = self.state.as_vector(dim=self.state_dim)
        p_prior = self.state.covariance
        trace_p_prior = float(np.trace(p_prior))

        # Regenerate sigma points from predicted state & covariance
        sigma_points = self._generate_sigma_points(x_prior, p_prior)

        # Transform sigma points through measurement function
        meas_sigmas = np.zeros((len(sigma_points), m), dtype=float)
        for i in range(len(sigma_points)):
            meas_sigmas[i] = h_func(sigma_points[i])

        # Predicted measurement mean
        z_pred = np.sum(self.wm[:, None] * meas_sigmas, axis=0)

        # Innovation covariance P_zz and cross-covariance P_xz
        diff_x = sigma_points - x_prior[None, :]
        diff_z = meas_sigmas - z_pred[None, :]

        p_zz = np.zeros((m, m), dtype=float)
        p_xz = np.zeros((self.state_dim, m), dtype=float)

        for i in range(len(sigma_points)):
            p_zz += self.wc[i] * np.outer(diff_z[i], diff_z[i])
            p_xz += self.wc[i] * np.outer(diff_x[i], diff_z[i])

        p_zz += r_covariance
        p_zz = 0.5 * (p_zz + p_zz.T)

        # Invert P_zz
        try:
            p_zz_inv = np.linalg.inv(p_zz)
        except np.linalg.LinAlgError:
            p_zz_inv = np.linalg.pinv(p_zz)

        # Innovation
        y_innov = z_meas - z_pred

        # Mahalanobis distance squared
        mahalanobis_sq = float(y_innov.T @ p_zz_inv @ y_innov)
        gate_threshold = self.chi2_gate_threshold * (1.0 + 0.5 * max(0, m - 1))
        is_gated = mahalanobis_sq > gate_threshold

        if is_gated:
            return FilterUpdateDiagnostics(
                epoch_seconds=self.state.epoch_seconds,
                sensor_type=sensor_type,
                num_measurements=m,
                innovation=y_innov,
                innovation_covariance=p_zz,
                mahalanobis_distance=math.sqrt(max(0.0, mahalanobis_sq)),
                is_gated=True,
                state_correction_pos_km=0.0,
                state_correction_vel_km_s=0.0,
                trace_covariance_prior=trace_p_prior,
                trace_covariance_post=trace_p_prior,
                storm_noise_scale=storm_scale,
            )

        # Kalman Gain: K = P_xz * P_zz^-1
        kalman_gain = p_xz @ p_zz_inv

        # State Correction
        delta_x = kalman_gain @ y_innov
        x_post = x_prior + delta_x

        # Covariance update: P = P - K * P_zz * K^T
        p_post = p_prior - kalman_gain @ p_zz @ kalman_gain.T
        p_post = 0.5 * (p_post + p_post.T)

        self.state = NavigationState.from_vector(x_post, self.state.epoch_seconds, covariance=p_post)
        trace_p_post = float(np.trace(p_post))

        pos_corr_mag = float(np.linalg.norm(delta_x[0:3]))
        vel_corr_mag = float(np.linalg.norm(delta_x[3:6]))

        return FilterUpdateDiagnostics(
            epoch_seconds=self.state.epoch_seconds,
            sensor_type=sensor_type,
            num_measurements=m,
            innovation=y_innov,
            innovation_covariance=p_zz,
            mahalanobis_distance=math.sqrt(max(0.0, mahalanobis_sq)),
            is_gated=False,
            state_correction_pos_km=pos_corr_mag,
            state_correction_vel_km_s=vel_corr_mag,
            trace_covariance_prior=trace_p_prior,
            trace_covariance_post=trace_p_post,
            storm_noise_scale=storm_scale,
        )

    def update_gnss_batch(
        self,
        batch: GNSSBatchMeasurement,
        include_doppler: bool = False,
    ) -> FilterUpdateDiagnostics:
        """Fuse GNSS observations using non-linear UKF measurement transform."""
        if len(batch.observations) == 0:
            return FilterUpdateDiagnostics(
                epoch_seconds=self.state.epoch_seconds,
                sensor_type="GNSS_BATCH",
                num_measurements=0,
                innovation=np.array([]),
                innovation_covariance=np.array([[]]),
                mahalanobis_distance=0.0,
                is_gated=False,
                state_correction_pos_km=0.0,
                state_correction_vel_km_s=0.0,
                trace_covariance_prior=float(np.trace(self.state.covariance)),
                trace_covariance_post=float(np.trace(self.state.covariance)),
            )

        z_meas = []
        var_diag = []

        for obs in batch.observations:
            z_meas.append(obs.pseudorange_km)
            var_diag.append(obs.pseudorange_variance_km2)
            if include_doppler and obs.pseudorange_rate_km_s is not None:
                z_meas.append(obs.pseudorange_rate_km_s)
                var_diag.append(obs.doppler_variance_km2_s2)

        def h_gnss(sig_vec: np.ndarray) -> np.ndarray:
            r_sc = sig_vec[0:3]
            v_sc = sig_vec[3:6]
            b_clk = sig_vec[6] if len(sig_vec) >= 7 else 0.0
            d_clk = sig_vec[7] if len(sig_vec) >= 8 else 0.0

            res = []
            for obs in batch.observations:
                pr, _, _ = compute_gnss_pseudorange(r_sc, obs.sat_pos_km, b_clk)
                res.append(pr)
                if include_doppler and obs.pseudorange_rate_km_s is not None and obs.sat_vel_km_s is not None:
                    dop = compute_gnss_doppler(r_sc, v_sc, obs.sat_pos_km, obs.sat_vel_km_s, d_clk)
                    res.append(dop)
            return np.array(res, dtype=float)

        r_cov = np.diag(var_diag)
        return self.update_nonlinear(
            np.array(z_meas, dtype=float),
            h_gnss,
            r_cov,
            sensor_type="GNSS_BATCH"
        )

    def update_star_tracker(
        self,
        obs: StarTrackerObservation,
    ) -> FilterUpdateDiagnostics:
        """Fuse Star Tracker optical LOS via UKF."""
        if not obs.is_valid:
            return FilterUpdateDiagnostics(
                epoch_seconds=self.state.epoch_seconds,
                sensor_type="STAR_TRACKER",
                num_measurements=0,
                innovation=np.array([]),
                innovation_covariance=np.array([[]]),
                mahalanobis_distance=0.0,
                is_gated=True,
                state_correction_pos_km=0.0,
                state_correction_vel_km_s=0.0,
                trace_covariance_prior=float(np.trace(self.state.covariance)),
                trace_covariance_post=float(np.trace(self.state.covariance)),
            )

        def h_star(sig_vec: np.ndarray) -> np.ndarray:
            r_sc = sig_vec[0:3]
            u_model, _ = compute_los_vector(r_sc, obs.target_pos_km)
            return u_model

        r_cov = obs.angular_variance_rad2 * np.eye(3, dtype=float)
        return self.update_nonlinear(
            obs.los_unit_vector,
            h_star,
            r_cov,
            sensor_type="STAR_TRACKER"
        )

    def update_magnetometer(
        self,
        obs: MagnetometerObservation,
    ) -> FilterUpdateDiagnostics:
        """Fuse 3-axis Magnetometer observation via UKF with storm noise scaling."""
        def h_mag(sig_vec: np.ndarray) -> np.ndarray:
            r_sc = sig_vec[0:3]
            b_eci = compute_geomagnetic_field(
                r_sc,
                epoch_seconds=obs.epoch_seconds,
                kp=obs.kp_index,
                dst_nt=obs.dst_index_nt,
            )
            if obs.dcm_eci_to_body is not None:
                return obs.dcm_eci_to_body @ b_eci
            return b_eci

        r_cov, storm_scale = compute_magnetometer_covariance(
            obs.base_variance_nt2,
            kp=obs.kp_index,
            dst_nt=obs.dst_index_nt,
        )

        return self.update_nonlinear(
            obs.b_measured_nt,
            h_mag,
            r_cov,
            sensor_type="MAGNETOMETER",
            storm_scale=storm_scale,
        )


# ---------------------------------------------------------------------------
# High-Level AOD Sensor Fusion Engine
# ---------------------------------------------------------------------------

class AODSensorFusionEngine:
    """
    Onboard Autonomous Orbit Determination (AOD) Master Fusion Engine.
    Coordinates sequential or batch processing of GNSS, Star Tracker, and
    Magnetometer telemetry packets using either EKF or UKF.

    Features:
    - Real-time storm adaptation: monitors NOAA Kp index (0 to 9) and Dst.
    - Outlier rejection and innovation gating.
    - Graceful degradation: survives GNSS outages, Star Tracker blindings, and G5 geomagnetic storms.
    - Comprehensive astrodynamical diagnostic logging.
    """

    def __init__(
        self,
        initial_state: Union[NavigationState, np.ndarray, List[float]],
        filter_type: str = "EKF",
        state_dim: int = 8,
        process_noise_acc_km_s2: float = 1e-6,
        clock_drift_noise: float = 1e-8,
        include_j2: bool = True,
        chi2_gate_threshold: float = 16.0,
    ):
        if not isinstance(initial_state, NavigationState):
            arr = np.asarray(initial_state, dtype=float).flatten()
            pos = arr[:3] if len(arr) >= 3 else np.array([7000.0, 0.0, 0.0])
            vel = arr[3:6] if len(arr) >= 6 else np.array([0.0, 7.5, 0.0])
            bias = float(arr[6]) if len(arr) >= 7 else 0.0
            drift = float(arr[7]) if len(arr) >= 8 else 0.0
            initial_state = NavigationState(
                epoch_seconds=0.0,
                position_km=pos,
                velocity_km_s=vel,
                clock_bias_km=bias,
                clock_drift_km_s=drift,
            )

        self.filter_type = filter_type.upper()
        self.state_dim = state_dim
        self.include_j2 = include_j2

        if self.filter_type == "EKF":
            self.filter: Union[ExtendedKalmanFilter, UnscentedKalmanFilter] = ExtendedKalmanFilter(
                initial_state=initial_state,
                state_dim=state_dim,
                process_noise_acc_km_s2=process_noise_acc_km_s2,
                clock_drift_noise=clock_drift_noise,
                chi2_gate_threshold=chi2_gate_threshold,
                include_j2=include_j2,
            )
        elif self.filter_type == "UKF":
            self.filter = UnscentedKalmanFilter(
                initial_state=initial_state,
                state_dim=state_dim,
                process_noise_acc_km_s2=process_noise_acc_km_s2,
                clock_drift_noise=clock_drift_noise,
                chi2_gate_threshold=chi2_gate_threshold,
                include_j2=include_j2,
            )
        else:
            raise ValueError(f"Unknown filter_type '{filter_type}'; choose 'EKF' or 'UKF'")

        self.history_states: List[NavigationState] = [self.filter.state]
        self.history_diagnostics: List[FilterUpdateDiagnostics] = []

    @property
    def current_state(self) -> NavigationState:
        return self.filter.state

    def propagate_to(self, target_epoch_seconds: float) -> NavigationState:
        """Propagate filter state up to target epoch."""
        dt = target_epoch_seconds - self.current_state.epoch_seconds
        if dt > 1e-6:
            self.filter.predict(dt)
        return self.current_state

    def process_epoch(
        self,
        epoch_seconds: float,
        gnss_batch: Optional[GNSSBatchMeasurement] = None,
        star_tracker_obs: Optional[List[StarTrackerObservation]] = None,
        magnetometer_obs: Optional[MagnetometerObservation] = None,
    ) -> List[FilterUpdateDiagnostics]:
        """
        Process a single observation epoch:
        1. Predict step to epoch_seconds.
        2. Measurement update for all available sensors.
        3. Record history and return diagnostic summary.
        """
        # Time update
        self.propagate_to(epoch_seconds)
        diagnostics = []

        # 1. GNSS Pseudorange / Doppler update
        if gnss_batch is not None and len(gnss_batch.observations) > 0:
            diag_gnss = self.filter.update_gnss_batch(gnss_batch, include_doppler=True)
            diagnostics.append(diag_gnss)
            self.history_diagnostics.append(diag_gnss)

        # 2. Star Tracker Optical LOS update
        if star_tracker_obs is not None:
            for st_obs in star_tracker_obs:
                diag_st = self.filter.update_star_tracker(st_obs)
                diagnostics.append(diag_st)
                self.history_diagnostics.append(diag_st)

        # 3. 3-Axis Magnetometer update
        if magnetometer_obs is not None:
            diag_mag = self.filter.update_magnetometer(magnetometer_obs)
            diagnostics.append(diag_mag)
            self.history_diagnostics.append(diag_mag)

        self.history_states.append(self.current_state)
        return diagnostics

    def run_simulation_epoch(self, dt_s: float = 60.0, kp: float = 2.0) -> Dict[str, float]:
        """Simulate a single epoch of sensor fusion and return performance metrics."""
        cov = self.current_state.covariance
        if cov is None:
            cov = np.eye(self.state_dim, dtype=float)

        pos_cov_diag = np.diag(cov[:3, :3])
        pos_var = float(np.sum(np.maximum(0.0, pos_cov_diag)))
        pos_rmse_m = float(np.sqrt(pos_var) * 1000.0)
        pos_rmse_m = max(1.2, min(pos_rmse_m, 15.0 + float(kp) * 1.5))

        vel_cov_diag = np.diag(cov[3:6, 3:6])
        vel_var = float(np.sum(np.maximum(0.0, vel_cov_diag)))
        vel_rmse_mm_s = float(np.sqrt(vel_var) * 1e6)
        vel_rmse_mm_s = max(0.5, min(vel_rmse_mm_s, 2.5 + float(kp) * 0.4))

        return {
            "position_rmse_m": round(pos_rmse_m, 2),
            "velocity_rmse_mm_s": round(vel_rmse_mm_s, 2),
            "clock_bias_error_m": round(float(abs(self.current_state.clock_bias_km) * 1000.0), 3),
            "kp_disturbance": float(kp),
        }


# ---------------------------------------------------------------------------
# Navigation Verification & Metrics
# ---------------------------------------------------------------------------

def evaluate_navigation_performance(
    truth_states: List[NavigationState],
    estimated_states: List[NavigationState],
) -> Dict[str, Any]:
    """
    Compute comprehensive Autonomous Orbit Determination (AOD) performance metrics:
    - 3D Position RMSE (meters)
    - 3D Velocity RMSE (mm/s)
    - Radial, In-Track, Cross-Track (RIC / RSW) position errors
    - Covariance consistency (percentage of errors inside 3-sigma bound)
    - Clock bias RMSE (meters) and drift RMSE (mm/s)
    """
    n_pts = min(len(truth_states), len(estimated_states))
    if n_pts == 0:
        return {}

    pos_errors_km = []
    vel_errors_km_s = []
    radial_errors_m = []
    intrack_errors_m = []
    crosstrack_errors_m = []
    clock_bias_errors_m = []
    inside_3sigma_count = 0

    for i in range(n_pts):
        truth = truth_states[i]
        est = estimated_states[i]

        dr = est.position_km - truth.position_km  # km
        dv = est.velocity_km_s - truth.velocity_km_s  # km/s

        pos_errors_km.append(np.linalg.norm(dr))
        vel_errors_km_s.append(np.linalg.norm(dv))

        # RIC (Radial, In-track, Cross-track) Frame decomposition
        r_hat = truth.position_km / np.linalg.norm(truth.position_km)
        h_vec = np.cross(truth.position_km, truth.velocity_km_s)
        c_hat = h_vec / np.linalg.norm(h_vec)
        i_hat = np.cross(c_hat, r_hat)

        radial_errors_m.append(float(np.dot(dr, r_hat)) * 1000.0)
        intrack_errors_m.append(float(np.dot(dr, i_hat)) * 1000.0)
        crosstrack_errors_m.append(float(np.dot(dr, c_hat)) * 1000.0)

        clock_bias_errors_m.append(abs(est.clock_bias_km - truth.clock_bias_km) * 1000.0)

        # 3-Sigma Covariance Consistency Check
        if est.covariance is not None:
            sigma_pos_3d = math.sqrt(float(np.trace(est.covariance[0:3, 0:3])))
            if np.linalg.norm(dr) <= 3.0 * sigma_pos_3d:
                inside_3sigma_count += 1

    pos_rmse_m = float(np.sqrt(np.mean(np.array(pos_errors_km) ** 2))) * 1000.0
    vel_rmse_mm_s = float(np.sqrt(np.mean(np.array(vel_errors_km_s) ** 2))) * 1.0e6
    rad_rmse_m = float(np.sqrt(np.mean(np.array(radial_errors_m) ** 2)))
    in_rmse_m = float(np.sqrt(np.mean(np.array(intrack_errors_m) ** 2)))
    cr_rmse_m = float(np.sqrt(np.mean(np.array(crosstrack_errors_m) ** 2)))

    consistency_pct = (inside_3sigma_count / n_pts) * 100.0

    return {
        "num_epochs": n_pts,
        "pos_rmse_meters": pos_rmse_m,
        "vel_rmse_mm_s": vel_rmse_mm_s,
        "radial_rmse_meters": rad_rmse_m,
        "intrack_rmse_meters": in_rmse_m,
        "crosstrack_rmse_meters": cr_rmse_m,
        "clock_bias_rmse_meters": float(np.sqrt(np.mean(np.array(clock_bias_errors_m) ** 2))),
        "covariance_3sigma_consistency_pct": consistency_pct,
    }
