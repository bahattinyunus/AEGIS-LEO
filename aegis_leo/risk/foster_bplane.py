"""
AEGIS-LEO: B-Plane Encounter Geometry & Conjunction Collision Probability (Pc)
Implements:
1. Foster (1992) 1D numerical B-plane collision probability integration using error functions (erf).
2. Chan (1997) analytical isotropic series approximation using regularized incomplete gamma functions.
3. 3D-to-2D encounter frame (B-plane) projection and principal axis covariance diagonalization.
4. Maximum collision probability (Max Pc) under covariance scaling / dilution.

References:
- Foster, J. L., & Estes, H. S. (1992). "A Parametric Analysis of Orbital Debris Collision Probability
  and Maneuver Rate for Space Vehicles", NASA/JSC-25898.
- Chan, F. K. (1997). "Collision Probability Analyses for Earth-Orbiting Satellites",
  Advances in the Astronautical Sciences, Vol. 96, pp. 1033-1048.
- Chan, F. K. (2008). "Spacecraft Collision Probability", Aerospace Press / AIAA.
- Alfano, S. (2005). "A Numerical Implementation of Spherical Object Collision Probability",
  Journal of the Astronautical Sciences, Vol. 53, No. 1, pp. 69-90.
- NASA CARA (Conjunction Assessment Risk Analysis) Standard Work Instructions.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from scipy.special import erf, gammainc
from scipy.integrate import quad


@dataclass
class BPlaneConjunctionResult:
    """
    Comprehensive result container for B-plane conjunction assessment.
    """
    pc_foster: float
    pc_chan: float
    pc_max: float
    miss_distance_km: float
    relative_speed_km_s: float
    sigma_x_km: float
    sigma_y_km: float
    aspect_ratio: float
    mahalanobis_distance: float
    bplane_miss_coords_km: Tuple[float, float]
    principal_miss_coords_km: Tuple[float, float]
    rotation_angle_deg: float
    combined_radius_m: float
    is_critical: bool
    method: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for telemetry / logging."""
        return {
            "pc_foster": self.pc_foster,
            "pc_chan": self.pc_chan,
            "pc_max": self.pc_max,
            "miss_distance_km": self.miss_distance_km,
            "relative_speed_km_s": self.relative_speed_km_s,
            "sigma_x_km": self.sigma_x_km,
            "sigma_y_km": self.sigma_y_km,
            "aspect_ratio": self.aspect_ratio,
            "mahalanobis_distance": self.mahalanobis_distance,
            "bplane_miss_coords_km": list(self.bplane_miss_coords_km),
            "principal_miss_coords_km": list(self.principal_miss_coords_km),
            "rotation_angle_deg": self.rotation_angle_deg,
            "combined_radius_m": self.combined_radius_m,
            "is_critical": self.is_critical,
            "method": self.method,
        }


def build_encounter_frame(
    r1_eci: np.ndarray,
    v1_eci: np.ndarray,
    v2_eci: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the right-handed orthonormal encounter frame (B-plane) coordinate system.

    Parameters
    ----------
    r1_eci : np.ndarray
        Position vector of primary satellite in ECI frame (km), shape (3,).
    v1_eci : np.ndarray
        Velocity vector of primary satellite in ECI frame (km/s), shape (3,).
    v2_eci : np.ndarray
        Velocity vector of secondary object in ECI frame (km/s), shape (3,).

    Returns
    -------
    x_e : np.ndarray
        Unit vector in encounter plane orthogonal to relative velocity and orbit normal.
    y_e : np.ndarray
        Unit vector along the relative velocity direction (v_rel = v2 - v1).
    z_e : np.ndarray
        Unit vector completing right-handed triad in encounter plane (z_e = y_e x x_e).
    P : np.ndarray
        2x3 projection matrix mapping 3D ECI vectors onto the 2D encounter plane (x_e, z_e).
    """
    v_rel = np.asarray(v2_eci, dtype=float) - np.asarray(v1_eci, dtype=float)
    v_rel_mag = float(np.linalg.norm(v_rel))
    if v_rel_mag < 1e-7:
        raise ValueError("Relative velocity is near zero; encounter frame is undefined.")

    # y_e is along relative velocity
    y_e = v_rel / v_rel_mag

    # Primary orbital angular momentum h1 = r1 x v1
    h1 = np.cross(r1_eci, v1_eci)
    h1_mag = float(np.linalg.norm(h1))

    if h1_mag > 1e-7:
        h1_unit = h1 / h1_mag
        cross_cand = np.cross(y_e, h1_unit)
        cand_norm = float(np.linalg.norm(cross_cand))
    else:
        cand_norm = 0.0

    # If relative velocity is not collinear with orbit normal, use h1
    if cand_norm > 1e-4:
        x_e = cross_cand / cand_norm
    else:
        # Fallback to arbitrary perpendicular vector
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(y_e, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        cross_cand = np.cross(ref, y_e)
        x_e = cross_cand / np.linalg.norm(cross_cand)

    # z_e completes right-handed system: x_e x y_e = z_e
    z_e = np.cross(x_e, y_e)
    z_e /= np.linalg.norm(z_e)

    # Projection matrix P (2x3) projecting 3D vectors onto the encounter plane (x_e, z_e)
    P = np.vstack([x_e, z_e])

    return x_e, y_e, z_e, P


def project_to_encounter_plane(
    rel_pos_eci: np.ndarray,
    rel_vel_eci: np.ndarray,
    cov_primary_km2: np.ndarray,
    cov_secondary_km2: np.ndarray,
    r1_eci: Optional[np.ndarray] = None,
    v1_eci: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Project relative position and 3D combined covariance onto the 2D encounter plane (B-plane),
    and diagonalize into principal axes of uncertainty.

    Parameters
    ----------
    rel_pos_eci : np.ndarray
        Relative position (r2 - r1) at TCA in ECI (km).
    rel_vel_eci : np.ndarray
        Relative velocity (v2 - v1) at TCA in ECI (km/s).
    cov_primary_km2 : np.ndarray
        3x3 position error covariance matrix of primary satellite (km^2).
    cov_secondary_km2 : np.ndarray
        3x3 position error covariance matrix of secondary object (km^2).
    r1_eci : Optional[np.ndarray]
        Primary position vector (km). If None, encounter frame uses geometric fallback.
    v1_eci : Optional[np.ndarray]
        Primary velocity vector (km/s).

    Returns
    -------
    dict
        Contains:
        - "d_2d_km": 2D projected miss vector in encounter plane (x_e, z_e)
        - "cov_2d_km2": 2x2 projected covariance matrix in encounter plane
        - "sigma_x_km": major principal axis 1-sigma uncertainty (km)
        - "sigma_y_km": minor principal axis 1-sigma uncertainty (km)
        - "xm_km": miss distance component along major principal axis (km)
        - "ym_km": miss distance component along minor principal axis (km)
        - "rotation_angle_deg": orientation angle of principal axes in encounter plane (deg)
        - "mahalanobis_distance": Mahalanobis distance sqrt(d_2d^T * C_2d^-1 * d_2d)
        - "miss_distance_km": total 3D miss distance (km)
        - "relative_speed_km_s": relative velocity magnitude (km/s)
    """
    rel_pos = np.asarray(rel_pos_eci, dtype=float)
    rel_vel = np.asarray(rel_vel_eci, dtype=float)

    v_rel_mag = float(np.linalg.norm(rel_vel))
    if v_rel_mag < 1e-7:
        raise ValueError("Relative speed is too small for encounter plane projection.")

    if r1_eci is not None and v1_eci is not None:
        v2_eci = v1_eci + rel_vel
        x_e, y_e, z_e, P = build_encounter_frame(r1_eci, v1_eci, v2_eci)
    else:
        # Fallback encounter frame without primary orbital elements
        y_e = rel_vel / v_rel_mag
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(y_e, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        x_e = np.cross(ref, y_e)
        x_e /= np.linalg.norm(x_e)
        z_e = np.cross(x_e, y_e)
        z_e /= np.linalg.norm(z_e)
        P = np.vstack([x_e, z_e])

    # Projected miss vector in 2D encounter plane
    d_2d = P @ rel_pos

    # Combined 3D covariance
    C_3d = np.asarray(cov_primary_km2, dtype=float) + np.asarray(cov_secondary_km2, dtype=float)

    # Projected 2D covariance: C_2d = P * C_3d * P^T
    C_2d = P @ C_3d @ P.T

    # Symmetrize to prevent numerical asymmetry
    C_2d = 0.5 * (C_2d + C_2d.T)

    # Eigendecomposition to find principal uncertainty axes
    eigvals, eigvecs = np.linalg.eigh(C_2d)

    # Sort eigenvalues descending: lambda_x >= lambda_y > 0
    sort_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    # Floor eigenvalues to prevent non-positive definite issues
    var_x = max(float(eigvals[0]), 1e-12)
    var_y = max(float(eigvals[1]), 1e-12)

    sigma_x = np.sqrt(var_x)
    sigma_y = np.sqrt(var_y)

    # Rotation matrix to align encounter coordinates with principal axes
    R_rot = eigvecs.T
    d_principal = R_rot @ d_2d

    xm = float(d_principal[0])
    ym = float(d_principal[1])

    # Mahalanobis distance
    v_maha2 = (xm / sigma_x) ** 2 + (ym / sigma_y) ** 2
    maha_dist = float(np.sqrt(max(0.0, v_maha2)))

    # Orientation angle of principal ellipse
    angle_rad = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    angle_deg = float(np.degrees(angle_rad))

    return {
        "d_2d_km": d_2d,
        "cov_2d_km2": C_2d,
        "sigma_x_km": sigma_x,
        "sigma_y_km": sigma_y,
        "xm_km": xm,
        "ym_km": ym,
        "rotation_angle_deg": angle_deg,
        "mahalanobis_distance": maha_dist,
        "miss_distance_km": float(np.linalg.norm(rel_pos)),
        "relative_speed_km_s": v_rel_mag,
    }


def foster_collision_probability(
    xm_km: float,
    ym_km: float,
    sigma_x_km: float,
    sigma_y_km: float,
    hard_body_radius_m: float = 10.0,
    method: str = "quad",
    n_points: int = 64,
) -> float:
    """
    Calculate conjunction collision probability (Pc) using Foster (1992) numerical integration.

    Foster reduces the 2D bivariate Gaussian integral over a circular collision cross-section
    of radius R into a single 1D numerical integral across the major axis by evaluating the
    inner orthogonal integral analytically with the error function (erf):

        Pc = 1 / (2 * sqrt(2*pi) * sigma_x) *
             integral_{-R}^{R} exp(-(xm + u)^2 / (2 * sigma_x^2)) *
             [ erf((ym + sqrt(R^2 - u^2)) / (sqrt(2)*sigma_y)) -
               erf((ym - sqrt(R^2 - u^2)) / (sqrt(2)*sigma_y)) ] du

    Parameters
    ----------
    xm_km : float
        Miss distance component along the first principal axis (km).
    ym_km : float
        Miss distance component along the second principal axis (km).
    sigma_x_km : float
        Positional standard deviation along the first principal axis (km).
    sigma_y_km : float
        Positional standard deviation along the second principal axis (km).
    hard_body_radius_m : float
        Combined hard-body sphere radius of primary and secondary objects (meters).
        Default is 10.0 m.
    method : str
        Numerical quadrature method:
        - "quad": Adaptive Gauss-Kronrod quadrature via scipy.integrate.quad (high precision).
        - "gauss_legendre": Fast fixed-node Gauss-Legendre quadrature (ideal for real-time edge AI).
    n_points : int
        Number of quadrature nodes for "gauss_legendre" method (default: 64).

    Returns
    -------
    float
        Computed collision probability Pc in [0.0, 1.0].
    """
    r_km = float(hard_body_radius_m) / 1000.0
    if r_km <= 0.0:
        return 0.0

    # Positional uncertainty safeguards
    sx = max(float(sigma_x_km), 1e-6)
    sy = max(float(sigma_y_km), 1e-6)

    # Ensure sx >= sy for optimal integrand smoothness
    xm = float(xm_km)
    ym = float(ym_km)
    if sx < sy:
        sx, sy = sy, sx
        xm, ym = ym, xm

    # Screen extreme conjunctions where miss distance is far beyond uncertainty boundary
    v_maha = (xm / sx) ** 2 + (ym / sy) ** 2
    miss_dist = np.sqrt(xm**2 + ym**2)
    if v_maha > 200.0 and (miss_dist - r_km) > 12.0 * max(sx, sy):
        return 0.0

    sqrt2_sy = np.sqrt(2.0) * sy
    sqrt2pi_sx = np.sqrt(2.0 * np.pi) * sx

    if method == "quad":
        def integrand(u: float) -> float:
            x = xm + u
            y_span = np.sqrt(max(0.0, r_km**2 - u**2))
            t1 = (ym + y_span) / sqrt2_sy
            t2 = (ym - y_span) / sqrt2_sy
            return float(np.exp(-0.5 * (x / sx)**2) * 0.5 * (erf(t1) - erf(t2)))

        integral_val, _ = quad(integrand, -r_km, r_km, epsabs=1e-15, epsrel=1e-10, limit=100)
        prob = integral_val / sqrt2pi_sx

    elif method == "gauss_legendre":
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        u = r_km * nodes
        x = xm + u
        y_span = np.sqrt(np.maximum(0.0, r_km**2 - u**2))
        t1 = (ym + y_span) / sqrt2_sy
        t2 = (ym - y_span) / sqrt2_sy
        f_vals = np.exp(-0.5 * (x / sx)**2) * 0.5 * (erf(t1) - erf(t2))
        prob = float(np.sum(weights * f_vals) * r_km / sqrt2pi_sx)

    else:
        raise ValueError(f"Unknown integration method: '{method}'. Choose 'quad' or 'gauss_legendre'.")

    return float(np.clip(prob, 0.0, 1.0))


def chan_collision_probability(
    xm_km: float,
    ym_km: float,
    sigma_x_km: float,
    sigma_y_km: float,
    hard_body_radius_m: float = 10.0,
    max_order: int = 50,
    tol: float = 1e-15,
) -> float:
    """
    Calculate conjunction collision probability (Pc) using Chan (1997) analytical isotropic series.

    Chan's formulation evaluates the 2D collision probability by mapping the conjunction geometry
    to dimensionless parameters:
        u = R^2 / (sigma_x * sigma_y)   (scaled cross-sectional area)
        v = (xm/sigma_x)^2 + (ym/sigma_y)^2  (squared Mahalanobis distance)

    The series expansion is:
        Pc = exp(-v / 2) * sum_{m=0}^M [ (v / 2)^m / m! * (1 - exp(-u / 2) * sum_{k=0}^m (u / 2)^k / k!) ]

    Using the regularized lower incomplete gamma function identity:
        1 - exp(-u / 2) * sum_{k=0}^m (u / 2)^k / k! = gammainc(m + 1, u / 2)

    and Poisson weights:
        w_m = (v / 2)^m / m! * exp(-v / 2)

    the series becomes numerically exact, stable, and immune to catastrophic cancellation:
        Pc = sum_{m=0}^M w_m * gammainc(m + 1, u / 2)

    Parameters
    ----------
    xm_km : float
        Miss distance component along the first principal axis (km).
    ym_km : float
        Miss distance component along the second principal axis (km).
    sigma_x_km : float
        Positional standard deviation along the first principal axis (km).
    sigma_y_km : float
        Positional standard deviation along the second principal axis (km).
    hard_body_radius_m : float
        Combined hard-body sphere radius of primary and secondary objects (meters).
    max_order : int
        Maximum number of series terms (default: 50).
    tol : float
        Convergence tolerance for truncating the series (default: 1e-15).

    Returns
    -------
    float
        Computed collision probability Pc in [0.0, 1.0].
    """
    r_km = float(hard_body_radius_m) / 1000.0
    if r_km <= 0.0:
        return 0.0

    sx = max(float(sigma_x_km), 1e-6)
    sy = max(float(sigma_y_km), 1e-6)

    # Scaled parameters
    u = (r_km**2) / (sx * sy)
    v = (float(xm_km) / sx) ** 2 + (float(ym_km) / sy) ** 2

    if v > 200.0:
        return 0.0

    lam = 0.5 * v
    half_u = 0.5 * u

    # Initial Poisson weight w_0 = exp(-lam)
    w = float(np.exp(-lam)) if lam < 700.0 else 0.0
    total = 0.0

    for m in range(max_order):
        if m > 0:
            w = w * (lam / m)

        # Incomplete gamma evaluation: gammainc(a, x) = gamma(a, x) / Gamma(a)
        s_m = float(gammainc(m + 1, half_u))
        term = w * s_m
        total += term

        # Early termination when subsequent terms contribute negligibly
        if m > 3 and term < tol * total:
            break

    return float(np.clip(total, 0.0, 1.0))


def chan_zeroth_order_probability(
    xm_km: float,
    ym_km: float,
    sigma_x_km: float,
    sigma_y_km: float,
    hard_body_radius_m: float = 10.0,
) -> float:
    """
    Calculate Chan (1997) zeroth-order (m=0) analytical collision probability.

    Pc_0 = exp(-v / 2) * (1 - exp(-u / 2))
    where u = R^2 / (sigma_x * sigma_y) and v = (xm/sigma_x)^2 + (ym/sigma_y)^2.

    For small u << 1, this asymptotically approaches the point-estimate probability:
    Pc_0 ~ (pi * R^2) / (2 * pi * sigma_x * sigma_y) * exp(-v / 2).
    """
    r_km = float(hard_body_radius_m) / 1000.0
    if r_km <= 0.0:
        return 0.0

    sx = max(float(sigma_x_km), 1e-6)
    sy = max(float(sigma_y_km), 1e-6)

    u = (r_km**2) / (sx * sy)
    v = (float(xm_km) / sx) ** 2 + (float(ym_km) / sy) ** 2

    if v > 200.0:
        return 0.0

    prob = np.exp(-0.5 * v) * (1.0 - np.exp(-0.5 * u))
    return float(np.clip(prob, 0.0, 1.0))


def compute_maximum_collision_probability(
    miss_distance_km: float,
    hard_body_radius_m: float = 10.0,
) -> float:
    """
    Compute theoretical worst-case maximum collision probability (Max Pc)
    under arbitrary covariance scale factor (dilution limit).

    As established by Alfano (2005) and NASA CARA, if the positional uncertainty
    covariance is scaled by k^2 * C, Pc reaches a unique maximum when k^2 * sigma^2 = d^2 / 2:
        Max Pc = (R / d)^2 * exp(-1) = R^2 / (e * d^2)  for d > R.

    Parameters
    ----------
    miss_distance_km : float
        Miss distance between objects at TCA (km).
    hard_body_radius_m : float
        Combined hard body sphere radius (meters).

    Returns
    -------
    float
        Maximum attainable collision probability under worst-case covariance dilution [0, 1].
    """
    r_km = float(hard_body_radius_m) / 1000.0
    d_km = max(float(miss_distance_km), 0.0)

    if d_km <= r_km:
        return 1.0

    # Max Pc asymptotic upper bound
    pc_max = (r_km**2) / (np.e * (d_km**2))
    return float(np.clip(pc_max, 0.0, 1.0))


def compute_bplane_conjunction_risk(
    r1_eci: np.ndarray,
    v1_eci: np.ndarray,
    cov1_eci_km2: np.ndarray,
    r2_eci: np.ndarray,
    v2_eci: np.ndarray,
    cov2_eci_km2: np.ndarray,
    hard_body_radius_m: float = 10.0,
    critical_pc_threshold: float = 1e-4,
    method: str = "both",
) -> BPlaneConjunctionResult:
    """
    Perform end-to-end B-plane encounter projection and conjunction collision risk evaluation.

    Computes:
    - 3D-to-2D encounter frame projection (B-plane)
    - Diagonalization of combined covariance into principal uncertainty axes
    - Foster (1992) numerical integration Pc
    - Chan (1997) analytical series Pc
    - Maximum collision probability (Max Pc)
    - Conjunction risk alert status (NASA 1e-4 threshold)

    Parameters
    ----------
    r1_eci : np.ndarray
        Primary satellite position vector in ECI (km).
    v1_eci : np.ndarray
        Primary satellite velocity vector in ECI (km/s).
    cov1_eci_km2 : np.ndarray
        Primary satellite 3x3 position covariance matrix (km^2).
    r2_eci : np.ndarray
        Secondary object position vector in ECI (km).
    v2_eci : np.ndarray
        Secondary object velocity vector in ECI (km/s).
    cov2_eci_km2 : np.ndarray
        Secondary object 3x3 position covariance matrix (km^2).
    hard_body_radius_m : float
        Combined collision sphere radius (meters). Default is 10.0 m.
    critical_pc_threshold : float
        Threshold for high-risk conjunction (default: 1e-4 per NASA-STD-8719.14).
    method : str
        Calculation method: 'foster', 'chan', or 'both' (default: 'both').

    Returns
    -------
    BPlaneConjunctionResult
        Structured conjunction assessment result dataclass.
    """
    rel_pos = np.asarray(r2_eci, dtype=float) - np.asarray(r1_eci, dtype=float)
    rel_vel = np.asarray(v2_eci, dtype=float) - np.asarray(v1_eci, dtype=float)

    proj = project_to_encounter_plane(
        rel_pos_eci=rel_pos,
        rel_vel_eci=rel_vel,
        cov_primary_km2=cov1_eci_km2,
        cov_secondary_km2=cov2_eci_km2,
        r1_eci=r1_eci,
        v1_eci=v1_eci,
    )

    xm = proj["xm_km"]
    ym = proj["ym_km"]
    sx = proj["sigma_x_km"]
    sy = proj["sigma_y_km"]
    miss_dist = proj["miss_distance_km"]
    rel_speed = proj["relative_speed_km_s"]
    aspect_ratio = sx / sy if sy > 0 else 1.0

    # Compute Foster (1992) Pc
    if method in ("foster", "both"):
        pc_foster = foster_collision_probability(
            xm_km=xm,
            ym_km=ym,
            sigma_x_km=sx,
            sigma_y_km=sy,
            hard_body_radius_m=hard_body_radius_m,
            method="quad",
        )
    else:
        pc_foster = 0.0

    # Compute Chan (1997) Pc
    if method in ("chan", "both"):
        pc_chan = chan_collision_probability(
            xm_km=xm,
            ym_km=ym,
            sigma_x_km=sx,
            sigma_y_km=sy,
            hard_body_radius_m=hard_body_radius_m,
        )
    else:
        pc_chan = 0.0

    # If only one method requested, synchronize the other
    if method == "foster":
        pc_chan = pc_foster
    elif method == "chan":
        pc_foster = pc_chan

    # Compute Max Pc under covariance scaling
    pc_max = compute_maximum_collision_probability(
        miss_distance_km=miss_dist,
        hard_body_radius_m=hard_body_radius_m,
    )

    is_critical = bool(max(pc_foster, pc_chan) >= critical_pc_threshold)

    return BPlaneConjunctionResult(
        pc_foster=pc_foster,
        pc_chan=pc_chan,
        pc_max=pc_max,
        miss_distance_km=miss_dist,
        relative_speed_km_s=rel_speed,
        sigma_x_km=sx,
        sigma_y_km=sy,
        aspect_ratio=aspect_ratio,
        mahalanobis_distance=proj["mahalanobis_distance"],
        bplane_miss_coords_km=(float(proj["d_2d_km"][0]), float(proj["d_2d_km"][1])),
        principal_miss_coords_km=(xm, ym),
        rotation_angle_deg=proj["rotation_angle_deg"],
        combined_radius_m=hard_body_radius_m,
        is_critical=is_critical,
        method=method,
    )
