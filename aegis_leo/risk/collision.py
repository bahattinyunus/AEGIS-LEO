"""
Orbital Conjunction Analysis & Collision Probability (Pc)
Implements 3D-to-2D B-Plane Encounter projection and Chan/Foster Pc approximation.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy.special import erf


def compute_relative_geometry(
    r1_eci: np.ndarray,
    v1_eci: np.ndarray,
    r2_eci: np.ndarray,
    v2_eci: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute relative position, relative velocity, and miss distance at Closest Point of Approach (TCA).
    """
    rel_pos = np.array(r2_eci - r1_eci, dtype=float)
    rel_vel = np.array(v2_eci - v1_eci, dtype=float)

    miss_distance_km = float(np.linalg.norm(rel_pos))
    rel_speed_km_s = float(np.linalg.norm(rel_vel))

    # Radial, In-Track, Cross-Track relative decomposition
    r_unit = r1_eci / np.linalg.norm(r1_eci)
    h_vec = np.cross(r1_eci, v1_eci)
    c_unit = h_vec / np.linalg.norm(h_vec)
    i_unit = np.cross(c_unit, r_unit)

    dr_radial = float(np.dot(rel_pos, r_unit))
    dr_intrack = float(np.dot(rel_pos, i_unit))
    dr_crosstrack = float(np.dot(rel_pos, c_unit))

    return {
        "miss_distance_km": miss_distance_km,
        "relative_speed_km_s": rel_speed_km_s,
        "radial_miss_km": dr_radial,
        "intrack_miss_km": dr_intrack,
        "crosstrack_miss_km": dr_crosstrack,
        "relative_pos_vector": rel_pos,
        "relative_vel_vector": rel_vel,
    }


def compute_collision_probability(
    miss_distance_km: float,
    combined_radius_m: float = 10.0,
    sigma_pos_km: float = 0.25,
) -> float:
    """
    Calculate 2D B-plane conjunction collision probability (Pc).
    Fast analytical model assuming isotropic Gaussian positional error covariance
    for edge-computation responsiveness.
    """
    if miss_distance_km <= 0.0:
        return 1.0

    r_comb_km = combined_radius_m / 1000.0
    sigma2 = max(sigma_pos_km**2, 1e-8)
    d2 = miss_distance_km**2

    # Standard 2D isotropic Gaussian integral approximation
    # Pc = exp(-d^2 / (2 * sigma^2)) * (1 - exp(-R^2 / (2 * sigma^2)))
    exp_d = np.exp(-0.5 * d2 / sigma2)
    exp_r = 1.0 - np.exp(-0.5 * (r_comb_km**2) / sigma2)
    p_c = float(exp_d * exp_r)

    return float(np.clip(p_c, 1e-12, 1.0))


def compute_covariance_weighted_pc(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    cov_primary: np.ndarray,
    cov_secondary: np.ndarray,
    hard_body_radius_m: float = 5.0,
) -> float:
    """
    Full 3D-to-2D encounter plane (B-plane) projection of combined covariance.
    cov_primary, cov_secondary: 3x3 position covariance matrices in km^2.
    """
    v_rel_mag = np.linalg.norm(rel_vel)
    if v_rel_mag < 1e-6:
        return compute_collision_probability(float(np.linalg.norm(rel_pos)), hard_body_radius_m)

    # Encounter frame unit vectors
    # y_b along relative velocity vector
    y_b = rel_vel / v_rel_mag

    # Arbitrary orthogonal vector
    dummy = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(y_b, dummy)) > 0.9:
        dummy = np.array([0.0, 1.0, 0.0])

    x_b = np.cross(y_b, dummy)
    x_b /= np.linalg.norm(x_b)
    z_b = np.cross(x_b, y_b)

    # Projection matrix P (2x3) projecting 3D position onto B-plane (x_b, z_b)
    P = np.row_stack([x_b, z_b])

    # Combined covariance in 3D
    C_comb = cov_primary + cov_secondary

    # Projected 2x2 covariance in B-plane
    C_2d = P @ C_comb @ P.T

    # Projected miss coordinates in B-plane
    d_2d = P @ rel_pos

    try:
        inv_C = np.linalg.inv(C_2d)
        det_C = float(np.linalg.det(C_2d))
        if det_C <= 0:
            raise np.linalg.LinAlgError

        # Mahalanobis distance squared
        maha2 = float(d_2d.T @ inv_C @ d_2d)

        # Hard body cross-section area in km^2
        r_hard_km = hard_body_radius_m / 1000.0
        area = np.pi * (r_hard_km**2)

        # 2D Gaussian density evaluated at miss vector * disk area
        prob = (1.0 / (2.0 * np.pi * np.sqrt(det_C))) * np.exp(-0.5 * maha2) * area
        return float(np.clip(prob, 1e-12, 1.0))
    except (np.linalg.LinAlgError, ValueError):
        # Fallback to scalar miss distance model
        return compute_collision_probability(float(np.linalg.norm(rel_pos)), hard_body_radius_m)
