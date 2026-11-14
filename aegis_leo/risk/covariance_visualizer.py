"""
3D Covariance Ellipsoid Mesh Generator
Generates 1-sigma, 2-sigma, and 3-sigma spatial uncertainty ellipsoids in RTN frame
for 3D collision conjunction visualization.
"""

from typing import Dict, Any, Tuple
import numpy as np


def generate_covariance_ellipsoid_mesh(
    center_pos: np.ndarray,
    sigma_radial_km: float = 0.20,
    sigma_intrack_km: float = 0.85,
    sigma_crosstrack_km: float = 0.30,
    num_sigma: float = 2.0,
    grid_resolution: int = 25,
) -> Dict[str, Any]:
    """
    Generate parametric 3D surface coordinates for uncertainty ellipsoid.
    """
    u = np.linspace(0, 2.0 * np.pi, grid_resolution)
    v = np.linspace(0, np.pi, grid_resolution)

    # Parametric sphere
    x_unit = np.outer(np.cos(u), np.sin(v))
    y_unit = np.outer(np.sin(u), np.sin(v))
    z_unit = np.outer(np.ones(np.size(u)), np.cos(v))

    # Scale by sigmas
    a = num_sigma * sigma_radial_km
    b = num_sigma * sigma_intrack_km
    c = num_sigma * sigma_crosstrack_km

    x_mesh = center_pos[0] + a * x_unit
    y_mesh = center_pos[1] + b * y_unit
    z_mesh = center_pos[2] + c * z_unit

    volume_km3 = (4.0 / 3.0) * np.pi * a * b * c

    return {
        "x": x_mesh.tolist(),
        "y": y_mesh.tolist(),
        "z": z_mesh.tolist(),
        "semi_axes_km": {"a_radial": round(a, 3), "b_intrack": round(b, 3), "c_crosstrack": round(c, 3)},
        "uncertainty_volume_km3": round(float(volume_km3), 4),
        "sigma_level": num_sigma,
    }
