"""
Tests for Foster (1992) B-Plane Numerical Integration and Chan (1997) Analytical Series.
"""

import pytest
import numpy as np
from aegis_leo.risk.foster_bplane import (
    build_encounter_frame,
    project_to_encounter_plane,
    foster_collision_probability,
    chan_collision_probability,
    chan_zeroth_order_probability,
    compute_maximum_collision_probability,
    compute_bplane_conjunction_risk,
    BPlaneConjunctionResult,
)


def test_encounter_frame_orthonormality():
    """Verify encounter frame forms an orthonormal right-handed basis."""
    r1 = np.array([7000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7.5, 0.0])
    v2 = np.array([0.0, 0.0, 7.5])  # Conjunction at 90 deg inclination

    x_e, y_e, z_e, P = build_encounter_frame(r1, v1, v2)

    # Unit lengths
    assert np.isclose(np.linalg.norm(x_e), 1.0, atol=1e-10)
    assert np.isclose(np.linalg.norm(y_e), 1.0, atol=1e-10)
    assert np.isclose(np.linalg.norm(z_e), 1.0, atol=1e-10)

    # Mutual orthogonality
    assert np.isclose(np.dot(x_e, y_e), 0.0, atol=1e-10)
    assert np.isclose(np.dot(y_e, z_e), 0.0, atol=1e-10)
    assert np.isclose(np.dot(x_e, z_e), 0.0, atol=1e-10)

    # Right-handed coordinate frame
    det = np.linalg.det(np.column_stack([x_e, y_e, z_e]))
    assert np.isclose(det, 1.0, atol=1e-10)

    # 2x3 projection matrix shape and row orthogonality
    assert P.shape == (2, 3)
    assert np.isclose(np.dot(P[0], P[1]), 0.0, atol=1e-10)


def test_encounter_frame_fallback_when_collinear():
    """Verify encounter frame gracefully handles collinear relative velocity and orbit normal."""
    r1 = np.array([7000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7.5, 0.0])
    # Primary orbit normal is along Z: h1 = r1 x v1 = [0, 0, 7000*7.5]
    # Secondary velocity such that v_rel is along Z (collinear with h1)
    v2 = np.array([0.0, 7.5, 5.0])

    x_e, y_e, z_e, P = build_encounter_frame(r1, v1, v2)

    assert np.isclose(np.linalg.norm(x_e), 1.0, atol=1e-10)
    assert np.isclose(np.dot(x_e, y_e), 0.0, atol=1e-10)
    assert np.isclose(np.dot(y_e, z_e), 0.0, atol=1e-10)


def test_projection_mahalanobis_invariance():
    """Verify that diagonalization preserves Mahalanobis distance."""
    rel_pos = np.array([0.2, 0.1, -0.15])
    rel_vel = np.array([-5.0, 4.0, 3.0])

    # 3x3 positive definite covariances
    cov1 = np.diag([0.04, 0.09, 0.02])
    cov2 = np.diag([0.05, 0.03, 0.06])

    proj = project_to_encounter_plane(rel_pos, rel_vel, cov1, cov2)

    # Direct 2D Mahalanobis distance from projected matrix
    d_2d = proj["d_2d_km"]
    C_2d = proj["cov_2d_km2"]
    direct_maha2 = float(d_2d.T @ np.linalg.inv(C_2d) @ d_2d)
    direct_maha = np.sqrt(direct_maha2)

    assert np.isclose(proj["mahalanobis_distance"], direct_maha, rtol=1e-8)
    assert proj["sigma_x_km"] >= proj["sigma_y_km"] > 0.0


def test_foster_and_chan_isotropic_agreement():
    """
    Foster (1992) and Chan (1997) should yield near-identical results
    for isotropic covariance across various miss distances.
    """
    sigma = 0.5  # 500 m
    radius_m = 10.0  # 10 m

    miss_distances = [0.0, 0.1, 0.25, 0.5, 1.0]

    for d in miss_distances:
        p_foster = foster_collision_probability(
            xm_km=d,
            ym_km=0.0,
            sigma_x_km=sigma,
            sigma_y_km=sigma,
            hard_body_radius_m=radius_m,
            method="quad",
        )
        p_chan = chan_collision_probability(
            xm_km=d,
            ym_km=0.0,
            sigma_x_km=sigma,
            sigma_y_km=sigma,
            hard_body_radius_m=radius_m,
        )

        assert 0.0 <= p_foster <= 1.0
        assert 0.0 <= p_chan <= 1.0

        # Relative agreement between Foster and Chan in isotropic case should be extremely tight
        if p_foster > 1e-12:
            rel_diff = abs(p_foster - p_chan) / p_foster
            assert rel_diff < 1e-6, f"Mismatch at miss distance {d}: Foster={p_foster}, Chan={p_chan}"


def test_foster_quad_vs_gauss_legendre():
    """Verify adaptive quad and fixed-node Gauss-Legendre yield consistent results."""
    xm = 0.2
    ym = -0.15
    sx = 1.2
    sy = 0.4
    r_m = 15.0

    p_quad = foster_collision_probability(xm, ym, sx, sy, r_m, method="quad")
    p_gl32 = foster_collision_probability(xm, ym, sx, sy, r_m, method="gauss_legendre", n_points=32)
    p_gl64 = foster_collision_probability(xm, ym, sx, sy, r_m, method="gauss_legendre", n_points=64)

    assert np.isclose(p_quad, p_gl64, rtol=1e-5)
    assert np.isclose(p_gl32, p_gl64, rtol=1e-4)


def test_collision_probability_monotonicity():
    """Pc should decrease monotonically as miss distance increases."""
    sx, sy = 0.8, 0.4
    r_m = 10.0

    d_list = [0.05, 0.2, 0.5, 1.0, 2.0]
    p_foster_vals = [foster_collision_probability(d, 0.0, sx, sy, r_m) for d in d_list]
    p_chan_vals = [chan_collision_probability(d, 0.0, sx, sy, r_m) for d in d_list]

    for i in range(len(d_list) - 1):
        assert p_foster_vals[i] > p_foster_vals[i + 1]
        assert p_chan_vals[i] > p_chan_vals[i + 1]


def test_collision_probability_scales_with_hard_body_radius():
    """Pc should increase monotonically with larger collision sphere radius."""
    sx, sy = 0.5, 0.3
    d = 0.2

    r_small = 2.0
    r_medium = 10.0
    r_large = 50.0

    p_small = foster_collision_probability(d, 0.0, sx, sy, r_small)
    p_med = foster_collision_probability(d, 0.0, sx, sy, r_medium)
    p_large = foster_collision_probability(d, 0.0, sx, sy, r_large)

    assert p_small < p_med < p_large


def test_edge_cases():
    """Verify zero radius, extreme miss distance, and head-on collision."""
    sx, sy = 0.5, 0.5

    # Zero radius
    assert foster_collision_probability(0.1, 0.0, sx, sy, hard_body_radius_m=0.0) == 0.0
    assert chan_collision_probability(0.1, 0.0, sx, sy, hard_body_radius_m=0.0) == 0.0

    # Negative radius
    assert foster_collision_probability(0.1, 0.0, sx, sy, hard_body_radius_m=-5.0) == 0.0
    assert chan_collision_probability(0.1, 0.0, sx, sy, hard_body_radius_m=-5.0) == 0.0

    # Extreme miss distance (15 km with 0.5 km sigma -> ~30 sigma)
    assert foster_collision_probability(15.0, 0.0, sx, sy, hard_body_radius_m=10.0) == 0.0
    assert chan_collision_probability(15.0, 0.0, sx, sy, hard_body_radius_m=10.0) == 0.0

    # Head-on collision (miss distance = 0)
    p_headon_foster = foster_collision_probability(0.0, 0.0, sx, sy, hard_body_radius_m=10.0)
    p_headon_chan = chan_collision_probability(0.0, 0.0, sx, sy, hard_body_radius_m=10.0)
    assert p_headon_foster > 0.0
    assert np.isclose(p_headon_foster, p_headon_chan, rtol=1e-8)


def test_chan_zeroth_order_approximation():
    """Verify Chan zeroth-order approximation matches series for small u."""
    sx, sy = 1.0, 1.0
    r_m = 5.0  # 5 m / 1000 m -> u = 2.5e-5 (very small)
    xm, ym = 0.2, 0.1

    p_full = chan_collision_probability(xm, ym, sx, sy, r_m)
    p_zero = chan_zeroth_order_probability(xm, ym, sx, sy, r_m)

    assert np.isclose(p_full, p_zero, rtol=1e-4)


def test_maximum_collision_probability_bounds():
    """Verify Max Pc upper bound property."""
    r_m = 10.0
    d_km = 0.2

    pc_max = compute_maximum_collision_probability(d_km, r_m)
    assert 0.0 < pc_max <= 1.0

    # For any sigma, computed Pc must not exceed theoretical Max Pc
    for sigma in [0.05, 0.1, 0.14, 0.2, 0.5, 1.0, 2.0]:
        pc = foster_collision_probability(d_km, 0.0, sigma, sigma, r_m)
        assert pc <= pc_max * 1.001, f"Pc {pc} exceeded Max Pc {pc_max} at sigma {sigma}"

    # For d <= R, Max Pc is 1.0
    assert compute_maximum_collision_probability(0.005, 10.0) == 1.0


def test_compute_bplane_conjunction_risk_pipeline():
    """Test full end-to-end BPlane conjunction risk assessment."""
    r1 = np.array([7000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7.5, 0.0])
    cov1 = np.eye(3) * 0.01  # 100 m 1-sigma

    # Secondary with 150 m miss distance at high relative speed
    r2 = np.array([7000.1, 0.0, 0.1])
    v2 = np.array([0.0, -7.5, 0.0])  # Head-on 15 km/s relative speed
    cov2 = np.eye(3) * 0.02

    res = compute_bplane_conjunction_risk(
        r1_eci=r1,
        v1_eci=v1,
        cov1_eci_km2=cov1,
        r2_eci=r2,
        v2_eci=v2,
        cov2_eci_km2=cov2,
        hard_body_radius_m=10.0,
        critical_pc_threshold=1e-4,
    )

    assert isinstance(res, BPlaneConjunctionResult)
    assert res.relative_speed_km_s > 14.0
    assert 0.0 < res.pc_foster < 1.0
    assert 0.0 < res.pc_chan < 1.0
    assert res.miss_distance_km > 0.1
    assert res.sigma_x_km > 0.0
    assert res.sigma_y_km > 0.0

    d_out = res.to_dict()
    assert "pc_foster" in d_out
    assert "pc_chan" in d_out
    assert "is_critical" in d_out
    assert isinstance(d_out["bplane_miss_coords_km"], list)
