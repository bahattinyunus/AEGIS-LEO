"""
Unit tests for Universal Variable Lambert Solver and Multi-Debris ADR Tour Planner.
"""

import math
import numpy as np
import pytest

from aegis_leo.astrodynamics.lambert_tour import (
    compute_j2_raan_drift_rate_deg_day,
    plan_multi_debris_adr_tour,
    solve_lambert_universal,
    stumpff_c,
    stumpff_s,
)


def test_stumpff_functions_continuity():
    assert math.isclose(stumpff_c(0.0), 0.5, rel_tol=1e-7)
    assert math.isclose(stumpff_s(0.0), 1.0 / 6.0, rel_tol=1e-7)
    assert math.isclose(stumpff_c(1e-8), stumpff_c(-1e-8), rel_tol=1e-5)
    assert math.isclose(stumpff_s(1e-8), stumpff_s(-1e-8), rel_tol=1e-5)


def test_solve_lambert_universal_leo_transfer():
    r1 = np.array([7000.0, 0.0, 0.0])
    r2 = np.array([-6800.0, 2100.0, 400.0])
    sol = solve_lambert_universal(r1, r2, tof_s=2800.0)
    assert sol.converged is True
    assert sol.semi_major_axis_km > 6400.0
    assert 90.0 < sol.transfer_angle_deg < 270.0
    assert np.linalg.norm(sol.v1_km_s) > 5.0
    assert np.linalg.norm(sol.v2_km_s) > 5.0


def test_solve_lambert_invalid_tof():
    with pytest.raises(ValueError):
        solve_lambert_universal(np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7000.0, 0.0]), tof_s=-10.0)


def test_j2_raan_drift_rate_sign():
    # Prograde (<90 deg) has negative RAAN drift; retrograde (>90 deg) has positive RAAN drift (SSO ~ +0.9856 deg/day)
    rate_pro = compute_j2_raan_drift_rate_deg_day(550.0, 53.0)
    rate_sso = compute_j2_raan_drift_rate_deg_day(550.0, 97.6)
    assert rate_pro < 0.0
    assert math.isclose(rate_sso, 0.9856, rel_tol=0.05)


def test_plan_multi_debris_adr_tour():
    rep = plan_multi_debris_adr_tour(
        servicer_dry_mass_kg=450.0,
        initial_fuel_kg=180.0,
        isp_s=320.0,
        start_altitude_km=750.0,
        max_targets=4,
    )
    assert rep.targets_visited == 4
    assert rep.total_delta_v_m_s > 0.0
    assert 0.0 < rep.total_fuel_consumed_kg < rep.initial_fuel_kg
    assert rep.total_kessler_risk_removed > 30.0
    assert len(rep.legs) == 4
    d = rep.to_dict()
    assert "sample_lambert_transfer" in d
