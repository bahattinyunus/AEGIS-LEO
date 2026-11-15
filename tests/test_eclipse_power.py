"""Tests for Orbital Eclipse, Beta Angle & Solar/Battery Power Budget."""

import numpy as np
from aegis_leo.physics.eclipse_power import (
    compute_beta_angle_deg,
    compute_sso_inclination_deg,
    compute_sun_unit_vector,
    evaluate_shadow_state,
    simulate_eclipse_and_power,
)


def test_sso_inclination_range():
    inc_550 = compute_sso_inclination_deg(550.0)
    inc_800 = compute_sso_inclination_deg(800.0)
    assert 96.5 < inc_550 < 98.5
    assert inc_800 > inc_550


def test_dual_cone_shadow_states():
    sun_vec = np.array([1.0, 0.0, 0.0])
    # Directly in front of Earth -> SUNLIT
    s1, f1 = evaluate_shadow_state(np.array([6900.0, 0.0, 0.0]), sun_vec)
    assert s1 == "SUNLIT" and f1 == 1.0

    # Directly behind Earth -> UMBRA
    s2, f2 = evaluate_shadow_state(np.array([-6900.0, 0.0, 0.0]), sun_vec)
    assert s2 == "UMBRA" and f2 == 0.0


def test_beta_angle_bounds():
    sun_vec = compute_sun_unit_vector(172)
    beta = compute_beta_angle_deg(97.6, 15.0, sun_vec)
    assert -90.0 <= beta <= 90.0


def test_simulate_eclipse_and_power_cycle():
    rep = simulate_eclipse_and_power(
        altitude_km=550.0,
        inclination_deg=97.6,
        raan_deg=90.0,
        day_of_year=172,
        solar_array_area_m2=2.8,
        bus_load_w=150.0,
        battery_capacity_wh=450.0,
    )
    assert 5000.0 < rep.orbital_period_s < 6500.0
    assert 0.0 <= rep.eclipse_fraction <= 0.45
    assert rep.power_margin_positive is True
    assert 0.0 <= rep.max_dod_pct <= 50.0
    assert rep.temp_min_c < rep.temp_max_c
    assert len(rep.timeline) == 120
