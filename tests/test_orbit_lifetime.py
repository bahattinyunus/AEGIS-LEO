"""
Tests for Orbit Lifetime and Long-Term Orbital Decay Prediction Engine.
"""

import pytest
from aegis_leo.physics.orbit_lifetime import (
    SolarCycleParams,
    compute_scale_height_km,
    king_hele_analytical_lifetime,
    simulate_orbit_decay,
    evaluate_deorbit_compliance_matrix,
    DisposalRule,
)


def test_solar_cycle_flux_modulation():
    params = SolarCycleParams(cycle_period_years=11.0, current_phase_rad=0.0)
    # At t=0 (phase=0), sin(0)=0 -> solar minimum
    f_min, ap_min = params.sample_flux(0.0)
    assert 60.0 <= f_min <= 80.0

    # At t=2.75 years (phase = pi/2), solar max
    f_max, ap_max = params.sample_flux(2.75)
    assert f_max > f_min
    assert ap_max > ap_min


def test_scale_height_calculation():
    h200 = compute_scale_height_km(200.0)
    h400 = compute_scale_height_km(400.0)
    h700 = compute_scale_height_km(700.0)

    assert 30.0 < h200 < 45.0
    assert h400 > h200
    assert h700 > h400


def test_king_hele_analytical_lifetime_sensible():
    # Low altitude 250 km satellite should decay very rapidly (< 1 year)
    life_250 = king_hele_analytical_lifetime(
        altitude_km=250.0,
        mass_kg=200.0,
        drag_area_m2=2.5,
    )
    assert 0.001 < life_250 < 1.0

    # 600 km should have much longer lifetime
    life_600 = king_hele_analytical_lifetime(
        altitude_km=600.0,
        mass_kg=200.0,
        drag_area_m2=2.5,
    )
    assert life_600 > life_250 * 5.0


def test_simulate_orbit_decay_fcc_compliance():
    # 350 km orbit decays well under 5 years -> FCC 5-year compliant
    res_350 = simulate_orbit_decay(
        altitude_km=350.0,
        mass_kg=150.0,
        drag_area_m2=2.0,
        max_duration_years=10.0,
    )
    assert res_350.fcc_5year_compliant is True
    assert res_350.nasa_25year_compliant is True
    assert res_350.estimated_lifetime_years < 5.0
    assert len(res_350.altitude_series_km) > 1
    assert res_350.status == "REENTERED_COMPLIANT"

    d = res_350.to_dict()
    assert d["fcc_5year_compliant"] is True
    assert d["reentry_date_estimate_days"] > 0


def test_simulate_orbit_decay_high_altitude_non_compliant():
    # 850 km satellite without propulsion cannot deorbit within 25 years
    res_850 = simulate_orbit_decay(
        altitude_km=850.0,
        mass_kg=1000.0,
        drag_area_m2=1.0,
        max_duration_years=28.0,
    )
    assert res_850.fcc_5year_compliant is False
    assert res_850.nasa_25year_compliant is False


def test_evaluate_deorbit_compliance_matrix():
    alts = [300.0, 450.0, 600.0]
    matrix = evaluate_deorbit_compliance_matrix(alts, mass_kg=200.0, drag_area_m2=2.0)
    assert "sweep_records" in matrix
    assert len(matrix["sweep_records"]) == 3
    assert matrix["max_altitude_fcc_5yr_compliant_km"] >= 300.0
