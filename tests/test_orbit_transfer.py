"""
Unit tests for the impulsive orbit transfer planner (Hohmann, bi-elliptic, plane change, phasing).
"""

import math

import pytest

from aegis_leo.constants import EARTH_RADIUS_KM
from aegis_leo.astrodynamics.orbit_transfer import (
    bielliptic_transfer,
    circular_velocity_km_s,
    combined_plane_change_transfer,
    hohmann_transfer,
    orbital_period_s,
    phasing_maneuver,
    plan_orbit_transfer,
    plane_change_delta_v_m_s,
    propellant_for_delta_v,
)


def test_hohmann_leo_to_geo_matches_textbook():
    sol = hohmann_transfer(EARTH_RADIUS_KM + 300.0, 42164.0)
    assert sol.total_delta_v_m_s == pytest.approx(3893.0, abs=5.0)
    assert sol.time_of_flight_s / 3600.0 == pytest.approx(5.27, abs=0.02)
    assert sol.burns_m_s[0] > sol.burns_m_s[1]


def test_hohmann_is_symmetric_and_zero_for_same_orbit():
    r1, r2 = EARTH_RADIUS_KM + 400.0, EARTH_RADIUS_KM + 800.0
    assert hohmann_transfer(r1, r2).total_delta_v_m_s == pytest.approx(hohmann_transfer(r2, r1).total_delta_v_m_s)
    assert hohmann_transfer(r1, r1).total_delta_v_m_s == pytest.approx(0.0, abs=1e-9)


def test_bielliptic_beats_hohmann_only_for_large_radius_ratio():
    r1 = 7000.0
    assert bielliptic_transfer(r1, 15.0 * r1, 40.0 * r1).total_delta_v_m_s < hohmann_transfer(r1, 15.0 * r1).total_delta_v_m_s
    assert bielliptic_transfer(r1, 2.0 * r1, 4.0 * r1).total_delta_v_m_s > hohmann_transfer(r1, 2.0 * r1).total_delta_v_m_s
    with pytest.raises(ValueError):
        bielliptic_transfer(r1, 2.0 * r1, 1.5 * r1)


def test_plane_change_formula_and_combined_saving():
    r = EARTH_RADIUS_KM + 550.0
    assert plane_change_delta_v_m_s(r, 60.0) == pytest.approx(circular_velocity_km_s(r) * 1000.0, rel=1e-9)
    r2 = EARTH_RADIUS_KM + 1200.0
    combined = combined_plane_change_transfer(r, r2, 5.0)
    separate = hohmann_transfer(r, r2).total_delta_v_m_s + plane_change_delta_v_m_s(r2, 5.0)
    assert combined.total_delta_v_m_s < separate
    d = combined.details
    assert d["plane_change_at_departure_deg"] + d["plane_change_at_arrival_deg"] == pytest.approx(5.0)
    assert d["plane_change_at_arrival_deg"] > d["plane_change_at_departure_deg"]


def test_phasing_period_and_direction():
    r = EARTH_RADIUS_KM + 600.0
    ahead = phasing_maneuver(r, 30.0, phasing_orbits=10)
    behind = phasing_maneuver(r, -30.0, phasing_orbits=10)
    assert ahead.time_of_flight_s == pytest.approx(orbital_period_s(r) * (10.0 - 30.0 / 360.0))
    assert ahead.details["phasing_sma_km"] < r < behind.details["phasing_sma_km"]
    assert phasing_maneuver(r, 30.0, 20).total_delta_v_m_s < ahead.total_delta_v_m_s
    with pytest.raises(ValueError):
        phasing_maneuver(r, 180.0, phasing_orbits=1)


def test_tsiolkovsky_propellant():
    assert propellant_for_delta_v(100.0, 250.0 * 9.80665 * math.log(2.0), 250.0) == pytest.approx(50.0)
    assert propellant_for_delta_v(100.0, 0.0, 250.0) == 0.0


def test_plan_orbit_transfer_report_and_feasibility():
    rep = plan_orbit_transfer(550.0, 600.0, thrust_n=0.08)
    assert rep.recommended_strategy == "HOHMANN"
    assert rep.total_delta_v_m_s == pytest.approx(27.22, abs=0.05)
    assert rep.feasible and rep.propellant_remaining_kg > 0.0
    assert all(d > 0.0 for d in rep.burn_durations_s)

    costly = plan_orbit_transfer(550.0, 600.0, delta_inclination_deg=2.0, phase_angle_deg=30.0, bielliptic_apogee_altitude_km=5000.0)
    assert costly.recommended_strategy == "COMBINED_PLANE_CHANGE_HOHMANN"
    assert not costly.feasible
    assert costly.separate_plane_change_penalty_m_s > 0.0
    assert costly.phasing is not None and len(costly.strategies) == 2
    assert "strategies" in costly.to_dict()

    with pytest.raises(ValueError):
        plan_orbit_transfer(50.0, 600.0)
