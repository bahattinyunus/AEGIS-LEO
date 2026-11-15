"""
Unit tests for the Time of Closest Approach (TCA) finder & conjunction screening engine.
"""

import math

import numpy as np
import pytest

from aegis_leo.constants import EARTH_RADIUS_KM
from aegis_leo.risk.tca_finder import (
    OrbitalElements,
    build_intercepting_secondary,
    find_close_approaches,
    j2_secular_rates,
    propagate_elements,
    ric_frame,
    screen_conjunctions,
    solve_kepler,
)


def _primary(tca_s: float = 1500.0) -> OrbitalElements:
    p = OrbitalElements(altitude_km=550.0, inclination_deg=53.0, name="PRIMARY")
    p.mean_anomaly_deg = math.degrees(-p.mean_motion_rad_s * tca_s) % 360.0
    return p


def test_solve_kepler_satisfies_equation():
    for ecc in (0.0, 0.1, 0.7, 0.9):
        for m in (0.3, 2.0, 5.5):
            e_anom = solve_kepler(m, ecc)
            assert e_anom - ecc * math.sin(e_anom) == pytest.approx(m, abs=1e-10)


def test_propagation_is_periodic_and_conserves_radius():
    el = OrbitalElements(altitude_km=700.0, inclination_deg=98.0, eccentricity=0.01, mean_anomaly_deg=40.0)
    period = 2.0 * math.pi / el.mean_motion_rad_s
    r0, v0 = propagate_elements(el, 0.0, include_j2=False)
    r1, v1 = propagate_elements(el, period, include_j2=False)
    assert np.allclose(r0, r1, atol=1e-6) and np.allclose(v0, v1, atol=1e-9)
    radii = [np.linalg.norm(propagate_elements(el, t)[0]) for t in np.linspace(0.0, period, 50)]
    assert min(radii) >= el.perigee_radius_km - 1e-6 and max(radii) <= el.apogee_radius_km + 1e-6


def test_j2_rates_sun_synchronous_and_critical_inclination():
    sso = OrbitalElements(altitude_km=693.0, inclination_deg=98.18)
    raan_dot_deg_day = math.degrees(j2_secular_rates(sso)[0]) * 86400.0
    assert raan_dot_deg_day == pytest.approx(0.9856, abs=0.02)
    assert j2_secular_rates(OrbitalElements(550.0, 53.0))[0] < 0.0
    assert j2_secular_rates(OrbitalElements(550.0, 63.4349, eccentricity=0.01))[1] == pytest.approx(0.0, abs=1e-10)


def test_ric_frame_is_orthonormal():
    r, v = propagate_elements(_primary(), 321.0)
    frame = ric_frame(r, v)
    assert np.allclose(frame @ frame.T, np.eye(3), atol=1e-12)
    assert frame[0] @ r == pytest.approx(np.linalg.norm(r))


def test_engineered_conjunction_is_recovered():
    primary = _primary(1500.0)
    debris = build_intercepting_secondary(primary, 1500.0, inclination_deg=97.5, radial_offset_km=0.4)
    rep = screen_conjunctions(primary, debris, duration_hours=3.0, include_j2=False)
    assert rep.screening_status == "RED_CONJUNCTION_ALERT"
    assert rep.tca_of_min_miss_s == pytest.approx(1500.0, abs=0.01)
    assert rep.min_miss_distance_km == pytest.approx(0.4, abs=1e-4)
    closest = min(rep.events, key=lambda e: e["miss_distance_km"])
    assert closest["radial_miss_km"] == pytest.approx(0.4, abs=1e-4)
    assert closest["severity"] == "RED"
    assert closest["approach_angle_deg"] == pytest.approx(44.5, abs=0.01)
    assert 5.0 < closest["relative_speed_km_s"] < 6.5
    assert rep.events_within_threshold >= 1


def test_tca_is_a_true_local_minimum():
    primary = _primary(900.0)
    debris = build_intercepting_secondary(primary, 900.0, inclination_deg=80.0, radial_offset_km=2.0)
    events = find_close_approaches(primary, debris, 5400.0)
    assert events

    def sep(t):
        return np.linalg.norm(propagate_elements(debris, t)[0] - propagate_elements(primary, t)[0])

    for ev in events:
        assert ev.miss_distance_km <= sep(ev.tca_s - 0.5) + 1e-6
        assert ev.miss_distance_km <= sep(ev.tca_s + 0.5) + 1e-6


def test_shell_filter_rejects_separated_altitudes():
    rep = screen_conjunctions(OrbitalElements(550.0, 53.0), OrbitalElements(800.0, 53.0), duration_hours=2.0)
    assert rep.shell_filter_passed is False
    assert rep.radial_shell_gap_km == pytest.approx(250.0)
    assert rep.screening_status == "NO_CONJUNCTION_POSSIBLE_SHELL_FILTER"
    assert rep.events == [] and math.isinf(rep.min_miss_distance_km)


def test_clear_status_and_input_validation():
    primary = _primary(1500.0)
    debris = build_intercepting_secondary(primary, 1500.0, inclination_deg=97.5, radial_offset_km=0.4)
    debris.mean_anomaly_deg = (debris.mean_anomaly_deg + 20.0) % 360.0
    rep = screen_conjunctions(primary, debris, duration_hours=1.5, include_j2=False)
    assert rep.screening_status == "CLEAR"
    assert rep.min_miss_distance_km > 5.0
    assert primary.semi_major_axis_km == EARTH_RADIUS_KM + 550.0
    with pytest.raises(ValueError):
        find_close_approaches(primary, debris, -1.0)
    with pytest.raises(ValueError):
        build_intercepting_secondary(OrbitalElements(550.0, 53.0, eccentricity=0.1), 100.0, 97.0)


def test_engineered_conjunction_with_j2_drift():
    primary = OrbitalElements(altitude_km=700.0, inclination_deg=98.2, raan_deg=40.0, mean_anomaly_deg=75.0)
    debris = build_intercepting_secondary(primary, 4000.0, inclination_deg=65.0, radial_offset_km=-0.25, include_j2=True)
    rep = screen_conjunctions(primary, debris, duration_hours=2.0)
    assert rep.min_miss_distance_km == pytest.approx(0.25, abs=1e-3)
    assert rep.tca_of_min_miss_s == pytest.approx(4000.0, abs=0.05)
    closest = min(rep.events, key=lambda e: e["miss_distance_km"])
    assert closest["radial_miss_km"] == pytest.approx(-0.25, abs=1e-3)
    with pytest.raises(ValueError):
        build_intercepting_secondary(primary, 0.0, inclination_deg=5.0)
