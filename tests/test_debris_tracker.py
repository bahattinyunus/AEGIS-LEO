"""
Tests for aegis_leo.analytics.debris_tracker (v2.2.0)
"""

import math
import pytest
from aegis_leo.analytics.debris_tracker import (
    TrackedObject,
    ConjunctionEvent,
    DebrisFleetRiskMatrix,
    DebrisTrackerEngine,
    build_synthetic_catalogue,
    build_debris_risk_matrix,
    _relative_velocity_km_s,
    _secular_j2_raan_rate,
    _altitude_shell_overlap,
    _compute_conjunction_risk_score,
)
from aegis_leo.constants import EARTH_RADIUS_KM, EARTH_MU


# ---------------------------------------------------------------------------
# TrackedObject
# ---------------------------------------------------------------------------

def test_tracked_object_derived_fields():
    obj = TrackedObject(
        object_id="TEST-001",
        name="TestSat",
        altitude_km=550.0,
        inclination_deg=53.0,
    )
    expected_sma = EARTH_RADIUS_KM + 550.0
    assert obj.semi_major_axis_km == pytest.approx(expected_sma, rel=1e-6)
    assert obj.orbital_period_s > 0
    assert obj.mean_motion_deg_s > 0

    # Check orbital period for circular orbit at 550 km
    expected_period = 2.0 * math.pi * math.sqrt((expected_sma**3) / EARTH_MU)
    assert obj.orbital_period_s == pytest.approx(expected_period, rel=1e-6)


def test_tracked_object_to_dict():
    obj = TrackedObject("ID1", "SAT1", 550.0, 53.0)
    d = obj.to_dict()
    assert d["object_id"] == "ID1"
    assert d["altitude_km"] == 550.0
    assert "semi_major_axis_km" in d
    assert "orbital_period_s" in d


@pytest.mark.parametrize("alt", [160.0, 400.0, 550.0, 800.0, 1200.0])
def test_tracked_object_various_altitudes(alt):
    obj = TrackedObject("X", "X", alt, 45.0)
    assert obj.semi_major_axis_km == pytest.approx(EARTH_RADIUS_KM + alt, rel=1e-6)
    assert obj.mean_motion_deg_s > 0
    assert obj.orbital_period_s > 0


# ---------------------------------------------------------------------------
# Physics helper functions
# ---------------------------------------------------------------------------

def test_relative_velocity_co_orbital():
    """Same orbit plane → relative speed close to 0 for circular co-orbital."""
    a = TrackedObject("A", "A", 550.0, 53.0)
    b = TrackedObject("B", "B", 550.0, 53.0)
    v_rel = _relative_velocity_km_s(a, b)
    # Should be near 0 but function returns max(result, 0.01)
    assert v_rel >= 0.01


def test_relative_velocity_polar_crossing():
    """Near-polar and near-equatorial → relative speed near circular velocity."""
    a = TrackedObject("A", "A", 550.0, 0.0)
    b = TrackedObject("B", "B", 550.0, 90.0)
    v_rel = _relative_velocity_km_s(a, b)
    v_circ = math.sqrt(EARTH_MU / (EARTH_RADIUS_KM + 550.0))
    # Should be between v_circ and 2*v_circ
    assert v_circ * 0.5 < v_rel < v_circ * 2.5


def test_secular_j2_raan_rate_prograde():
    """Prograde orbit should have negative (westward) RAAN drift."""
    obj = TrackedObject("X", "X", 550.0, 53.0)
    rate = _secular_j2_raan_rate(obj)
    assert rate < 0.0  # westward precession for prograde


def test_secular_j2_raan_rate_retrograde_sso():
    """Sun-synchronous retrograde orbit (~98°) should have positive (eastward) drift."""
    obj = TrackedObject("X", "X", 550.0, 97.6)
    rate = _secular_j2_raan_rate(obj)
    assert rate > 0.0


def test_altitude_shell_overlap_yes():
    a = TrackedObject("A", "A", 550.0, 53.0)
    b = TrackedObject("B", "B", 560.0, 53.0)
    assert _altitude_shell_overlap(a, b) is True


def test_altitude_shell_overlap_no():
    a = TrackedObject("A", "A", 400.0, 53.0)
    b = TrackedObject("B", "B", 800.0, 53.0)
    assert _altitude_shell_overlap(a, b) is False


def test_altitude_shell_overlap_boundary():
    a = TrackedObject("A", "A", 550.0, 53.0)
    b = TrackedObject("B", "B", 601.0, 53.0)  # just over 50 km
    assert _altitude_shell_overlap(a, b) is False


def test_conjunction_risk_score_close_approach():
    a = TrackedObject("A", "A", 550.0, 53.0, mass_kg=260.0, area_m2=2.2)
    b = TrackedObject("B", "B", 550.5, 97.0, mass_kg=5.0, area_m2=0.1)
    score = _compute_conjunction_risk_score(0.05, 11.5, a, b)
    assert 0.0 <= score <= 1.0
    assert score > 0.4  # close miss should produce significant risk


def test_conjunction_risk_score_distant():
    a = TrackedObject("A", "A", 550.0, 53.0)
    b = TrackedObject("B", "B", 550.0, 53.0)
    score = _compute_conjunction_risk_score(50.0, 1.0, a, b)
    assert score < 0.5  # large miss distance → lower risk


# ---------------------------------------------------------------------------
# DebrisTrackerEngine
# ---------------------------------------------------------------------------

def test_engine_add_and_list():
    engine = DebrisTrackerEngine()
    obj = TrackedObject("SAT-1", "PrimaryTest", 550.0, 53.0, is_active_satellite=True)
    engine.add_object(obj)
    assert engine.catalogue_size == 1
    objects = engine.list_objects()
    assert len(objects) == 1
    assert objects[0].object_id == "SAT-1"


def test_engine_remove():
    engine = DebrisTrackerEngine()
    obj = TrackedObject("SAT-1", "PrimaryTest", 550.0, 53.0)
    engine.add_object(obj)
    removed = engine.remove_object("SAT-1")
    assert removed is True
    assert engine.catalogue_size == 0


def test_engine_remove_nonexistent():
    engine = DebrisTrackerEngine()
    removed = engine.remove_object("GHOST")
    assert removed is False


def test_engine_risk_matrix_basic():
    primary = TrackedObject("PRIMARY", "MainSat", 550.0, 53.0, is_active_satellite=True)
    engine = build_synthetic_catalogue(primary, num_debris=20)
    matrix = engine.compute_risk_matrix(primary_id="PRIMARY", horizon_hours=6.0)
    assert isinstance(matrix, DebrisFleetRiskMatrix)
    assert matrix.primary_satellite_id == "PRIMARY"
    assert matrix.total_tracked_objects >= 21  # 1 primary + 20 debris
    assert matrix.total_conjunctions_screened == 20


def test_engine_risk_matrix_missing_primary():
    engine = DebrisTrackerEngine()
    with pytest.raises(KeyError):
        engine.compute_risk_matrix(primary_id="NONEXISTENT")


def test_engine_risk_matrix_to_dict():
    primary = TrackedObject("P1", "Sat", 550.0, 53.0, is_active_satellite=True)
    engine = build_synthetic_catalogue(primary, num_debris=10)
    matrix = engine.compute_risk_matrix("P1")
    d = matrix.to_dict()
    assert isinstance(d, dict)
    assert "timestamp_utc" in d
    assert "conjunction_events" in d
    assert "summary" in d


def test_engine_top_threats_sorted():
    primary = TrackedObject("P2", "Sat", 550.0, 53.0, is_active_satellite=True)
    engine = build_synthetic_catalogue(primary, num_debris=30)
    matrix = engine.compute_risk_matrix("P2")
    if len(matrix.conjunction_events) >= 3:
        top3 = matrix.top_threats(3)
        for i in range(len(top3) - 1):
            assert top3[i].risk_score >= top3[i + 1].risk_score


def test_engine_secondary_chain_detection():
    """With a large catalogue, secondary chain detection should run without error."""
    primary = TrackedObject("P3", "ChainTest", 550.0, 53.0, is_active_satellite=True)
    engine = build_synthetic_catalogue(primary, num_debris=40)
    matrix = engine.compute_risk_matrix("P3", detect_secondary_chains=True)
    # secondary_chain_risk is a bool
    assert isinstance(matrix.secondary_chain_risk, bool)


# ---------------------------------------------------------------------------
# build_debris_risk_matrix convenience function
# ---------------------------------------------------------------------------

def test_build_debris_risk_matrix_nominal():
    matrix = build_debris_risk_matrix(
        satellite_id="CONV-TEST",
        altitude_km=550.0,
        inclination_deg=53.0,
        num_debris=30,
        horizon_hours=6.0,
    )
    assert matrix.primary_satellite_id == "CONV-TEST"
    assert matrix.total_tracked_objects >= 31
    assert matrix.highest_risk_score >= 0.0
    assert matrix.min_miss_distance_km >= 0.0


def test_build_debris_risk_matrix_various_altitudes():
    for alt in [300.0, 550.0, 800.0]:
        matrix = build_debris_risk_matrix("SAT", alt, 53.0, num_debris=15)
        assert isinstance(matrix, DebrisFleetRiskMatrix)


def test_build_debris_risk_matrix_counts_consistent():
    matrix = build_debris_risk_matrix("S1", 550.0, 53.0, num_debris=50)
    assert matrix.critical_conjunctions + matrix.warning_conjunctions <= len(matrix.conjunction_events)
