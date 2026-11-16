"""
Tests for aegis_leo.analytics.mission_analytics (v2.2.0)
"""

import pytest
from aegis_leo.analytics.mission_analytics import (
    MissionAnalyticsEngine,
    MissionHealthSnapshot,
    SubsystemTrendReport,
    analyze_mission_health,
    HealthGrade,
    TrendDirection,
    MissionStage,
    AnomalySeverity,
    _grade_from_score,
    _trend_from_history,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nominal_telemetry():
    return {
        "satellite_name": "TEST-SAT",
        "altitude_km": 550.0,
        "mass_kg": 260.0,
        "fuel_remaining_kg": 15.0,
        "isp_s": 250.0,
        "kp": 1.5,
        "f10_7": 70.0,
        "adri_score": 0.10,
        "collision_pc": 1e-8,
        "power_margin_positive": True,
        "dod_pct": 40.0,
        "beta_deg": 20.0,
        "all_thermal_ok": True,
        "peak_aero_w": 0.5,
        "total_passes": 8,
        "total_data_mb": 400.0,
        "tid_krad_yr": 1.0,
        "seu_per_day": 0.05,
        "scrub_s": 300.0,
        "pqc_authentic": True,
        "pqc_replay_ok": True,
        "pqc_tamper_ok": True,
        "mission_elapsed_years": 1.0,
        "component_tid_tolerance_krad": 30.0,
    }


@pytest.fixture
def storm_telemetry():
    return {
        "satellite_name": "TEST-SAT",
        "altitude_km": 550.0,
        "mass_kg": 260.0,
        "fuel_remaining_kg": 2.0,   # critically low
        "isp_s": 250.0,
        "kp": 8.5,
        "f10_7": 265.0,
        "adri_score": 0.85,          # critical risk
        "collision_pc": 2e-3,
        "power_margin_positive": False,  # negative power margin
        "dod_pct": 92.0,
        "beta_deg": 5.0,
        "all_thermal_ok": False,
        "peak_aero_w": 25.0,
        "total_passes": 0,            # no comms
        "total_data_mb": 0.0,
        "tid_krad_yr": 12.0,          # above 10 krad/yr limit
        "seu_per_day": 8.0,
        "scrub_s": 60.0,
        "pqc_authentic": False,       # auth failure
        "pqc_replay_ok": False,
        "pqc_tamper_ok": False,
        "mission_elapsed_years": 3.0,
        "component_tid_tolerance_krad": 30.0,
    }


# ---------------------------------------------------------------------------
# _grade_from_score
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("score,expected", [
    (95.0, HealthGrade.EXCELLENT),
    (80.0, HealthGrade.NOMINAL),
    (60.0, HealthGrade.CAUTION),
    (40.0, HealthGrade.WARNING),
    (10.0, HealthGrade.CRITICAL),
    (0.0,  HealthGrade.CRITICAL),
    (100.0, HealthGrade.EXCELLENT),
    (70.0, HealthGrade.NOMINAL),
    (50.0, HealthGrade.CAUTION),
    (30.0, HealthGrade.WARNING),
])
def test_grade_from_score(score, expected):
    assert _grade_from_score(score) == expected


# ---------------------------------------------------------------------------
# _trend_from_history
# ---------------------------------------------------------------------------

def test_trend_stable():
    history = [70.0, 70.5, 69.8, 70.2, 70.1]
    trend, slope = _trend_from_history(history)
    assert trend == TrendDirection.STABLE
    assert abs(slope) < 2.0


def test_trend_improving():
    history = [50.0, 55.0, 60.0, 65.0, 70.0]
    trend, slope = _trend_from_history(history)
    assert trend == TrendDirection.IMPROVING
    assert slope > 0


def test_trend_degrading():
    history = [90.0, 85.0, 78.0, 70.0, 60.0]
    trend, slope = _trend_from_history(history)
    assert trend == TrendDirection.DEGRADING
    assert slope < 0


def test_trend_volatile():
    history = [90.0, 20.0, 85.0, 15.0, 95.0, 10.0]
    trend, slope = _trend_from_history(history)
    assert trend == TrendDirection.VOLATILE


def test_trend_too_short():
    trend, slope = _trend_from_history([70.0, 65.0])
    assert trend == TrendDirection.STABLE
    assert slope == 0.0


# ---------------------------------------------------------------------------
# analyze_mission_health — nominal conditions
# ---------------------------------------------------------------------------

def test_analyze_nominal_returns_snapshot(nominal_telemetry):
    snap = analyze_mission_health("TEST-SAT", nominal_telemetry, initial_fuel_kg=20.0)
    assert isinstance(snap, MissionHealthSnapshot)
    assert snap.satellite_id == "TEST-SAT"
    assert 0.0 <= snap.overall_health_score <= 100.0
    assert snap.overall_grade in list(HealthGrade)
    assert snap.mission_stage in list(MissionStage)


def test_analyze_nominal_no_critical_anomalies(nominal_telemetry):
    snap = analyze_mission_health("TEST-SAT", nominal_telemetry, initial_fuel_kg=20.0)
    critical = [a for a in snap.anomalies if a.severity == AnomalySeverity.CRITICAL]
    assert len(critical) == 0, f"Unexpected critical anomalies: {critical}"


def test_analyze_nominal_high_health(nominal_telemetry):
    snap = analyze_mission_health("TEST-SAT", nominal_telemetry, initial_fuel_kg=20.0)
    assert snap.overall_health_score >= 70.0


def test_analyze_nominal_has_all_subsystems(nominal_telemetry):
    snap = analyze_mission_health("TEST-SAT", nominal_telemetry, initial_fuel_kg=20.0)
    for name in ["PROPULSION", "COLLISION_RISK", "POWER", "THERMAL", "COMMS", "RADIATION", "SECURITY"]:
        assert name in snap.subsystems


def test_analyze_nominal_fuel_projection(nominal_telemetry):
    snap = analyze_mission_health("TEST-SAT", nominal_telemetry, initial_fuel_kg=20.0)
    fp = snap.fuel_projection
    assert fp is not None
    assert fp.current_propellant_kg == pytest.approx(15.0, abs=0.1)
    assert fp.days_remaining_nominal > 0
    assert fp.days_remaining_worst_case > 0
    assert 0.0 <= fp.fuel_fraction_pct <= 100.0


def test_analyze_nominal_radiation_projection(nominal_telemetry):
    snap = analyze_mission_health("TEST-SAT", nominal_telemetry, initial_fuel_kg=20.0)
    rp = snap.radiation_projection
    assert rp is not None
    assert rp.tid_accumulated_krad >= 0
    assert rp.years_to_tid_limit > 0


def test_analyze_nominal_kpi_summary(nominal_telemetry):
    snap = analyze_mission_health("TEST-SAT", nominal_telemetry, initial_fuel_kg=20.0)
    kpi = snap.kpi_summary
    assert "overall_health_pct" in kpi
    assert "risk_level" in kpi
    assert "fuel_days_remaining" in kpi
    assert kpi["risk_level"] == "NOMINAL"


def test_analyze_nominal_to_dict(nominal_telemetry):
    snap = analyze_mission_health("TEST-SAT", nominal_telemetry, initial_fuel_kg=20.0)
    d = snap.to_dict()
    assert isinstance(d, dict)
    assert "overall_health_score" in d
    assert "subsystems" in d
    assert "anomalies" in d
    assert "fuel_projection" in d


# ---------------------------------------------------------------------------
# analyze_mission_health — storm / failure conditions
# ---------------------------------------------------------------------------

def test_analyze_storm_critical_anomalies(storm_telemetry):
    snap = analyze_mission_health("TEST-SAT-STORM", storm_telemetry, initial_fuel_kg=20.0)
    critical = [a for a in snap.anomalies if a.severity == AnomalySeverity.CRITICAL]
    assert len(critical) >= 3, f"Expected ≥3 critical anomalies, got {len(critical)}"


def test_analyze_storm_low_health(storm_telemetry):
    snap = analyze_mission_health("TEST-SAT-STORM", storm_telemetry, initial_fuel_kg=20.0)
    assert snap.overall_health_score < 60.0


def test_analyze_storm_risk_level_critical(storm_telemetry):
    snap = analyze_mission_health("TEST-SAT-STORM", storm_telemetry, initial_fuel_kg=20.0)
    assert snap.kpi_summary["risk_level"] in ("CRITICAL", "HIGH")


def test_analyze_storm_security_subsystem_flagged(storm_telemetry):
    snap = analyze_mission_health("TEST-SAT-STORM", storm_telemetry, initial_fuel_kg=20.0)
    security_score = snap.subsystems["SECURITY"].health_score
    assert security_score < 50.0


def test_analyze_storm_radiation_subsystem_flagged(storm_telemetry):
    snap = analyze_mission_health("TEST-SAT-STORM", storm_telemetry, initial_fuel_kg=20.0)
    rad_score = snap.subsystems["RADIATION"].health_score
    assert rad_score < 70.0


# ---------------------------------------------------------------------------
# MissionAnalyticsEngine — stateful trend tracking
# ---------------------------------------------------------------------------

def test_engine_trends_after_multiple_snapshots(nominal_telemetry):
    engine = MissionAnalyticsEngine("E1", initial_fuel_kg=20.0)
    for _ in range(5):
        engine.ingest_snapshot(nominal_telemetry)
    reports = engine.get_subsystem_trend_reports()
    assert len(reports) == 7
    for r in reports:
        assert isinstance(r, SubsystemTrendReport)
        assert r.trend in list(TrendDirection)
        assert len(r.history) >= 1


def test_engine_degrading_trend_detected():
    engine = MissionAnalyticsEngine("E2", initial_fuel_kg=20.0)
    for i in range(8):
        # Progressively more critical telemetry
        t = {
            "altitude_km": 550.0,
            "mass_kg": 260.0,
            "fuel_remaining_kg": max(0.5, 15.0 - i * 2.0),
            "kp": 1.5 + i * 0.5,
            "f10_7": 70.0,
            "adri_score": min(0.9, 0.1 + i * 0.1),
            "collision_pc": 10 ** (-8 + i * 0.8),
            "power_margin_positive": True,
            "dod_pct": 40.0,
            "beta_deg": 20.0,
            "all_thermal_ok": True,
            "peak_aero_w": 1.0,
            "total_passes": 7,
            "total_data_mb": 300.0,
            "tid_krad_yr": 1.0,
            "seu_per_day": 0.05,
            "scrub_s": 300.0,
            "pqc_authentic": True,
            "pqc_replay_ok": True,
            "pqc_tamper_ok": True,
        }
        engine.ingest_snapshot(t)
    trends = engine.get_subsystem_trend_reports()
    risk_trend = next(r for r in trends if r.subsystem == "COLLISION_RISK")
    # As adri_score increases, risk health should degrade
    assert risk_trend.trend in (TrendDirection.DEGRADING, TrendDirection.VOLATILE)


def test_engine_history_capped_at_depth(nominal_telemetry):
    engine = MissionAnalyticsEngine("E3", initial_fuel_kg=20.0)
    for _ in range(15):
        engine.ingest_snapshot(nominal_telemetry)
    reports = engine.get_subsystem_trend_reports()
    for r in reports:
        assert len(r.history) <= engine._HISTORY_DEPTH


# ---------------------------------------------------------------------------
# Mission stage classification
# ---------------------------------------------------------------------------

def test_mission_stage_nominal_ops(nominal_telemetry):
    snap = analyze_mission_health("S1", nominal_telemetry, initial_fuel_kg=16.5)
    # fuel_remaining=15.0 / initial=16.5 ≈ 91% → LAUNCH_AND_EARLY_ORBIT
    # or if ratio > 0.70 → COMMISSIONING
    assert snap.mission_stage in (MissionStage.LAUNCH_AND_EARLY_ORBIT, MissionStage.COMMISSIONING,
                                   MissionStage.NOMINAL_OPERATIONS)


def test_mission_stage_eol():
    t = {
        "altitude_km": 550.0, "mass_kg": 260.0, "fuel_remaining_kg": 0.3,
        "kp": 1.5, "f10_7": 70.0, "adri_score": 0.1, "collision_pc": 1e-8,
        "power_margin_positive": True, "dod_pct": 40.0, "beta_deg": 20.0,
        "all_thermal_ok": True, "peak_aero_w": 0.5, "total_passes": 7,
        "total_data_mb": 300.0, "tid_krad_yr": 1.0, "seu_per_day": 0.05,
        "scrub_s": 300.0, "pqc_authentic": True, "pqc_replay_ok": True, "pqc_tamper_ok": True,
    }
    snap = analyze_mission_health("S2", t, initial_fuel_kg=20.0)
    assert snap.mission_stage in (MissionStage.DEORBIT_SEQUENCE, MissionStage.END_OF_LIFE)
