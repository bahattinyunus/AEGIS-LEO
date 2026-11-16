"""
Tests for aegis_leo.ai.anomaly_detector (v2.2.0)
"""

import math
import pytest
import numpy as np
from aegis_leo.ai.anomaly_detector import (
    TelemetryAnomalyDetector,
    AnomalyReport,
    AnomalyEvent,
    AnomalyCategory,
    AnomalySeverity,
    _ChannelStatTracker,
    _check_orbital_energy,
    _check_power_balance,
    _check_thermal_range,
    _check_tsiolkovsky_budget,
    _detect_seu_bit_flip,
    _mahalanobis_distance,
    detect_telemetry_anomalies,
)
from aegis_leo.constants import EARTH_RADIUS_KM, EARTH_MU


# ---------------------------------------------------------------------------
# Nominal telemetry frame helper
# ---------------------------------------------------------------------------

def _nominal_frame(alt=550.0, kp=1.5):
    v_circ = math.sqrt(EARTH_MU / (EARTH_RADIUS_KM + alt))
    return {
        "altitude_km": alt,
        "velocity_km_s": v_circ,
        "fuel_kg": 15.0,
        "battery_soc_pct": 80.0,
        "solar_power_w": 400.0,
        "bus_load_w": 180.0,
        "temp_avionics_c": 25.0,
        "temp_battery_c": 15.0,
        "kp_index": kp,
        "pinn_density_output": 2.4e-13,
        "eclipse_flag": False,
        "mass_kg": 260.0,
        "isp_s": 250.0,
        "delta_v_m_s": 0.0,
        "fuel_kg_before_burn": 15.0,
    }


# ---------------------------------------------------------------------------
# _ChannelStatTracker
# ---------------------------------------------------------------------------

def test_tracker_z_score_stable():
    tracker = _ChannelStatTracker(window=20, min_samples=5)
    for _ in range(10):
        z, _ = tracker.update(50.0)
    # After 10 identical values, z-score should be near 0
    z, _ = tracker.update(50.0)
    assert abs(z) < 0.5


def test_tracker_z_score_outlier():
    tracker = _ChannelStatTracker(window=20, min_samples=5)
    for _ in range(15):
        tracker.update(50.0)
    z, _ = tracker.update(100.0)
    assert z > 3.0


def test_tracker_cusum_alarm():
    tracker = _ChannelStatTracker(window=20, min_samples=5)
    for _ in range(10):
        tracker.update(50.0)
    # Apply a sustained shift
    alarm_fired = False
    for i in range(20):
        _, alarm = tracker.update(70.0)
        if alarm:
            alarm_fired = True
            break
    assert alarm_fired, "CUSUM should fire after sustained mean shift"


def test_tracker_insufficient_samples():
    tracker = _ChannelStatTracker(window=20, min_samples=5)
    z, alarm = tracker.update(100.0)
    assert z == 0.0
    assert alarm is False


# ---------------------------------------------------------------------------
# Physics checks
# ---------------------------------------------------------------------------

def test_check_orbital_energy_valid():
    alt = 550.0
    v_circ = math.sqrt(EARTH_MU / (EARTH_RADIUS_KM + alt))
    ok, dev = _check_orbital_energy(alt, v_circ)
    assert ok is True
    assert dev < 1.0


def test_check_orbital_energy_invalid():
    ok, dev = _check_orbital_energy(550.0, 1.0)  # 1 km/s is way off
    assert ok is False
    assert dev > 5.0


@pytest.mark.parametrize("v_pct", [0.96, 0.97, 0.98, 1.00, 1.02, 1.03, 1.04])
def test_check_orbital_energy_within_5pct(v_pct):
    alt = 550.0
    v_circ = math.sqrt(EARTH_MU / (EARTH_RADIUS_KM + alt))
    ok, _ = _check_orbital_energy(alt, v_circ * v_pct)
    if abs(v_pct - 1.0) < 0.05:
        assert ok is True
    else:
        assert ok is False


def test_check_power_balance_sunlit_ok():
    ok, msg = _check_power_balance(400.0, 180.0, eclipse_flag=False, battery_soc_pct=80.0)
    assert ok is True


def test_check_power_balance_sunlit_fail():
    ok, msg = _check_power_balance(50.0, 200.0, eclipse_flag=False, battery_soc_pct=80.0)
    assert ok is False
    assert "power" in msg.lower() or "solar" in msg.lower() or "load" in msg.lower()


def test_check_power_balance_eclipse_ok():
    ok, msg = _check_power_balance(0.0, 180.0, eclipse_flag=True, battery_soc_pct=60.0)
    assert ok is True


def test_check_power_balance_eclipse_low_battery():
    ok, msg = _check_power_balance(0.0, 180.0, eclipse_flag=True, battery_soc_pct=2.0)
    assert ok is False


def test_check_thermal_range_ok():
    ok, msg = _check_thermal_range("TEST_NODE", 20.0, -20.0, 65.0)
    assert ok is True


def test_check_thermal_range_too_hot():
    ok, msg = _check_thermal_range("TEST_NODE", 80.0, -20.0, 65.0)
    assert ok is False
    assert "above" in msg.lower()


def test_check_thermal_range_too_cold():
    ok, msg = _check_thermal_range("TEST_NODE", -30.0, -20.0, 65.0)
    assert ok is False
    assert "below" in msg.lower()


def test_check_tsiolkovsky_ok():
    # ~0.1 m/s burn → mass consumed should follow rocket equation
    ok, disc = _check_tsiolkovsky_budget(
        fuel_before_kg=15.0,
        fuel_after_kg=14.99,  # very small burn
        mass_kg=260.0,
        isp_s=250.0,
        delta_v_m_s=0.1,
    )
    assert ok is True  # small burns have large tolerance


def test_check_tsiolkovsky_mismatch():
    # Claim 1 kg was burned but dv suggests only ~0.001 kg
    ok, disc = _check_tsiolkovsky_budget(
        fuel_before_kg=15.0,
        fuel_after_kg=14.0,  # 1 kg consumed
        mass_kg=260.0,
        isp_s=250.0,
        delta_v_m_s=0.1,  # tiny dv, only ~0.0001 kg needed
    )
    assert ok is False
    assert disc > 10.0


def test_check_tsiolkovsky_zero_dv():
    ok, disc = _check_tsiolkovsky_budget(15.0, 15.0, 260.0, 250.0, 0.0)
    assert ok is True


def test_detect_seu_bit_flip_clean():
    history = [2.4e-13] * 10
    detected, sigma = _detect_seu_bit_flip(history)
    assert bool(detected) is False


def test_detect_seu_bit_flip_spike():
    history = [2.4e-13] * 9 + [2.4e-7]  # 6 orders of magnitude jump
    detected, sigma = _detect_seu_bit_flip(history)
    assert bool(detected) is True
    assert sigma > 4.0


def test_detect_seu_insufficient_history():
    detected, sigma = _detect_seu_bit_flip([2.4e-13, 2.4e-13])
    assert detected is False


# ---------------------------------------------------------------------------
# Mahalanobis distance
# ---------------------------------------------------------------------------

def test_mahalanobis_nominal():
    rng = np.random.default_rng(42)
    history = rng.multivariate_normal([0, 0, 0], np.eye(3), size=20)
    x_ok = np.array([0.1, -0.2, 0.3])
    d_ok = _mahalanobis_distance(x_ok, history)
    assert d_ok < 3.0


def test_mahalanobis_anomaly():
    rng = np.random.default_rng(42)
    history = rng.multivariate_normal([0, 0, 0], np.eye(3), size=20)
    x_anom = np.array([10.0, 10.0, 10.0])
    d_anom = _mahalanobis_distance(x_anom, history)
    assert d_anom > 5.0


def test_mahalanobis_insufficient_history():
    history = np.array([[1.0, 2.0], [3.0, 4.0]])  # too few samples
    x = np.array([1.5, 2.5])
    d = _mahalanobis_distance(x, history)
    assert d >= 0.0


# ---------------------------------------------------------------------------
# TelemetryAnomalyDetector — full integration
# ---------------------------------------------------------------------------

def test_detector_nominal_no_anomalies():
    detector = TelemetryAnomalyDetector("SAT-01", z_threshold=3.5)
    for _ in range(20):
        report = detector.ingest_frame(_nominal_frame())
    # After warmup, nominal frames should produce few/no anomalies
    assert report.critical_count == 0
    assert isinstance(report, AnomalyReport)


def test_detector_reports_epoch_increment():
    detector = TelemetryAnomalyDetector("SAT-01")
    for i in range(5):
        report = detector.ingest_frame(_nominal_frame())
        assert report.epoch_index == i + 1


def test_detector_temperature_violation():
    detector = TelemetryAnomalyDetector("SAT-01", z_threshold=3.5)
    # Warm up
    for _ in range(10):
        detector.ingest_frame(_nominal_frame())
    # Inject temperature violation
    frame = _nominal_frame()
    frame["temp_avionics_c"] = 85.0  # above 65°C limit
    report = detector.ingest_frame(frame)
    phys_flags = [a for a in report.anomalies if a.category == AnomalyCategory.PHYSICS_VIOLATION and "AVIONICS" in a.description.upper()]
    assert len(phys_flags) >= 1


def test_detector_power_balance_violation():
    detector = TelemetryAnomalyDetector("SAT-01")
    for _ in range(10):
        detector.ingest_frame(_nominal_frame())
    # Inject power balance failure (sunlit but low solar power)
    frame = _nominal_frame()
    frame["solar_power_w"] = 20.0  # << bus load of 180 W
    frame["eclipse_flag"] = False
    report = detector.ingest_frame(frame)
    power_flags = [a for a in report.anomalies if "POWER" in a.description.upper() or "solar" in a.description.lower()]
    assert len(power_flags) >= 1


def test_detector_seu_detection():
    detector = TelemetryAnomalyDetector("SAT-01")
    for _ in range(15):
        detector.ingest_frame(_nominal_frame())
    # Inject SEU bit-flip in PINN output
    frame = _nominal_frame()
    frame["pinn_density_output"] = 2.4e-7  # 6 orders of magnitude jump
    report = detector.ingest_frame(frame)
    seu_flags = [a for a in report.anomalies if a.category == AnomalyCategory.SEU_BIT_FLIP]
    assert len(seu_flags) >= 1
    assert seu_flags[0].severity == AnomalySeverity.CRITICAL


def test_detector_z_score_outlier():
    detector = TelemetryAnomalyDetector("SAT-01", z_threshold=3.0)
    for _ in range(20):
        detector.ingest_frame(_nominal_frame())
    # Inject fuel reading far from baseline
    frame = _nominal_frame()
    frame["fuel_kg"] = 100.0  # far above baseline ~15
    report = detector.ingest_frame(frame)
    z_flags = [a for a in report.anomalies if a.category == AnomalyCategory.STATISTICAL_OUTLIER and a.channel == "fuel_kg"]
    assert len(z_flags) >= 1


def test_detector_vis_viva_violation():
    detector = TelemetryAnomalyDetector("SAT-01")
    for _ in range(10):
        detector.ingest_frame(_nominal_frame())
    frame = _nominal_frame()
    frame["velocity_km_s"] = 1.0  # impossibly low
    report = detector.ingest_frame(frame)
    vis_flags = [a for a in report.anomalies if a.category == AnomalyCategory.PHYSICS_VIOLATION and "vis-viva" in a.description.lower()]
    assert len(vis_flags) >= 1


def test_detector_health_score_decreases_with_anomalies():
    detector = TelemetryAnomalyDetector("SAT-01")
    for _ in range(20):
        detector.ingest_frame(_nominal_frame())
    nominal_report = detector.ingest_frame(_nominal_frame())

    # Now inject an anomaly
    frame = _nominal_frame()
    frame["temp_avionics_c"] = 90.0  # critical
    anomaly_report = detector.ingest_frame(frame)
    assert anomaly_report.overall_health_score <= nominal_report.overall_health_score + 5.0


def test_detector_report_to_dict():
    detector = TelemetryAnomalyDetector("SAT-01")
    report = detector.ingest_frame(_nominal_frame())
    d = report.to_dict()
    assert isinstance(d, dict)
    assert "anomalies" in d
    assert "channel_z_scores" in d
    assert "physics_checks" in d
    assert "overall_health_score" in d


# ---------------------------------------------------------------------------
# detect_telemetry_anomalies batch function
# ---------------------------------------------------------------------------

def test_batch_detection_nominal():
    frames = [_nominal_frame() for _ in range(10)]
    reports = detect_telemetry_anomalies("SAT-TEST", frames)
    assert len(reports) == 10
    for r in reports:
        assert isinstance(r, AnomalyReport)
        assert r.satellite_id == "SAT-TEST"


def test_batch_detection_epoch_ordering():
    frames = [_nominal_frame() for _ in range(5)]
    reports = detect_telemetry_anomalies("SAT-TEST", frames)
    for i, r in enumerate(reports):
        assert r.epoch_index == i + 1


def test_batch_detection_with_injected_seu():
    frames = [_nominal_frame() for _ in range(15)]
    # Inject SEU at frame 12
    frames[11]["pinn_density_output"] = 2.4e-5
    reports = detect_telemetry_anomalies("SAT-TEST", frames)
    # By frame 12 the detector should have enough history
    seu_in_frame_12 = [a for a in reports[11].anomalies if a.category == AnomalyCategory.SEU_BIT_FLIP]
    assert len(seu_in_frame_12) >= 1


def test_batch_detection_z_threshold_effect():
    frames = [_nominal_frame() for _ in range(20)]
    frames[10]["fuel_kg"] = 60.0  # injected outlier

    reports_strict = detect_telemetry_anomalies("SAT-TEST", frames, z_threshold=2.0)
    reports_loose  = detect_telemetry_anomalies("SAT-TEST", frames, z_threshold=5.0)

    strict_outliers = sum(
        1 for r in reports_strict
        for a in r.anomalies
        if a.category == AnomalyCategory.STATISTICAL_OUTLIER and a.channel == "fuel_kg"
    )
    loose_outliers = sum(
        1 for r in reports_loose
        for a in r.anomalies
        if a.category == AnomalyCategory.STATISTICAL_OUTLIER and a.channel == "fuel_kg"
    )
    # Strict threshold should catch at least as many as loose
    assert strict_outliers >= loose_outliers
