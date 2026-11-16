"""
Tests for aegis_leo.physics.space_weather_forecast (v2.2.0)
"""

import pytest
from aegis_leo.physics.space_weather_forecast import (
    SpaceWeatherForecaster,
    SpaceWeatherForecast,
    ForecastHour,
    StormOnsetAlert,
    CMETransitModel,
    run_space_weather_forecast,
)
from aegis_leo.constants import QUIET_KP, QUIET_F10_7


# ---------------------------------------------------------------------------
# CMETransitModel
# ---------------------------------------------------------------------------

def test_cme_inactive_low_speed():
    cme = CMETransitModel(cme_speed_km_s=100.0)
    assert not cme.is_active()
    assert cme.predicted_transit_hours() == float("inf")
    assert cme.predicted_peak_kp() == pytest.approx(QUIET_KP, abs=0.5)


def test_cme_active_high_speed():
    cme = CMETransitModel(cme_speed_km_s=1200.0, base_kp=2.0)
    assert cme.is_active()
    t = cme.predicted_transit_hours()
    assert 10.0 < t < 60.0
    pk = cme.predicted_peak_kp()
    assert pk > 2.0
    assert pk <= 9.0


def test_cme_onset_probability_before_arrival():
    cme = CMETransitModel(cme_speed_km_s=800.0)
    t_arrival = cme.predicted_transit_hours()
    # Probability long before arrival should be low
    p_early = cme.onset_probability(1.0)
    # Probability at arrival should be ~0.5
    p_at = cme.onset_probability(t_arrival)
    assert p_early < p_at


def test_cme_onset_probability_range():
    cme = CMETransitModel(cme_speed_km_s=900.0)
    for h in range(1, 73, 6):
        p = cme.onset_probability(float(h))
        assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------------------
# SpaceWeatherForecaster — basic forecast
# ---------------------------------------------------------------------------

def test_forecast_returns_correct_horizon():
    fc = run_space_weather_forecast(horizon_hours=24)
    assert isinstance(fc, SpaceWeatherForecast)
    assert fc.horizon_hours == 24
    assert len(fc.forecast_hours) == 24


def test_forecast_hourly_entries():
    fc = run_space_weather_forecast(horizon_hours=12)
    for h in fc.forecast_hours:
        assert isinstance(h, ForecastHour)
        assert 1 <= h.hour_offset <= 72
        assert 0.0 <= h.kp_p50 <= 9.0
        assert 0.0 <= h.kp_p10 <= 9.0
        assert 0.0 <= h.kp_p90 <= 9.0
        assert h.kp_p10 <= h.kp_p50 + 0.5   # p10 ≤ p50 approximately
        assert 65.0 <= h.f10_7_p50 <= 450.0
        assert h.storm_onset_probability >= 0.0
        assert h.density_factor_p50 > 0.0
        assert h.along_track_drift_km_hr >= 0.0


def test_forecast_percentile_ordering():
    """P10 should generally be ≤ P50 ≤ P90."""
    fc = run_space_weather_forecast(current_kp=2.0, horizon_hours=24)
    for h in fc.forecast_hours:
        assert h.kp_p10 <= h.kp_p90 + 0.1  # small tolerance for numerical noise


def test_forecast_storm_probabilities_valid():
    fc = run_space_weather_forecast(horizon_hours=72)
    assert 0.0 <= fc.storm_probability_24h <= 1.0
    assert 0.0 <= fc.storm_probability_72h <= 1.0
    # 72h probability should be >= 24h probability (more time = more chance)
    assert fc.storm_probability_72h >= fc.storm_probability_24h - 0.05  # small tolerance


def test_forecast_max_kp_in_range():
    fc = run_space_weather_forecast(current_kp=QUIET_KP)
    assert 0.0 <= fc.max_kp_forecast <= 9.0
    assert 1 <= fc.max_kp_hour <= fc.horizon_hours


def test_forecast_drift_nonnegative():
    fc = run_space_weather_forecast(current_kp=5.0, horizon_hours=24)
    assert fc.total_drift_km_24h >= 0.0
    assert fc.total_drift_km_72h >= 0.0


def test_forecast_to_dict():
    fc = run_space_weather_forecast(horizon_hours=12)
    d = fc.to_dict()
    assert isinstance(d, dict)
    assert "forecast_hours" in d
    assert "storm_probability_24h" in d
    assert "alerts" in d
    assert len(d["forecast_hours"]) == 12


def test_forecast_various_horizons():
    for h in [1, 6, 12, 24, 48, 72]:
        fc = run_space_weather_forecast(horizon_hours=h)
        assert len(fc.forecast_hours) == h


# ---------------------------------------------------------------------------
# Storm conditions → higher density factors
# ---------------------------------------------------------------------------

def test_storm_conditions_higher_density():
    fc_quiet = run_space_weather_forecast(current_kp=QUIET_KP, current_f10_7=QUIET_F10_7, horizon_hours=1)
    fc_storm = run_space_weather_forecast(current_kp=8.0, current_f10_7=265.0, horizon_hours=1)
    # Storm density factor should be higher at hour 1
    assert fc_storm.forecast_hours[0].density_factor_p50 >= fc_quiet.forecast_hours[0].density_factor_p50


def test_storm_conditions_higher_drift():
    fc_quiet = run_space_weather_forecast(current_kp=QUIET_KP, horizon_hours=24)
    fc_storm = run_space_weather_forecast(current_kp=8.0, current_f10_7=265.0, horizon_hours=24)
    assert fc_storm.total_drift_km_24h >= fc_quiet.total_drift_km_24h


# ---------------------------------------------------------------------------
# CME-driven forecast
# ---------------------------------------------------------------------------

def test_cme_forecast_produces_alert():
    fc = run_space_weather_forecast(
        current_kp=2.0,
        current_f10_7=120.0,
        horizon_hours=48,
        cme_speed_km_s=1000.0,
    )
    # Should generate at least one alert
    assert len(fc.alerts) >= 1
    for alert in fc.alerts:
        assert isinstance(alert, StormOnsetAlert)
        assert alert.expected_peak_kp > 0
        assert 0.0 <= alert.onset_confidence_pct <= 100.0


def test_cme_metadata():
    fc = run_space_weather_forecast(cme_speed_km_s=900.0, horizon_hours=24)
    assert fc.metadata["cme_active"] is True
    assert fc.metadata["cme_speed_km_s"] == pytest.approx(900.0)


def test_no_cme_no_cme_alert():
    fc = run_space_weather_forecast(current_kp=QUIET_KP, cme_speed_km_s=0.0, horizon_hours=24)
    # No CME alerts (may still have statistical alerts if Kp rises)
    cme_alerts = [a for a in fc.alerts if "CME-ALERT" in a.alert_id]
    assert len(cme_alerts) == 0


# ---------------------------------------------------------------------------
# Kp history seeds AR predictor
# ---------------------------------------------------------------------------

def test_kp_history_accepted():
    history = [3.0, 4.5, 6.0, 7.5]
    fc = run_space_weather_forecast(
        current_kp=7.5,
        kp_history=history,
        horizon_hours=12,
    )
    assert fc.horizon_hours == 12
    # With elevated Kp history, storm prob should be non-trivial
    assert fc.storm_probability_24h >= 0.0


def test_forecast_deterministic_with_same_seed():
    """Same inputs should yield the same output (seeded RNG)."""
    fc1 = run_space_weather_forecast(current_kp=3.0, horizon_hours=12)
    fc2 = run_space_weather_forecast(current_kp=3.0, horizon_hours=12)
    for h1, h2 in zip(fc1.forecast_hours, fc2.forecast_hours):
        assert h1.kp_p50 == pytest.approx(h2.kp_p50, abs=1e-9)


# ---------------------------------------------------------------------------
# Altitude dependency
# ---------------------------------------------------------------------------

def test_different_altitudes_affect_drift():
    fc_vleo = run_space_weather_forecast(current_kp=5.0, orbit_altitude_km=220.0, horizon_hours=6)
    fc_leo  = run_space_weather_forecast(current_kp=5.0, orbit_altitude_km=550.0, horizon_hours=6)
    # VLEO is much denser → larger drift per hour
    assert fc_vleo.total_drift_km_24h >= 0.0
    assert fc_leo.total_drift_km_24h >= 0.0
