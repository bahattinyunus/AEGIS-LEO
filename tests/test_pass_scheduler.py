"""Tests for Ground Station Pass Scheduler & RF/Optical Link Budget."""

import numpy as np
from aegis_leo.physics.pass_scheduler import (
    LinkBand,
    compute_link_budget,
    compute_topocentric_look_angles,
    geodetic_to_ecef_km,
    schedule_ground_passes,
)


def test_geodetic_to_ecef_equator():
    ecef = geodetic_to_ecef_km(0.0, 0.0, 0.0)
    assert np.isclose(ecef[0], 6378.137, atol=0.1)
    assert np.isclose(ecef[1], 0.0, atol=1e-6)
    assert np.isclose(ecef[2], 0.0, atol=1e-6)


def test_topocentric_overhead_pass():
    # Satellite directly above Ankara (39.8912 N, 32.7785 E) at 550 km
    sat_ecef = geodetic_to_ecef_km(39.8912, 32.7785, 550.0)
    look = compute_topocentric_look_angles(
        sat_pos_eci_km=sat_ecef,
        sat_vel_eci_km_s=np.array([0.0, 7.58, 0.0]),
        station_lat_deg=39.8912,
        station_lon_deg=32.7785,
        gmst_rad=0.0,
    )
    assert look["elevation_deg"] > 88.0
    assert np.isclose(look["slant_range_km"], 550.0, atol=2.0)


def test_link_budget_bands_and_rain_fade():
    x_clear = compute_link_budget(slant_range_km=800.0, elevation_deg=45.0, band=LinkBand.X_BAND, rain_rate_mm_hr=0.0)
    x_rain = compute_link_budget(slant_range_km=800.0, elevation_deg=45.0, band=LinkBand.X_BAND, rain_rate_mm_hr=25.0)
    ka_clear = compute_link_budget(slant_range_km=800.0, elevation_deg=45.0, band=LinkBand.KA_BAND, rain_rate_mm_hr=0.0)
    opt_clear = compute_link_budget(slant_range_km=800.0, elevation_deg=45.0, band=LinkBand.OPTICAL, rain_rate_mm_hr=0.0)

    assert x_clear["link_margin_db"] > x_rain["link_margin_db"]
    assert ka_clear["throughput_mbps"] > x_clear["throughput_mbps"]
    assert opt_clear["throughput_mbps"] > ka_clear["throughput_mbps"]


def test_schedule_ground_passes_12h():
    sched = schedule_ground_passes(
        altitude_km=550.0,
        inclination_deg=53.0,
        duration_hours=12.0,
        band=LinkBand.X_BAND,
    )
    assert sched.total_passes > 0
    assert sched.total_contact_time_s > 0.0
    assert sched.total_data_downlinked_mb > 0.0
    assert sched.best_station is not None
    for p in sched.passes:
        assert p.max_elevation_deg >= 5.0
        assert p.duration_s > 0.0
