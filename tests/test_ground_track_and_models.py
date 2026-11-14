import pytest
import numpy as np
from aegis_leo.physics.ground_track import (
    calculate_ground_track,
    compute_ground_station_visibility,
    analyze_communication_blackouts,
    GROUND_STATIONS,
)
from aegis_leo.physics.atmosphere_models import (
    compute_jacchia_density,
    compute_nrlmsise_density_approx,
    compare_atmospheric_models,
)


def test_ground_station_visibility():
    ankara = GROUND_STATIONS[0]
    # Direct overhead pass
    vis = compute_ground_station_visibility(
        sat_lat=ankara["lat"],
        sat_lon=ankara["lon"],
        sat_alt_km=550.0,
        station_lat=ankara["lat"],
        station_lon=ankara["lon"],
    )

    assert vis["is_visible"] is True
    assert vis["elevation_deg"] > 85.0
    assert vis["ground_distance_km"] < 10.0


def test_ground_track_and_blackout_analysis():
    # Synthetic circular trajectory
    times = np.linspace(0, 3600, 30)
    r_orbit = 6378.137 + 550.0
    positions = np.zeros((30, 3))
    for i, t in enumerate(times):
        theta = (t / 3600.0) * 2.0 * np.pi
        positions[i] = [r_orbit * np.cos(theta), r_orbit * np.sin(theta) * 0.8, r_orbit * np.sin(theta) * 0.6]

    track = calculate_ground_track(positions, times)
    assert len(track) == 30
    for pt in track:
        assert -90.0 <= pt["latitude_deg"] <= 90.0
        assert -180.0 <= pt["longitude_deg"] <= 180.0

    blackout = analyze_communication_blackouts(track, storm_kp=8.5)
    assert "blackout_window_percentage" in blackout
    assert blackout["storm_kp"] == 8.5


def test_atmosphere_models_comparison():
    alts = [300.0, 500.0, 700.0]
    comp = compare_atmospheric_models(alts, kp=7.5, f10_7=210.0)

    assert "comparison_table" in comp
    table = comp["comparison_table"]
    assert len(table) == 3

    for row in table:
        assert row["exponential_kg_m3"] > 0.0
        assert row["jacchia_77_kg_m3"] > 0.0
        assert row["nrlmsise_00_kg_m3"] > 0.0
