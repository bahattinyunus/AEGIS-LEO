import pytest
from aegis_leo.screening.constellation_screen import (
    generate_walker_delta_constellation,
    generate_synthetic_debris_cloud,
    ConstellationConjunctionScreener,
)


def test_walker_delta_generation():
    fleet = generate_walker_delta_constellation(total_satellites=24, planes=4)
    assert len(fleet) == 24

    planes = set(s["plane"] for s in fleet)
    assert len(planes) == 4

    for s in fleet:
        assert 0.0 <= s["raan_deg"] <= 360.0
        assert 0.0 <= s["true_anomaly_deg"] <= 360.0
        assert s["altitude_km"] == 550.0


def test_synthetic_debris_cloud_generation():
    debris = generate_synthetic_debris_cloud(count=50, center_alt_km=550.0)
    assert len(debris) == 50
    for d in debris:
        assert d["altitude_km"] > 400.0
        assert d["rcs_m2"] > 0.0


def test_constellation_screening():
    fleet = generate_walker_delta_constellation(total_satellites=8, planes=2)
    debris = generate_synthetic_debris_cloud(count=30, center_alt_km=550.0)

    screener = ConstellationConjunctionScreener()
    res = screener.screen_constellation(
        fleet=fleet,
        debris_population=debris,
        storm_kp=8.0,
        screening_threshold_km=5.0,
    )

    assert res["total_fleet_size"] == 8
    assert res["debris_population_size"] == 30
    assert "summary" in res
    assert "critical_satellites" in res["summary"]
