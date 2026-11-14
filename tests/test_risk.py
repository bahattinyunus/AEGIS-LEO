import pytest
import numpy as np
from aegis_leo.risk.collision import compute_collision_probability, compute_relative_geometry
from aegis_leo.risk.debris_density import get_spatial_debris_density
from aegis_leo.risk.adri import ADRICalculator, RiskLevel


def test_collision_probability_decreases_with_miss_distance():
    p_close = compute_collision_probability(miss_distance_km=0.1, combined_radius_m=10.0, sigma_pos_km=0.2)
    p_far = compute_collision_probability(miss_distance_km=2.0, combined_radius_m=10.0, sigma_pos_km=0.2)

    assert p_close > p_far
    assert 0.0 <= p_close <= 1.0
    assert 0.0 <= p_far <= 1.0


def test_spatial_debris_density_peaks_at_800km():
    dense_800 = get_spatial_debris_density(altitude_km=800.0, inclination_deg=98.0)
    low_300 = get_spatial_debris_density(altitude_km=300.0, inclination_deg=98.0)

    # 800 km Iridium/Cosmos belt is known to have substantially higher debris density than 300 km
    assert dense_800["spatial_density_objects_per_km3"] > low_300["spatial_density_objects_per_km3"]
    assert 0.0 <= dense_800["normalized_debris_factor"] <= 1.0


def test_adri_scales_with_storm_and_miss_distance():
    calc = ADRICalculator()

    # Nominal quiet condition with large miss distance
    risk_nominal = calc.evaluate_risk(
        miss_distance_km=5.0,
        altitude_km=550.0,
        inclination_deg=53.0,
        kp=1.5,
        f10_7=70.0,
        along_track_disp_km=0.1,
    )

    # Critical storm condition with close conjunction
    risk_critical = calc.evaluate_risk(
        miss_distance_km=0.2,
        altitude_km=550.0,
        inclination_deg=53.0,
        kp=8.5,
        f10_7=280.0,
        along_track_disp_km=18.0,
    )

    assert risk_nominal["a_dri"] < 0.35
    assert risk_nominal["risk_level"] in [RiskLevel.NOMINAL.value, RiskLevel.ELEVATED.value]

    assert risk_critical["a_dri"] > 0.60
    assert risk_critical["risk_level"] in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]
    assert 0.0 <= risk_critical["a_dri"] <= 1.0
