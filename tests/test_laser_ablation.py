"""
Tests for Laser Orbital Debris Remediation (LODR) & Momentum Ablation.
"""

import pytest
from aegis_leo.adr.laser_ablation import (
    LaserStationSpecs,
    DebrisTargetSpecs,
    LaserPlatformType,
    compute_atmospheric_laser_transmission,
    compute_momentum_coupling,
    simulate_laser_debris_engagement,
)


def test_laser_station_specs():
    specs = LaserStationSpecs(pulse_energy_joules=150.0, repetition_rate_hz=50.0)
    assert specs.average_power_kw == 7.5


def test_atmospheric_laser_transmission():
    t_zenith = compute_atmospheric_laser_transmission(0.0)
    t_slanted = compute_atmospheric_laser_transmission(0.785)  # 45 deg

    assert 0.85 < t_zenith < 1.0
    assert t_slanted < t_zenith


def test_momentum_coupling_response():
    # Sub-threshold
    cm_low, risk_low = compute_momentum_coupling(0.5)
    assert cm_low < 0.1
    assert risk_low < 0.01

    # Optimal ablation
    cm_opt, risk_opt = compute_momentum_coupling(8.5)
    assert 40.0 < cm_opt < 60.0
    assert risk_opt < 0.10

    # Extreme over-fluence: fragmentation risk elevates
    cm_high, risk_high = compute_momentum_coupling(120.0)
    assert risk_high > 0.50


def test_simulate_laser_debris_engagement_space_based():
    laser = LaserStationSpecs(platform=LaserPlatformType.SPACE_BASED_ORBITAL, pulse_energy_joules=120.0)
    debris = DebrisTargetSpecs(mass_kg=0.8, effective_diameter_cm=15.0)

    res = simulate_laser_debris_engagement(
        laser=laser,
        debris=debris,
        slant_range_km=250.0,
        engagement_duration_s=30.0,
    )

    assert res.total_impulse_n_s > 0.0005
    assert res.delta_v_m_s > 0.0005
    assert res.perigee_drop_km > 0.0
    assert res.new_perigee_alt_km < debris.perigee_alt_km
    assert res.fragmentation_risk_score < 0.10

    d = res.to_dict()
    assert d["status"] == "PASS_COMPLETED"
    assert d["delta_v_m_s"] > 0


def test_simulate_laser_debris_engagement_ground_based():
    laser = LaserStationSpecs(platform=LaserPlatformType.GROUND_BASED, director_diameter_m=2.0)
    debris = DebrisTargetSpecs(mass_kg=1.5)

    res = simulate_laser_debris_engagement(
        laser=laser,
        debris=debris,
        slant_range_km=400.0,
        engagement_duration_s=60.0,
        zenith_angle_deg=30.0,
    )

    assert res.pulses_on_target > 0
    assert res.deorbit_accelerated_years > 0.0
