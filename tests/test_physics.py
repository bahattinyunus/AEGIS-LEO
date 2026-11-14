import pytest
import numpy as np
from aegis_leo.constants import EARTH_RADIUS_KM, QUIET_KP, QUIET_F10_7
from aegis_leo.physics.atmosphere import (
    compute_atmospheric_density,
    compute_drag_acceleration,
    compute_ballistic_coefficient,
)
from aegis_leo.physics.coordinates import (
    kepler_to_cartesian,
    cartesian_to_kepler,
    eci_to_geodetic,
    relative_atmospheric_velocity,
)
from aegis_leo.physics.propagator import OrbitalPropagator, compute_gravitational_and_j2_acceleration


def test_atmospheric_density_decreases_with_altitude():
    rho_200 = compute_atmospheric_density(200.0, QUIET_KP, QUIET_F10_7)
    rho_400 = compute_atmospheric_density(400.0, QUIET_KP, QUIET_F10_7)
    rho_600 = compute_atmospheric_density(600.0, QUIET_KP, QUIET_F10_7)
    rho_800 = compute_atmospheric_density(800.0, QUIET_KP, QUIET_F10_7)

    assert rho_200 > rho_400 > rho_600 > rho_800 > 0.0


def test_geomagnetic_storm_increases_density():
    # Test at 500 km altitude (typical LEO constellation)
    rho_quiet = compute_atmospheric_density(500.0, kp=1.5, f10_7=70.0)
    rho_g4 = compute_atmospheric_density(500.0, kp=8.0, f10_7=240.0)
    rho_g5 = compute_atmospheric_density(500.0, kp=9.0, f10_7=320.0)

    # Severe storm should cause substantial density expansion
    assert rho_g4 > 2.5 * rho_quiet
    assert rho_g5 > rho_g4


def test_drag_acceleration_vector_opposes_relative_velocity():
    r = np.array([EARTH_RADIUS_KM + 500.0, 0.0, 0.0])
    v = np.array([0.0, 7.6, 0.0])  # circular velocity ~7.6 km/s

    a_drag = compute_drag_acceleration(
        r_eci_km=r,
        v_eci_km_s=v,
        cd=2.2,
        area_m2=2.0,
        mass_kg=200.0,
        kp=8.0,
        f10_7=200.0,
    )

    # Drag acceleration should oppose velocity vector (negative along y)
    assert a_drag[1] < 0.0
    assert np.linalg.norm(a_drag) > 0.0


def test_kepler_cartesian_roundtrip():
    a = 6900.0
    ecc = 0.015
    inc = 51.6
    raan = 45.0
    argp = 30.0
    nu = 60.0

    r, v = kepler_to_cartesian(a, ecc, inc, raan, argp, nu)
    assert len(r) == 3 and len(v) == 3
    assert np.linalg.norm(r) > EARTH_RADIUS_KM

    elements = cartesian_to_kepler(r, v)
    assert np.isclose(elements["semi_major_axis_km"], a, rtol=1e-3)
    assert np.isclose(elements["eccentricity"], ecc, atol=1e-4)
    assert np.isclose(elements["inclination_deg"], inc, atol=1e-3)


def test_geodetic_conversion():
    r = np.array([EARTH_RADIUS_KM + 500.0, 0.0, 0.0])
    lat, lon, alt = eci_to_geodetic(r, gmst_rad=0.0)
    assert np.isclose(lat, 0.0, atol=1e-2)
    assert np.isclose(lon, 0.0, atol=1e-2)
    assert np.isclose(alt, 500.0, rtol=1e-2)


def test_numerical_propagation_and_storm_divergence():
    r0, v0 = kepler_to_cartesian(
        a_km=EARTH_RADIUS_KM + 500.0,
        ecc=0.001,
        inc_deg=53.0,
        raan_deg=0.0,
        argp_deg=0.0,
        nu_deg=0.0,
    )

    propagator = OrbitalPropagator(cd=2.2, area_m2=2.5, mass_kg=150.0)
    res = propagator.simulate_storm_divergence(
        r0=r0,
        v0=v0,
        duration_hours=2.0,  # 2 hours for quick test
        step_seconds=30.0,
        storm_kp=8.0,
        storm_f10_7=240.0,
    )

    # Storm condition must exhibit higher decay and non-zero along-track displacement
    assert res["storm_altitude_decay_m"] > res["quiet_altitude_decay_m"]
    assert res["total_divergence_km"] > 0.0
