"""
Unit tests for AEGIS-LEO high-precision perturbation models:
Solar Radiation Pressure (SRP) and Third-Body Gravitation (Sun & Moon).
"""

import pytest
import numpy as np
from datetime import datetime, timezone
import math

from aegis_leo.constants import EARTH_RADIUS_KM
from aegis_leo.physics.perturbations import (
    datetime_to_jd,
    mjd_to_jd,
    jd_to_mjd,
    resolve_epoch_jd,
    sun_position_eci,
    moon_position_eci,
    compute_cylindrical_shadow,
    compute_dual_cone_shadow,
    compute_shadow_factor,
    compute_srp_acceleration,
    compute_third_body_acceleration,
    compute_solar_third_body_acceleration,
    compute_lunar_third_body_acceleration,
    compute_all_perturbations,
    total_perturbation_acceleration,
    rk4_step_high_precision,
    HighPrecisionPerturbationEngine,
    SUN_MU,
    MOON_MU,
    ASTRONOMICAL_UNIT_KM,
)


def test_julian_date_conversions():
    # J2000.0 epoch: 2000-01-01 12:00:00 UTC = JD 2451545.0 = MJD 51544.5
    dt_j2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    jd = datetime_to_jd(dt_j2000)
    assert np.isclose(jd, 2451545.0, atol=1e-6)

    mjd = jd_to_mjd(jd)
    assert np.isclose(mjd, 51544.5, atol=1e-6)
    assert np.isclose(mjd_to_jd(mjd), jd, atol=1e-6)

    # Test resolve_epoch_jd with various input formats
    assert np.isclose(resolve_epoch_jd(dt_j2000), 2451545.0)
    assert np.isclose(resolve_epoch_jd(2451545.0), 2451545.0)
    assert np.isclose(resolve_epoch_jd(51544.5), 2451545.0)


def test_sun_position_eci():
    jd_j2000 = 2451545.0
    r_sun = sun_position_eci(jd_j2000)

    assert r_sun.shape == (3,)
    dist = float(np.linalg.norm(r_sun))
    # Distance to Sun is ~1 AU (between 0.983 AU and 1.017 AU)
    assert 0.98 * ASTRONOMICAL_UNIT_KM < dist < 1.02 * ASTRONOMICAL_UNIT_KM

    # Jan 1 is northern winter; Sun declination should be negative (in -Z)
    assert r_sun[2] < 0.0


def test_moon_position_eci():
    jd_j2000 = 2451545.0
    r_moon = moon_position_eci(jd_j2000)

    assert r_moon.shape == (3,)
    dist = float(np.linalg.norm(r_moon))
    # Moon geocentric distance is between ~356,000 km (perigee) and ~406,700 km (apogee)
    assert 350000.0 < dist < 415000.0


def test_cylindrical_shadow():
    r_sun = np.array([1.496e8, 0.0, 0.0])

    # Day side: satellite at +X
    r_sc_day = np.array([EARTH_RADIUS_KM + 500.0, 0.0, 0.0])
    assert compute_cylindrical_shadow(r_sc_day, r_sun) == 1.0

    # Night side in shadow: satellite at -X
    r_sc_night = np.array([-(EARTH_RADIUS_KM + 500.0), 0.0, 0.0])
    assert compute_cylindrical_shadow(r_sc_night, r_sun) == 0.0

    # Night side high above cylinder: satellite at -X but Y > Earth radius
    r_sc_high = np.array([-(EARTH_RADIUS_KM + 500.0), EARTH_RADIUS_KM + 2000.0, 0.0])
    assert compute_cylindrical_shadow(r_sc_high, r_sun) == 1.0


def test_dual_cone_shadow():
    r_sun = np.array([1.496e8, 0.0, 0.0])

    # Day side
    r_sc_day = np.array([EARTH_RADIUS_KM + 500.0, 0.0, 0.0])
    assert compute_dual_cone_shadow(r_sc_day, r_sun) == 1.0

    # Deep umbra
    r_sc_umbra = np.array([-(EARTH_RADIUS_KM + 500.0), 0.0, 0.0])
    assert compute_dual_cone_shadow(r_sc_umbra, r_sun) == 0.0

    # Penumbral zone near edge of Earth disk
    r_sc_penumbra = np.array([-(EARTH_RADIUS_KM + 500.0), EARTH_RADIUS_KM, 0.0])
    nu = compute_dual_cone_shadow(r_sc_penumbra, r_sun)
    assert 0.0 < nu < 1.0

    # Dispatcher check
    assert compute_shadow_factor(r_sc_day, r_sun, model="dual_cone") == 1.0
    assert compute_shadow_factor(r_sc_day, r_sun, model="cylindrical") == 1.0
    assert compute_shadow_factor(r_sc_umbra, r_sun, model="none") == 1.0

    with pytest.raises(ValueError):
        compute_shadow_factor(r_sc_day, r_sun, model="invalid_model")


def test_srp_acceleration_direction_and_magnitude():
    # Sun at +X
    r_sun = np.array([1.496e8, 0.0, 0.0])
    # Satellite at +X (day side)
    r_sc = np.array([EARTH_RADIUS_KM + 500.0, 0.0, 0.0])

    a_srp = compute_srp_acceleration(
        r_sc_km=r_sc,
        r_sun_km=r_sun,
        cr=1.2,
        area_m2=2.0,
        mass_kg=150.0,
        shadow_factor=1.0,
    )

    # Direction: sunlight moves from Sun to satellite (+X to +X, outwards in -X direction towards Earth)
    # r_sc - r_sun = [7000 - 1.496e8, 0, 0] = negative X
    assert a_srp[0] < 0.0
    assert np.isclose(a_srp[1], 0.0, atol=1e-15)
    assert np.isclose(a_srp[2], 0.0, atol=1e-15)

    # Magnitude in km/s^2 should be ~ 1e-11 to 1e-10 km/s^2 (~ 1e-8 to 1e-7 m/s^2)
    norm_a_srp = float(np.linalg.norm(a_srp))
    assert 1e-12 < norm_a_srp < 1e-9

    # Zero if eclipsed
    a_eclipsed = compute_srp_acceleration(
        r_sc_km=r_sc,
        r_sun_km=r_sun,
        shadow_factor=0.0,
    )
    assert np.all(a_eclipsed == 0.0)

    # Zero if zero area or mass
    assert np.all(compute_srp_acceleration(r_sc, r_sun, area_m2=0.0) == 0.0)
    assert np.all(compute_srp_acceleration(r_sc, r_sun, mass_kg=0.0) == 0.0)


def test_third_body_battin_vs_direct_equivalence():
    r_sc = np.array([6878.137, 1000.0, -1500.0])
    r_body = np.array([1.496e8, 2.5e7, -1.2e7])
    mu = SUN_MU

    a_direct = compute_third_body_acceleration(r_sc, r_body, mu, use_battin=False)
    a_battin = compute_third_body_acceleration(r_sc, r_body, mu, use_battin=True)

    # Should match to within machine precision
    assert np.allclose(a_direct, a_battin, rtol=1e-9, atol=1e-16)


def test_sun_and_moon_third_body_orders_of_magnitude():
    jd = 2451545.0
    r_leo = np.array([EARTH_RADIUS_KM + 500.0, 0.0, 0.0])

    a_sun = compute_solar_third_body_acceleration(r_leo, jd)
    a_moon = compute_lunar_third_body_acceleration(r_leo, jd)

    norm_sun = float(np.linalg.norm(a_sun))
    norm_moon = float(np.linalg.norm(a_moon))

    # In LEO, Sun 3rd-body perturbation is ~1e-10 km/s^2 (~1e-7 m/s^2)
    # Moon 3rd-body perturbation is ~1e-9 km/s^2 (~1e-6 m/s^2)
    assert 1e-11 < norm_sun < 1e-8
    assert 1e-11 < norm_moon < 1e-8

    # Moon gravitational tidal acceleration is typically stronger than Sun on Earth orbits (~2.2 ratio)
    assert norm_moon > norm_sun


def test_third_body_meo_increase():
    # In MEO (e.g. GPS at r ~ 26,500 km), third body tidal effects are much larger than in LEO
    jd = 2451545.0
    r_leo = np.array([EARTH_RADIUS_KM + 500.0, 0.0, 0.0])
    r_meo = np.array([26500.0, 0.0, 0.0])

    a_moon_leo = float(np.linalg.norm(compute_lunar_third_body_acceleration(r_leo, jd)))
    a_moon_meo = float(np.linalg.norm(compute_lunar_third_body_acceleration(r_meo, jd)))

    # Tidal acceleration scales with distance from Earth center (r / r_body^3)
    assert a_moon_meo > 3.0 * a_moon_leo


def test_compute_all_perturbations_structure():
    jd = 2451545.0
    r_sc = np.array([7000.0, 0.0, 0.0])

    res = compute_all_perturbations(
        r_sc_km=r_sc,
        jd_or_epoch=jd,
        cr=1.3,
        area_m2=2.5,
        mass_kg=200.0,
        include_sun=True,
        include_moon=True,
        include_srp=True,
    )

    assert "a_total_km_s2" in res
    assert "a_srp_km_s2" in res
    assert "a_sun_km_s2" in res
    assert "a_moon_km_s2" in res
    assert "shadow_factor" in res
    assert "r_sun_km" in res
    assert "r_moon_km" in res

    # Total must equal sum of individual components
    expected_total = res["a_srp_km_s2"] + res["a_sun_km_s2"] + res["a_moon_km_s2"]
    assert np.allclose(res["a_total_km_s2"], expected_total)

    # Convenience helper check
    a_tot_direct = total_perturbation_acceleration(r_sc, jd, cr=1.3, area_m2=2.5, mass_kg=200.0)
    assert np.allclose(a_tot_direct, res["a_total_km_s2"])


def test_high_precision_engine_class():
    engine = HighPrecisionPerturbationEngine(
        cr=1.4,
        area_srp_m2=3.0,
        mass_kg=350.0,
        include_sun=True,
        include_moon=True,
        include_srp=True,
    )

    r_sc = np.array([0.0, 7200.0, 0.0])
    jd = 2451545.0
    res = engine.evaluate(r_sc, jd)

    assert res["a_total_km_s2"].shape == (3,)
    tot = engine.total_acceleration(r_sc, jd)
    assert np.allclose(tot, res["a_total_km_s2"])


def test_rk4_step_high_precision():
    # Circular LEO orbit ~ 500 km
    r0 = np.array([EARTH_RADIUS_KM + 500.0, 0.0, 0.0])
    v0 = np.array([0.0, 7.61, 0.0])
    dt = 30.0
    jd0 = 2451545.0

    r_next, v_next = rk4_step_high_precision(
        r=r0,
        v=v0,
        dt=dt,
        epoch_jd=jd0,
        cd=2.2,
        area_drag_m2=1.5,
        mass_kg=120.0,
        cr=1.2,
        area_srp_m2=1.5,
        include_drag=True,
        include_sun=True,
        include_moon=True,
        include_srp=True,
    )

    assert r_next.shape == (3,)
    assert v_next.shape == (3,)
    assert np.linalg.norm(r_next) > EARTH_RADIUS_KM
    assert np.linalg.norm(v_next) > 0.0
