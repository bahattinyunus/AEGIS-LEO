"""
Unit tests for variable attitude ballistic coefficient and solar array cross-sectional area dynamics.
Tests aegis_leo/physics/drag_cross_section.py
"""

import pytest
import numpy as np
from aegis_leo.constants import EARTH_RADIUS_KM, QUIET_KP, QUIET_F10_7
from aegis_leo.physics.drag_cross_section import (
    AttitudeMode,
    TrackingAxis,
    SpacecraftGeometry,
    compute_solar_beta_angle,
    compute_solar_beta_angle_from_vectors,
    is_in_eclipse,
    compute_eclipse_fraction,
    compute_solar_array_cross_section,
    compute_bus_cross_section,
    compute_sentman_plate_cd,
    compute_ballistic_coefficient,
    compute_instantaneous_drag_properties,
    compute_orbit_profile,
    compute_orbit_averaged_metrics,
    compute_beta_angle_sweep,
    compute_variable_attitude_drag_acceleration,
)


def test_spacecraft_geometry_defaults_and_properties():
    geom = SpacecraftGeometry(
        mass_kg=200.0,
        bus_length_m=1.2,
        bus_width_m=0.8,
        bus_height_m=1.5,
        solar_array_area_m2=4.0,
        solar_array_thickness_m=0.02,
        solar_array_span_m=4.0,
    )

    assert geom.solar_array_edge_area_m2 == pytest.approx(0.08, rel=1e-5)
    assert geom.bus_frontal_area_m2 == pytest.approx(0.8 * 1.5, rel=1e-5)
    assert geom.bus_side_area_m2 == pytest.approx(1.2 * 1.5, rel=1e-5)
    assert geom.bus_top_bottom_area_m2 == pytest.approx(1.2 * 0.8, rel=1e-5)
    assert geom.bus_min_area_m2 == pytest.approx(0.96, rel=1e-5)
    assert geom.bus_max_area_m2 == pytest.approx(1.80, rel=1e-5)


def test_solar_beta_angle_analytical_and_vectors():
    # Equatorial orbit (inc = 0), Sun on equator (dec = 0) -> beta = 0
    beta_equatorial = compute_solar_beta_angle(
        inc_rad=0.0,
        raan_rad=0.0,
        sun_ra_rad=0.0,
        sun_dec_rad=0.0,
    )
    assert np.isclose(beta_equatorial, 0.0, atol=1e-6)

    # Polar orbit (inc = 90 deg), Sun along RAAN (raan = 0, sun_ra = 0) -> beta = 0
    beta_polar_zero = compute_solar_beta_angle(
        inc_rad=np.pi / 2.0,
        raan_rad=0.0,
        sun_ra_rad=0.0,
        sun_dec_rad=0.0,
    )
    assert np.isclose(beta_polar_zero, 0.0, atol=1e-6)

    # Polar orbit, Sun perpendicular to orbit plane (raan = 90 deg, sun_ra = 0) -> beta = 90 deg
    beta_polar_max = compute_solar_beta_angle(
        inc_rad=np.pi / 2.0,
        raan_rad=np.pi / 2.0,
        sun_ra_rad=0.0,
        sun_dec_rad=0.0,
    )
    assert np.isclose(beta_polar_max, np.pi / 2.0, atol=1e-6)

    # Vector-based calculation:
    # Circular orbit in XY plane: r along X, v along Y -> h along Z
    r = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])
    sun_inplane = np.array([1.5e8, 0.0, 0.0])
    sun_outofplane = np.array([0.0, 0.0, 1.5e8])

    beta_vec_inplane = compute_solar_beta_angle_from_vectors(r, v, sun_inplane)
    assert np.isclose(beta_vec_inplane, 0.0, atol=1e-6)

    beta_vec_outofplane = compute_solar_beta_angle_from_vectors(r, v, sun_outofplane)
    assert np.isclose(beta_vec_outofplane, np.pi / 2.0, atol=1e-6)


def test_eclipse_detection_and_fraction():
    # Orbit noon (nu = 0, nu_noon = 0): should NOT be in eclipse
    assert not is_in_eclipse(true_anomaly_rad=0.0, beta_rad=0.0, altitude_km=500.0)

    # Orbit midnight (nu = pi, nu_noon = 0): at beta = 0, should BE in eclipse
    assert is_in_eclipse(true_anomaly_rad=np.pi, beta_rad=0.0, altitude_km=500.0)

    # High beta angle (|beta| > beta_crit ~ 68 deg at 500 km) -> in full sunlight (no eclipse)
    beta_high = np.radians(75.0)
    assert not is_in_eclipse(true_anomaly_rad=np.pi, beta_rad=beta_high, altitude_km=500.0)
    assert compute_eclipse_fraction(beta_high, altitude_km=500.0) == 0.0

    # At beta = 0, eclipse fraction is ~ 0.35 - 0.40 in LEO
    f_ecl_0 = compute_eclipse_fraction(0.0, altitude_km=500.0)
    assert 0.30 < f_ecl_0 < 0.45

    # Fraction decreases monotonically with increasing beta
    f_ecl_30 = compute_eclipse_fraction(np.radians(30.0), altitude_km=500.0)
    f_ecl_60 = compute_eclipse_fraction(np.radians(60.0), altitude_km=500.0)
    assert f_ecl_0 > f_ecl_30 > f_ecl_60 > 0.0


def test_solar_array_cross_section_1axis():
    geom = SpacecraftGeometry(
        solar_array_area_m2=3.0,
        solar_array_span_m=3.0,
        solar_array_thickness_m=0.02,
    )
    edge_area = geom.solar_array_edge_area_m2  # 0.06 m^2

    # In 1-axis tracking rotating around pitch (orbit normal):
    # At orbit noon (nu = 0): array points at Sun, perpendicular to position -> edge-on to velocity!
    a_noon = compute_solar_array_cross_section(
        true_anomaly_rad=0.0,
        beta_rad=0.0,
        geometry=geom,
        attitude_mode=AttitudeMode.LVLH_SUN_TRACKING_1AXIS,
        nu_noon_rad=0.0,
    )
    assert np.isclose(a_noon, edge_area, atol=1e-4)

    # At dusk / terminator crossing (nu = pi/2): array faces into the ram velocity vector!
    a_dusk = compute_solar_array_cross_section(
        true_anomaly_rad=np.pi / 2.0,
        beta_rad=0.0,
        geometry=geom,
        attitude_mode=AttitudeMode.LVLH_SUN_TRACKING_1AXIS,
        nu_noon_rad=0.0,
    )
    assert np.isclose(a_dusk, 3.0, atol=1e-4)


def test_solar_array_cross_section_2axis_dynamics_with_beta():
    geom = SpacecraftGeometry(
        solar_array_area_m2=3.0,
        solar_array_span_m=3.0,
        solar_array_thickness_m=0.02,
    )
    edge_area = geom.solar_array_edge_area_m2

    # At beta = 0: 2-axis tracking behaves identically to 1-axis at dusk
    a_dusk_beta0 = compute_solar_array_cross_section(
        true_anomaly_rad=np.pi / 2.0,
        beta_rad=0.0,
        geometry=geom,
        attitude_mode=AttitudeMode.LVLH_SUN_TRACKING_2AXIS,
    )
    assert np.isclose(a_dusk_beta0, 3.0, atol=1e-3)

    # At beta = 90 deg (terminator orbit where Sun is normal to orbit plane):
    # Velocity vector is always perpendicular to Sun vector!
    # Array normal points at Sun (out of plane), so it is edge-on to velocity everywhere in orbit!
    a_dusk_beta90 = compute_solar_array_cross_section(
        true_anomaly_rad=np.pi / 2.0,
        beta_rad=np.pi / 2.0,
        geometry=geom,
        attitude_mode=AttitudeMode.LVLH_SUN_TRACKING_2AXIS,
    )
    assert np.isclose(a_dusk_beta90, edge_area, atol=1e-4)

    a_noon_beta90 = compute_solar_array_cross_section(
        true_anomaly_rad=0.0,
        beta_rad=np.pi / 2.0,
        geometry=geom,
        attitude_mode=AttitudeMode.LVLH_SUN_TRACKING_2AXIS,
    )
    assert np.isclose(a_noon_beta90, edge_area, atol=1e-4)


def test_attitude_modes_feathered_vs_drag_maximization():
    geom = SpacecraftGeometry(
        mass_kg=150.0,
        bus_length_m=1.0,
        bus_width_m=1.0,
        bus_height_m=1.5,
        solar_array_area_m2=3.0,
        solar_array_span_m=3.0,
        solar_array_thickness_m=0.02,
        cd_bus=2.2,
        cd_solar_array=2.2,
    )

    # In FEATHERED mode:
    # Array is edge-on, bus presents minimum area
    props_feathered = compute_instantaneous_drag_properties(
        true_anomaly_rad=np.pi / 2.0,
        beta_rad=0.0,
        geometry=geom,
        attitude_mode=AttitudeMode.FEATHERED,
    )

    # In DRAG_MAXIMIZATION mode:
    # Array is broadside, bus presents maximum area
    props_max_drag = compute_instantaneous_drag_properties(
        true_anomaly_rad=np.pi / 2.0,
        beta_rad=0.0,
        geometry=geom,
        attitude_mode=AttitudeMode.DRAG_MAXIMIZATION,
    )

    assert props_feathered["total_area_m2"] < props_max_drag["total_area_m2"]
    # Ballistic coefficient in feathered mode should be much higher than drag maximization
    assert props_feathered["ballistic_coefficient_kg_m2"] > 3.0 * props_max_drag["ballistic_coefficient_kg_m2"]


def test_eclipse_feathering_mode():
    geom = SpacecraftGeometry(solar_array_area_m2=3.0, solar_array_thickness_m=0.02, solar_array_span_m=3.0)
    edge_area = geom.solar_array_edge_area_m2

    # At midnight in eclipse without feathering:
    # 1-axis tracking at midnight (nu = pi): sin(pi) = 0 -> already edge-on
    # Let's test at an angle in eclipse where array would otherwise have face area, e.g. nu = 150 deg (5*pi/6)
    nu_eclipse = np.radians(150.0)
    assert is_in_eclipse(nu_eclipse, beta_rad=0.0, altitude_km=500.0)

    # Without eclipse feathering:
    a_standard = compute_solar_array_cross_section(
        true_anomaly_rad=nu_eclipse,
        beta_rad=0.0,
        geometry=geom,
        attitude_mode=AttitudeMode.LVLH_SUN_TRACKING_1AXIS,
        eclipse_feathering=False,
    )
    assert a_standard > 1.0  # Significant face projection (3 * sin(150 deg) = 1.5 m^2)

    # With eclipse feathering:
    a_feathered = compute_solar_array_cross_section(
        true_anomaly_rad=nu_eclipse,
        beta_rad=0.0,
        geometry=geom,
        attitude_mode=AttitudeMode.LVLH_SUN_TRACKING_1AXIS,
        eclipse_feathering=True,
    )
    assert np.isclose(a_feathered, edge_area, atol=1e-4)


def test_orbit_profile_and_averaged_metrics():
    geom = SpacecraftGeometry(mass_kg=150.0, solar_array_area_m2=3.0)

    profile = compute_orbit_profile(
        beta_deg=15.0,
        geometry=geom,
        attitude_mode=AttitudeMode.LVLH_SUN_TRACKING_1AXIS,
        num_points=180,
    )

    assert len(profile["true_anomaly_deg"]) == 180
    assert len(profile["total_area_m2"]) == 180
    assert len(profile["ballistic_coefficient_kg_m2"]) == 180
    assert np.all(profile["total_area_m2"] > 0.0)
    assert np.all(profile["ballistic_coefficient_kg_m2"] > 0.0)

    metrics = compute_orbit_averaged_metrics(
        beta_deg=15.0,
        geometry=geom,
        attitude_mode=AttitudeMode.LVLH_SUN_TRACKING_1AXIS,
        num_points=180,
    )

    assert metrics["mean_total_area_m2"] > metrics["min_total_area_m2"]
    assert metrics["mean_total_area_m2"] < metrics["max_total_area_m2"]
    assert metrics["effective_ballistic_coefficient_kg_m2"] > 0.0
    assert 0.0 <= metrics["eclipse_fraction"] <= 1.0


def test_beta_angle_sweep():
    geom = SpacecraftGeometry(mass_kg=120.0)
    sweep = compute_beta_angle_sweep(
        beta_min_deg=-60.0,
        beta_max_deg=60.0,
        num_steps=13,
        geometry=geom,
        attitude_mode=AttitudeMode.LVLH_SUN_TRACKING_2AXIS,
    )

    assert len(sweep["beta_deg"]) == 13
    assert len(sweep["effective_ballistic_coefficient_kg_m2"]) == 13
    # 2-axis tracking ballistic coefficient should increase as beta moves away from 0 towards +/- 60 deg
    # because velocity is more perpendicular to Sun vector, reducing projected area
    idx_center = 6  # beta = 0
    idx_edge = 0    # beta = -60
    assert sweep["effective_ballistic_coefficient_kg_m2"][idx_edge] > sweep["effective_ballistic_coefficient_kg_m2"][idx_center]


def test_variable_attitude_drag_acceleration_vector():
    geom = SpacecraftGeometry(mass_kg=150.0)
    r_eci = np.array([EARTH_RADIUS_KM + 400.0, 0.0, 0.0])
    v_eci = np.array([0.0, 7.67, 0.0])
    sun_eci = np.array([1.5e8, 0.0, 0.0])

    a_drag, tele = compute_variable_attitude_drag_acceleration(
        r_eci_km=r_eci,
        v_eci_km_s=v_eci,
        sun_eci_km=sun_eci,
        geometry=geom,
        attitude_mode=AttitudeMode.LVLH_SUN_TRACKING_1AXIS,
        kp=QUIET_KP,
        f10_7=QUIET_F10_7,
    )

    # Acceleration should directly oppose relative velocity (negative along y)
    assert a_drag[1] < 0.0
    assert np.linalg.norm(a_drag) > 0.0
    assert "ballistic_coefficient_kg_m2" in tele
    assert "total_area_m2" in tele
    assert tele["total_area_m2"] > 0.0

    # Test high altitude (> 1200 km) cutoff
    r_high = np.array([EARTH_RADIUS_KM + 1500.0, 0.0, 0.0])
    a_high, tele_high = compute_variable_attitude_drag_acceleration(
        r_eci_km=r_high,
        v_eci_km_s=v_eci,
        sun_eci_km=sun_eci,
        geometry=geom,
    )
    assert np.allclose(a_high, 0.0)
    assert tele_high.get("status") == "above_atmosphere"


def test_sentman_free_molecular_cd_bounds():
    cd_face = compute_sentman_plate_cd(incidence_angle_rad=0.0)      # Face-on (incidence = 0)
    cd_grazing = compute_sentman_plate_cd(incidence_angle_rad=np.pi / 2.0)  # Grazing (incidence = 90 deg)

    assert 2.0 <= cd_face <= 2.5
    assert 1.9 <= cd_grazing <= 2.1
    assert cd_face >= cd_grazing
