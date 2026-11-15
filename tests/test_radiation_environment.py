"""
Unit tests for the LEO trapped radiation, South Atlantic Anomaly (SAA) & dose engine.
"""

import math

import numpy as np
import pytest

from aegis_leo.constants import EARTH_RADIUS_KM
from aegis_leo.physics.radiation_environment import (
    DIPOLE_B0_NT,
    DIPOLE_CENTER_OFFSET_KM,
    GEOMAG_NORTH_POLE_LAT_DEG,
    GEOMAG_NORTH_POLE_LON_DEG,
    analyze_radiation_environment,
    circular_orbit_ecef_track,
    compute_saa_flux_map,
    geomagnetic_coordinates,
    sep_access_l_shell,
    stormer_cutoff_rigidity_gv,
    trapped_electron_flux,
    trapped_proton_flux,
)


def _ecef(lat_deg, lon_deg, alt_km):
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    r = EARTH_RADIUS_KM + alt_km
    return np.array([r * math.cos(lat) * math.cos(lon), r * math.cos(lat) * math.sin(lon), r * math.sin(lat)])


def test_dipole_field_equator_and_pole():
    axis = _ecef(GEOMAG_NORTH_POLE_LAT_DEG, GEOMAG_NORTH_POLE_LON_DEG, 0.0) / EARTH_RADIUS_KM
    perp = np.cross(axis, np.array([0.0, 0.0, 1.0]))
    perp /= np.linalg.norm(perp)

    g_eq = geomagnetic_coordinates(DIPOLE_CENTER_OFFSET_KM + EARTH_RADIUS_KM * perp)
    assert float(g_eq["b_nt"]) == pytest.approx(DIPOLE_B0_NT, rel=1e-9)
    assert float(g_eq["l_shell"]) == pytest.approx(1.0, rel=1e-9)
    assert float(g_eq["mag_lat_deg"]) == pytest.approx(0.0, abs=1e-9)

    g_pole = geomagnetic_coordinates(DIPOLE_CENTER_OFFSET_KM + EARTH_RADIUS_KM * axis)
    assert float(g_pole["b_nt"]) == pytest.approx(2.0 * DIPOLE_B0_NT, rel=1e-9)
    g_pole_2r = geomagnetic_coordinates(DIPOLE_CENTER_OFFSET_KM + 2.0 * EARTH_RADIUS_KM * axis)
    assert float(g_pole["b_nt"]) / float(g_pole_2r["b_nt"]) == pytest.approx(8.0, rel=1e-9)


def test_field_is_weakest_over_south_atlantic():
    b_saa = float(geomagnetic_coordinates(_ecef(-25.0, -45.0, 550.0))["b_nt"])
    b_pacific = float(geomagnetic_coordinates(_ecef(-25.0, 135.0, 550.0))["b_nt"])
    assert b_saa < 0.8 * b_pacific


def test_saa_is_located_over_south_atlantic_and_grows_with_altitude():
    m550 = compute_saa_flux_map(550.0)
    assert -50.0 <= m550["peak_lat_deg"] <= -15.0
    assert -70.0 <= m550["peak_lon_deg"] <= -10.0
    assert float(trapped_proton_flux(_ecef(-30.0, 150.0, 550.0))) == 0.0
    assert compute_saa_flux_map(800.0)["peak_flux_cm2_s"] > m550["peak_flux_cm2_s"] > compute_saa_flux_map(400.0)["peak_flux_cm2_s"]


def test_electron_flux_storm_enhancement_and_cutoff_rigidity():
    pos = _ecef(-62.0, -30.0, 700.0)
    assert float(trapped_electron_flux(pos, kp=8.0)) > 5.0 * float(trapped_electron_flux(pos, kp=2.0))
    assert float(stormer_cutoff_rigidity_gv(np.array(1.0))) == pytest.approx(14.9)
    assert sep_access_l_shell(8.0) < sep_access_l_shell(0.0)


def test_ecef_track_radius_and_latitude_bound():
    trk = circular_orbit_ecef_track(550.0, 53.0, 6000.0)
    assert np.allclose(np.linalg.norm(trk["r_ecef_km"], axis=1), EARTH_RADIUS_KM + 550.0)
    assert np.max(np.abs(trk["lat_deg"])) <= 53.0 + 1e-6


def test_nominal_leo_report():
    rep = analyze_radiation_environment(altitude_km=550.0, inclination_deg=53.0)
    assert rep.radiation_status == "NOMINAL_RADIATION_ENVIRONMENT"
    assert 0.01 < rep.saa_exposure_fraction < 0.25
    assert rep.saa_passes_per_day > 0.0
    assert rep.saa_peak_lat_deg < 0.0
    assert rep.tid_krad_per_year > 0.0 and rep.seu_per_day > 0.0
    assert 1.0 <= rep.recommended_scrub_interval_s <= 3600.0
    assert rep.mission_tid_krad == pytest.approx(rep.tid_krad_per_year * 5.0, rel=1e-3)
    assert 0 < len(rep.timeline) <= 250
    assert "timeline" in rep.to_dict()


def test_dose_trends_with_altitude_shielding_and_storm():
    low = analyze_radiation_environment(altitude_km=420.0)
    high = analyze_radiation_environment(altitude_km=1200.0)
    assert high.tid_krad_per_year > low.tid_krad_per_year
    assert high.recommended_scrub_interval_s < low.recommended_scrub_interval_s
    thin = analyze_radiation_environment(shielding_mm_al=1.0)
    thick = analyze_radiation_environment(shielding_mm_al=6.0)
    assert thin.tid_krad_per_year > thick.tid_krad_per_year
    quiet = analyze_radiation_environment(altitude_km=693.0, inclination_deg=98.2, kp=2.0)
    storm = analyze_radiation_environment(altitude_km=693.0, inclination_deg=98.2, kp=8.5)
    assert storm.mean_electron_flux_cm2_s > quiet.mean_electron_flux_cm2_s
    assert storm.polar_cap_exposure_fraction > quiet.polar_cap_exposure_fraction


def test_equatorial_orbit_avoids_polar_cap_and_sep():
    polar = analyze_radiation_environment(altitude_km=693.0, inclination_deg=98.2, kp=8.0, sep_flux_pfu=100.0)
    assert polar.radiation_status == "SEP_STORM_EXPOSURE"
    assert polar.sep_mean_flux_cm2_s > 0.0
    eq = analyze_radiation_environment(altitude_km=550.0, inclination_deg=5.0, kp=8.0, sep_flux_pfu=100.0)
    assert eq.polar_cap_exposure_fraction == 0.0 and eq.sep_mean_flux_cm2_s == 0.0


def test_tid_limit_status_and_validation():
    rep = analyze_radiation_environment(altitude_km=1200.0, component_tid_tolerance_krad=1.0, mission_years=10.0)
    assert rep.radiation_status == "TID_LIMIT_EXCEEDED" and rep.tid_margin_factor < 1.0
    with pytest.raises(ValueError):
        analyze_radiation_environment(altitude_km=80.0)
    with pytest.raises(ValueError):
        analyze_radiation_environment(shielding_mm_al=-1.0)
