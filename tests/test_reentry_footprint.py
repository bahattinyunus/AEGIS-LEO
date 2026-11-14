"""
Unit Tests for AEGIS-LEO Reentry Ground Dispersion & Casualty Risk Footprint
Tests Monte Carlo dispersion calculations, Point Nemo SPOUA geometry,
1-sigma/2-sigma/3-sigma ellipse contours, and NASA-STD-8719.14B compliance.
"""

import math
import pytest
import numpy as np

from aegis_leo.physics.reentry_footprint import (
    PointNemo,
    DebrisFragment,
    ReentryDispersionEllipse,
    CasualtyRiskReport,
    ReentryFootprintResult,
    create_default_debris_catalog,
    haversine_distance_km,
    enu_to_geodetic,
    geodetic_to_enu,
    compute_dispersion_ellipses,
    evaluate_casualty_risk,
    simulate_reentry_monte_carlo,
    generate_footprint_geojson,
    format_footprint_summary,
)


def test_point_nemo_geography():
    """Verify Point Nemo coordinates and distances to the 3 nearest landmasses (~2,688 km)."""
    assert math.isclose(PointNemo.LATITUDE_DEG, -48.876, abs_tol=1e-3)
    assert math.isclose(PointNemo.LONGITUDE_DEG, -123.393, abs_tol=1e-3)

    for land in PointNemo.NEAREST_LANDMASSES:
        dist = haversine_distance_km(
            PointNemo.LATITUDE_DEG,
            PointNemo.LONGITUDE_DEG,
            land.latitude_deg,
            land.longitude_deg,
        )
        # All three equidistant landmasses must be ~2,685 to ~2,700 km away
        assert 2650.0 < dist < 2750.0, f"{land.name} distance {dist:.1f} km out of expected range"


def test_geodetic_enu_conversions():
    """Test round-trip conversions between local ENU offsets and geodetic coordinates."""
    lat0, lon0 = PointNemo.LATITUDE_DEG, PointNemo.LONGITUDE_DEG

    test_offsets = [
        (0.0, 0.0),
        (150.0, 75.0),
        (-200.0, -100.0),
        (500.0, -350.0),
    ]

    for north_km, east_km in test_offsets:
        lat, lon = enu_to_geodetic(lat0, lon0, north_km, east_km)
        n_calc, e_calc = geodetic_to_enu(lat0, lon0, lat, lon)
        assert math.isclose(north_km, n_calc, abs_tol=1e-6)
        assert math.isclose(east_km, e_calc, abs_tol=1e-6)


def test_haversine_distance():
    """Test great circle distance calculation."""
    # Distance from equator (0, 0) to (0, 90) should be approx quarter circumference ~ 10,018 km
    dist = haversine_distance_km(0.0, 0.0, 0.0, 90.0)
    assert 10000.0 < dist < 10050.0

    # Distance to self should be 0
    assert haversine_distance_km(-48.876, -123.393, -48.876, -123.393) == 0.0


def test_default_debris_catalog():
    """Test default debris fragment catalog generation."""
    catalog = create_default_debris_catalog()
    assert len(catalog) >= 5

    total_mass = sum(f.mass_kg * f.count for f in catalog)
    assert 50.0 < total_mass < 200.0  # Surviving mass of typical 260 kg smallsat

    for frag in catalog:
        assert frag.ballistic_coefficient_kg_m2 > 10.0
        assert frag.casualty_area_m2 > 0.0
        assert frag.count >= 1


def test_simulate_reentry_monte_carlo():
    """Test Monte Carlo reentry simulation execution and result properties."""
    result = simulate_reentry_monte_carlo(
        nominal_altitude_km=120.0,
        nominal_speed_km_s=7.70,
        nominal_fpa_deg=-1.50,
        num_trials=800,
        seed=42,
    )

    assert isinstance(result, ReentryFootprintResult)
    assert result.num_trials == 800
    assert result.std_downrange_km > 50.0
    assert result.std_crossrange_km > 1.0
    assert result.std_downrange_km > result.std_crossrange_km  # Downrange dispersion dominates crossrange

    # Ellipses verification
    assert "1_sigma" in result.ellipses
    assert "2_sigma" in result.ellipses
    assert "3_sigma" in result.ellipses

    e1 = result.ellipses["1_sigma"]
    e2 = result.ellipses["2_sigma"]
    e3 = result.ellipses["3_sigma"]

    # Scaling checks
    assert math.isclose(e2.semi_major_km, 2.0 * e1.semi_major_km, rel_tol=1e-5)
    assert math.isclose(e3.semi_major_km, 3.0 * e1.semi_major_km, rel_tol=1e-5)
    assert math.isclose(e2.area_km2, 4.0 * e1.area_km2, rel_tol=1e-4)
    assert math.isclose(e3.area_km2, 9.0 * e1.area_km2, rel_tol=1e-4)

    # Containment checks (approximate statistical bounds)
    assert 30.0 < e1.empirical_containment_pct < 50.0
    assert 78.0 < e2.empirical_containment_pct < 94.0
    assert 94.0 < e3.empirical_containment_pct <= 100.0


def test_casualty_risk_and_compliance():
    """Verify casualty risk calculation and NASA-STD-8719.14B Requirement 4.7-1 compliance."""
    result = simulate_reentry_monte_carlo(num_trials=500, seed=99)
    report = result.casualty_risk

    assert isinstance(report, CasualtyRiskReport)
    assert report.casualty_expectation_ec < 1.0e-4  # Must meet NASA threshold
    assert report.nasa_req_4_7_compliant is True
    assert report.spoua_containment_compliant is True
    assert report.standoff_margin_3sigma_km > 1500.0  # Safely far from nearest land
    assert report.risk_level in ["NEGLIGIBLE", "NOMINAL"]
    assert "COMPLIANT" in report.summary


def test_point_in_ellipse_containment():
    """Test ReentryDispersionEllipse.contains_point method."""
    result = simulate_reentry_monte_carlo(num_trials=400, seed=123)
    e1 = result.ellipses["1_sigma"]
    e3 = result.ellipses["3_sigma"]

    center_lat = result.nominal_aimpoint["latitude_deg"]
    center_lon = result.nominal_aimpoint["longitude_deg"]

    # Center must always be contained
    assert e1.contains_point(center_lat, center_lon, center_lat, center_lon) is True
    assert e3.contains_point(center_lat, center_lon, center_lat, center_lon) is True

    # Far away point (e.g. Equator) must NOT be contained
    assert e3.contains_point(0.0, 0.0, center_lat, center_lon) is False


def test_to_dict_and_to_geojson():
    """Test serialization to dictionary and GeoJSON format."""
    result = simulate_reentry_monte_carlo(num_trials=300, seed=7)

    # to_dict
    d = result.to_dict()
    assert "nominal_aimpoint" in d
    assert "ellipses" in d
    assert "casualty_risk" in d
    assert "fragments" in d
    assert d["ellipses"]["1_sigma"]["semi_major_km"] > 0

    # to_geojson
    geo = result.to_geojson()
    assert geo["type"] == "FeatureCollection"
    assert len(geo["features"]) >= 4  # 1 aimpoint + 3 ellipses + sample points

    # Check polygon structure
    poly_features = [f for f in geo["features"] if f["geometry"]["type"] == "Polygon"]
    assert len(poly_features) == 3
    for pf in poly_features:
        coords = pf["geometry"]["coordinates"][0]
        assert len(coords) >= 70
        # Closed ring check
        assert coords[0] == coords[-1]


def test_format_footprint_summary():
    """Test text summary formatter."""
    result = simulate_reentry_monte_carlo(num_trials=300, seed=1)
    summary_text = format_footprint_summary(result)

    assert "AEGIS-LEO: CONTROLLED ATMOSPHERIC RE-ENTRY" in summary_text
    assert "Point Nemo" in summary_text
    assert "1-sigma" in summary_text
    assert "2-sigma" in summary_text
    assert "3-sigma" in summary_text
    assert "NASA-STD-8719.14B Requirement 4.7-1" in summary_text
    assert "COMPLIANT" in summary_text


def test_custom_debris_spectrum():
    """Test simulation with custom high-BC debris fragment (e.g. dense titanium pressure vessel)."""
    custom_frags = [
        DebrisFragment(
            name="Dense Titanium Apogee Tank",
            mass_kg=45.0,
            drag_area_m2=0.15,
            ballistic_coefficient_kg_m2=250.0,
            casualty_area_m2=1.5,
            survives_reentry=True,
            count=1,
        )
    ]

    result = simulate_reentry_monte_carlo(
        num_trials=400,
        seed=55,
        debris_catalog=custom_frags,
    )

    assert result.casualty_risk.total_casualty_area_m2 == 1.5
    assert result.casualty_risk.nasa_req_4_7_compliant is True
