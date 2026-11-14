"""
Ground Track & Communication Blackout Analysis
Calculates 2D sub-satellite ground track, ground station visibility passes,
and ionospheric communication blackout windows during geomagnetic storms.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    EARTH_ROTATION_RATE_RAD_S,
    EARTH_FLATTENING,
)
from aegis_leo.physics.coordinates import eci_to_geodetic


# World Major & Space Agency Ground Stations
GROUND_STATIONS = [
    {
        "name": "Ankara TÜBİTAK UZAY Ground Station",
        "country": "Türkiye",
        "lat": 39.8912,
        "lon": 32.7785,
        "min_elevation_deg": 5.0,
    },
    {
        "name": "Svalbard Arctic Satellite Station (Svalsat)",
        "country": "Norway",
        "lat": 78.2300,
        "lon": 15.4000,
        "min_elevation_deg": 5.0,
    },
    {
        "name": "Goldstone SSA & DSN Complex",
        "country": "USA",
        "lat": 35.4267,
        "lon": -116.8899,
        "min_elevation_deg": 5.0,
    },
    {
        "name": "Madrid Ground Station (Robledo)",
        "country": "Spain",
        "lat": 40.4314,
        "lon": -4.2480,
        "min_elevation_deg": 5.0,
    },
    {
        "name": "Canberra Deep Space Station (Tidbinbilla)",
        "country": "Australia",
        "lat": -35.4014,
        "lon": 148.9817,
        "min_elevation_deg": 5.0,
    },
]


def calculate_ground_track(
    positions_eci_km: np.ndarray,
    time_seconds: np.ndarray,
    start_gmst_rad: float = 0.0,
) -> List[Dict[str, float]]:
    """
    Convert time series of ECI positions into geographic Latitude, Longitude, and Altitude.
    """
    track = []
    for i, t in enumerate(time_seconds):
        pos = positions_eci_km[i]
        gmst = start_gmst_rad + EARTH_ROTATION_RATE_RAD_S * t
        lat, lon, alt = eci_to_geodetic(pos, gmst_rad=gmst)

        # Wrap longitude to [-180, 180]
        lon = ((lon + 180.0) % 360.0) - 180.0

        track.append({
            "time_seconds": float(t),
            "latitude_deg": round(float(lat), 3),
            "longitude_deg": round(float(lon), 3),
            "altitude_km": round(float(alt), 2),
        })

    return track


def compute_ground_station_visibility(
    sat_lat: float,
    sat_lon: float,
    sat_alt_km: float,
    station_lat: float,
    station_lon: float,
    min_elevation_deg: float = 5.0,
) -> Dict[str, Any]:
    """
    Compute great circle distance and elevation angle from ground station to satellite.
    """
    phi1, lambda1 = np.radians(station_lat), np.radians(station_lon)
    phi2, lambda2 = np.radians(sat_lat), np.radians(sat_lon)

    # Great circle central angle gamma
    delta_lambda = lambda2 - lambda1
    cos_gamma = np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(delta_lambda)
    gamma_rad = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
    ground_dist_km = gamma_rad * EARTH_RADIUS_KM

    # Slant range d
    r_sat = EARTH_RADIUS_KM + sat_alt_km
    slant_range_km = np.sqrt(EARTH_RADIUS_KM**2 + r_sat**2 - 2.0 * EARTH_RADIUS_KM * r_sat * np.cos(gamma_rad))

    # Elevation angle beta
    sin_beta = (r_sat * np.cos(gamma_rad) - EARTH_RADIUS_KM) / max(slant_range_km, 1e-4)
    elevation_deg = float(np.degrees(np.arcsin(np.clip(sin_beta, -1.0, 1.0))))

    is_visible = elevation_deg >= min_elevation_deg

    return {
        "elevation_deg": round(elevation_deg, 2),
        "slant_range_km": round(float(slant_range_km), 1),
        "ground_distance_km": round(float(ground_dist_km), 1),
        "is_visible": bool(is_visible),
    }


def analyze_communication_blackouts(
    ground_track: List[Dict[str, float]],
    storm_kp: float = 8.0,
) -> Dict[str, Any]:
    """
    Analyze ground station pass availability and quantify communication blackout risk.
    During severe storms (Kp >= 7), auroral oval expands and polar cap absorption disrupts telemetry.
    """
    total_steps = len(ground_track)
    contact_steps = 0
    blackout_steps = 0

    station_passes = {gs["name"]: 0 for gs in GROUND_STATIONS}

    # Severe storm auroral absorption latitude threshold
    # At Kp=8, auroral zone expands down to ~55 deg magnetic latitude
    auroral_cutoff_lat = max(52.0, 68.0 - (storm_kp - 3.0) * 3.0)

    for pt in ground_track:
        lat = pt["latitude_deg"]
        lon = pt["longitude_deg"]
        alt = pt["altitude_km"]

        has_los = False
        in_auroral_absorption = abs(lat) >= auroral_cutoff_lat and storm_kp >= 6.5

        for gs in GROUND_STATIONS:
            vis = compute_ground_station_visibility(lat, lon, alt, gs["lat"], gs["lon"])
            if vis["is_visible"]:
                has_los = True
                station_passes[gs["name"]] += 1

        if has_los:
            if in_auroral_absorption:
                blackout_steps += 1
            else:
                contact_steps += 1
        else:
            blackout_steps += 1

    contact_ratio = contact_steps / max(total_steps, 1)
    blackout_ratio = blackout_steps / max(total_steps, 1)

    return {
        "storm_kp": storm_kp,
        "auroral_cutoff_lat_deg": round(auroral_cutoff_lat, 1),
        "contact_window_percentage": round(contact_ratio * 100.0, 1),
        "blackout_window_percentage": round(blackout_ratio * 100.0, 1),
        "autonomous_decision_urgency": "CRITICAL" if blackout_ratio > 0.65 else "MODERATE",
        "station_pass_counts": station_passes,
        "recommendation": (
            "Ground-commanded collision avoidance impossible during 70%+ of the orbit due to blackout. "
            "Autonomous onboard Edge-AI collision avoidance is MANDATORY for orbital safety."
        ) if blackout_ratio > 0.60 else "Ground telemetry nominal.",
    }
