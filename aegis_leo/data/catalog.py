"""
Orbital Object Catalog & TLE Repository
Includes active LEO operational satellites and major cataloged orbital debris clouds.
"""

from typing import Dict, Any, List, Optional


SATELLITE_CATALOG: Dict[str, Dict[str, Any]] = {
    "STARLINK_LEO": {
        "norad_id": 44713,
        "name": "Starlink-1007",
        "type": "Communications Megaconstellation",
        "mass_kg": 260.0,
        "area_m2": 2.2,
        "cd": 2.2,
        "remaining_propellant_kg": 15.4,
        "isp_seconds": 250.0,
        "thrust_n": 0.08,
        "altitude_km": 548.0,
        "inclination_deg": 53.0,
        "tle_line1": "1 44713U 19074A   24120.50000000  .00015000  00000+0  35000-3 0  9993",
        "tle_line2": "2 44713  53.0540 120.4500 0001500  45.2000 314.9000 15.06000000250001",
    },
    "ISS_ZARYA": {
        "norad_id": 25544,
        "name": "ISS (Zarya)",
        "type": "Crewed Space Station",
        "mass_kg": 450000.0,
        "area_m2": 2400.0,
        "cd": 2.1,
        "remaining_propellant_kg": 2200.0,
        "isp_seconds": 310.0,
        "thrust_n": 2940.0,
        "altitude_km": 418.0,
        "inclination_deg": 51.64,
        "tle_line1": "1 25544U 98067A   24120.50000000  .00021000  00000+0  28000-3 0  9999",
        "tle_line2": "2 25544  51.6430  95.1200 0004500 120.3000 240.1000 15.50000000450001",
    },
    "SENTINEL_1A": {
        "norad_id": 39634,
        "name": "Sentinel-1A",
        "type": "ESA Earth Observation (SAR)",
        "mass_kg": 2300.0,
        "area_m2": 18.5,
        "cd": 2.2,
        "remaining_propellant_kg": 95.0,
        "isp_seconds": 220.0,
        "thrust_n": 10.0,
        "altitude_km": 693.0,
        "inclination_deg": 98.18,
        "tle_line1": "1 39634U 14016A   24120.50000000  .00001200  00000+0  65000-4 0  9995",
        "tle_line2": "2 39634  98.1820 180.2500 0001400  85.4000 274.7000 14.59000000520002",
    },
    "CUBESAT_3U": {
        "norad_id": 55890,
        "name": "Aegis-DemoSat (3U CubeSat)",
        "type": "Scientific Research Microsatellite",
        "mass_kg": 4.5,
        "area_m2": 0.045,
        "cd": 2.3,
        "remaining_propellant_kg": 0.35,
        "isp_seconds": 180.0,
        "thrust_n": 0.02,
        "altitude_km": 500.0,
        "inclination_deg": 97.4,
        "tle_line1": "1 55890U 23050A   24120.50000000  .00035000  00000+0  52000-3 0  9991",
        "tle_line2": "2 55890  97.4000 155.0000 0002000  60.0000 300.0000 15.20000000110003",
    },
}

DEBRIS_CATALOG: Dict[str, Dict[str, Any]] = {
    "COSMOS_2251_DEB": {
        "norad_id": 34125,
        "name": "Cosmos 2251 Debris (Fragment #125)",
        "source": "Cosmos-Iridium 2009 Collision",
        "estimated_mass_kg": 3.8,
        "cross_section_m2": 0.12,
        "altitude_km": 551.0,
        "inclination_deg": 74.0,
        "tle_line1": "1 34125U 93036AP  24120.50000000  .00022000  00000+0  42000-3 0  9997",
        "tle_line2": "2 34125  74.0200 118.8000 0025000 110.0000 250.0000 15.08000000850004",
    },
    "FENGYUN_1C_DEB": {
        "norad_id": 31114,
        "name": "Fengyun-1C Debris (ASAT Fragment)",
        "source": "2007 ASAT Intercept Test",
        "estimated_mass_kg": 1.2,
        "cross_section_m2": 0.05,
        "altitude_km": 692.0,
        "inclination_deg": 98.6,
        "tle_line1": "1 31114U 99025AP  24120.50000000  .00008500  00000+0  18000-3 0  9992",
        "tle_line2": "2 31114  98.6000 179.8000 0018000  95.0000 265.0000 14.61000000670005",
    },
    "SL16_ROCKET_BODY": {
        "norad_id": 22220,
        "name": "SL-16 R/B (Zenit Upper Stage)",
        "source": "Spent Rocket Body Defunct",
        "estimated_mass_kg": 9000.0,
        "cross_section_m2": 32.0,
        "altitude_km": 840.0,
        "inclination_deg": 71.0,
        "tle_line1": "1 22220U 92076B   24120.50000000  .00000450  00000+0  22000-4 0  9998",
        "tle_line2": "2 22220  71.0000  80.0000 0012000 130.0000 230.0000 14.15000000940006",
    },
}


def get_satellite_by_id(sat_id: str) -> Optional[Dict[str, Any]]:
    return SATELLITE_CATALOG.get(sat_id)


def list_satellites() -> List[Dict[str, Any]]:
    return [{"id": k, **v} for k, v in SATELLITE_CATALOG.items()]


def list_debris() -> List[Dict[str, Any]]:
    return [{"id": k, **v} for k, v in DEBRIS_CATALOG.items()]
