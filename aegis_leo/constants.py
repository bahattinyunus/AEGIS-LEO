"""
AEGIS-LEO: Physical & Astrodynamical Constants
Reference frame: WGS84 / GRS80 / NASA SP-8000 series
"""

import numpy as np

# Earth Geodetic & Gravitational Parameters (WGS84)
EARTH_RADIUS_KM: float = 6378.137          # Earth equatorial radius (km)
EARTH_RADIUS_M: float = 6378137.0          # Earth equatorial radius (meters)
EARTH_FLATTENING: float = 1.0 / 298.257223563
EARTH_POLAR_RADIUS_KM: float = 6356.7523   # Earth polar radius (km)
EARTH_MU: float = 398600.4418              # Standard gravitational parameter (km^3 / s^2)
EARTH_MU_SI: float = 3.986004418e14        # Standard gravitational parameter (m^3 / s^2)
EARTH_J2: float = 1.08262668e-3            # Earth J2 zonal harmonic (oblateness)
EARTH_ROTATION_RATE_RAD_S: float = 7.292115e-5  # Earth rotation rate (rad/s)

# Standard Atmosphere & Drag Reference Parameters
SEA_LEVEL_DENSITY: float = 1.225           # kg / m^3 at sea level
SPEED_OF_LIGHT: float = 299792458.0        # m/s
SOLAR_CONSTANT: float = 1361.0             # W/m^2 at 1 AU
STANDARD_GRAVITY: float = 9.80665          # m/s^2 (g0 for Tsiolkovsky equation)

# Space Weather Baselines (NOAA SWPC / NASA OMNIWeb)
QUIET_KP: float = 1.5                      # Quiet geomagnetic index
MODERATE_KP: float = 4.0                   # Unsettled / Active index
STORM_G1_KP: float = 5.0                   # Minor storm
STORM_G2_KP: float = 6.0                   # Moderate storm
STORM_G3_KP: float = 7.0                   # Strong storm
STORM_G4_KP: float = 8.0                   # Severe storm
STORM_G5_KP: float = 9.0                   # Extreme storm (e.g. May 2024 / Carrington)

QUIET_F10_7: float = 70.0                  # Solar flux units (SFU, 10^-22 W/(m^2 Hz))
MODERATE_F10_7: float = 120.0              # Typical mid-solar-cycle SFU
ACTIVE_F10_7: float = 200.0                # High solar activity SFU
EXTREME_F10_7: float = 320.0               # Peak solar storm SFU

# LEO Regime Boundaries (km)
LEO_MIN_ALTITUDE_KM: float = 160.0
LEO_MAX_ALTITUDE_KM: float = 2000.0

# Risk & Compliance Thresholds (NASA-STD-8719.14)
COLLISION_PROBABILITY_THRESHOLD: float = 1e-4   # 1 in 10,000 threshold for maneuver
DEBRIS_CRITICAL_DISTANCE_KM: float = 1.0        # Red alert screening radius (km)
DEBRIS_WARNING_DISTANCE_KM: float = 5.0         # Yellow alert screening radius (km)
MAX_LEO_LIFETIME_YEARS: float = 25.0            # Legacy 25-year rule
UPDATED_FCC_LIFETIME_YEARS: float = 5.0         # Modern 5-year post-mission rule
