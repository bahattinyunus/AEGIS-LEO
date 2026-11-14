"""
Spatial Orbital Debris Density Model
Calibrated against NASA ORDEM 3.1 and ESA MASTER flux models.
Models debris concentration peaks across altitude and inclination shells.
"""

from typing import Dict, Any
import numpy as np


def get_spatial_debris_density(altitude_km: float, inclination_deg: float) -> Dict[str, float]:
    """
    Calculate spatial debris density (objects / km^3) and normalized density score [0.0, 1.0].

    Characteristics:
    - Primary peak: 750 - 900 km (Iridium-Cosmos collision zone, Fengyun-1C debris).
    - Secondary peak: 500 - 600 km (Starlink / OneWeb / megaconstellation operating shell).
    - Inclination peaks: SSO (~97-98 deg), ISS band (~51.6 deg), Russian polar (~82 deg).
    """
    alt = float(np.clip(altitude_km, 150.0, 2000.0))
    inc = float(np.clip(inclination_deg, 0.0, 180.0))

    # Altitude distribution (sum of Gaussians over background)
    # Background baseline density in LEO
    base_density = 1.0e-9

    # 800 km major debris peak (highest density in LEO)
    peak_800 = 8.5e-8 * np.exp(-0.5 * ((alt - 800.0) / 60.0)**2)

    # 550 km megaconstellation peak
    peak_550 = 4.2e-8 * np.exp(-0.5 * ((alt - 550.0) / 45.0)**2)

    # 980 km sun-synchronous science peak
    peak_980 = 3.1e-8 * np.exp(-0.5 * ((alt - 980.0) / 50.0)**2)

    # Low altitude cleansing by drag (< 400 km)
    drag_cleansing = np.clip((alt - 200.0) / 200.0, 0.05, 1.0)

    alt_density = (base_density + peak_800 + peak_550 + peak_980) * drag_cleansing

    # Inclination weighting factor
    # SSO peak (97-98 deg)
    inc_sso = 2.4 * np.exp(-0.5 * ((inc - 97.8) / 3.0)**2)
    # ISS band (51.6 deg)
    inc_iss = 1.6 * np.exp(-0.5 * ((inc - 51.6) / 2.5)**2)
    # Polar band (82 deg)
    inc_polar = 1.8 * np.exp(-0.5 * ((inc - 82.5) / 3.0)**2)

    inc_factor = 1.0 + inc_sso + inc_iss + inc_polar

    total_density = float(alt_density * inc_factor)

    # Max expected density in LEO is ~ 2.5e-7 objects / km^3
    max_ref_density = 2.5e-7
    normalized_score = float(np.clip(total_density / max_ref_density, 0.0, 1.0))

    return {
        "spatial_density_objects_per_km3": total_density,
        "normalized_debris_factor": normalized_score,
        "altitude_km": alt,
        "inclination_deg": inc,
    }
