"""
CCSDS Orbit Ephemeris Message (OEM) & Orbit Parameter Message (OPM) Generator
Compliant with CCSDS 502.0-B-2 standard for space data system interoperability.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import numpy as np


def generate_ccsds_oem(
    object_name: str,
    object_id: str,
    positions_km: np.ndarray,
    velocities_km_s: np.ndarray,
    time_seconds: np.ndarray,
    start_epoch: Optional[datetime] = None,
    ref_frame: str = "EME2000",
) -> str:
    """
    Format state vectors into CCSDS OEM ASCII format.
    """
    epoch0 = start_epoch or datetime.now(timezone.utc)
    stop_epoch = epoch0 + timedelta(seconds=float(time_seconds[-1]))

    header_lines = [
        "CCSDS_OEM_VERS = 2.0",
        f"CREATION_DATE  = {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')}",
        "ORIGINATOR     = AEGIS-LEO SSA SHIELD (TEAM AYAZ)",
        "",
        "META_START",
        f"OBJECT_NAME          = {object_name}",
        f"OBJECT_ID            = {object_id}",
        "CENTER_NAME          = EARTH",
        f"REF_FRAME            = {ref_frame}",
        "TIME_SYSTEM          = UTC",
        f"START_TIME           = {epoch0.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}",
        f"STOP_TIME            = {stop_epoch.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}",
        "META_STOP",
        "",
    ]

    data_lines = []
    for i, t in enumerate(time_seconds):
        t_dt = epoch0 + timedelta(seconds=float(t))
        t_str = t_dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
        px, py, pz = positions_km[i]
        vx, vy, vz = velocities_km_s[i]
        data_lines.append(
            f"{t_str} {px:14.6f} {py:14.6f} {pz:14.6f} {vx:12.6f} {vy:12.6f} {vz:12.6f}"
        )

    return "\n".join(header_lines + data_lines)


def generate_ccsds_opm(
    object_name: str,
    object_id: str,
    epoch: datetime,
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    mass_kg: float,
    cd: float,
    drag_area_m2: float,
) -> str:
    """
    Format single state vector into CCSDS Orbit Parameter Message (OPM).
    """
    lines = [
        "CCSDS_OPM_VERS = 2.0",
        f"CREATION_DATE  = {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')}",
        "ORIGINATOR     = AEGIS-LEO MISSION CONTROL",
        "",
        "META_START",
        f"OBJECT_NAME          = {object_name}",
        f"OBJECT_ID            = {object_id}",
        "CENTER_NAME          = EARTH",
        "REF_FRAME            = EME2000",
        "TIME_SYSTEM          = UTC",
        "META_STOP",
        "",
        f"EPOCH                = {epoch.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]}",
        f"X                    = {r_eci_km[0]:14.6f} [km]",
        f"Y                    = {r_eci_km[1]:14.6f} [km]",
        f"Z                    = {r_eci_km[2]:14.6f} [km]",
        f"X_DOT                = {v_eci_km_s[0]:12.6f} [km/s]",
        f"Y_DOT                = {v_eci_km_s[1]:12.6f} [km/s]",
        f"Z_DOT                = {v_eci_km_s[2]:12.6f} [km/s]",
        "",
        f"MASS                 = {mass_kg:10.2f} [kg]",
        f"SOLAR_RAD_AREA       = {drag_area_m2:10.2f} [m**2]",
        f"DRAG_AREA            = {drag_area_m2:10.2f} [m**2]",
        f"DRAG_COEFF           = {cd:10.2f}",
    ]
    return "\n".join(lines)
