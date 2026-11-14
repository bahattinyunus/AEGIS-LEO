"""
CCSDS Conjunction Data Message (CDM) Generator & Parser
Follows CCSDS 508.0-B-1 Conjunction Data Message standard for space situational awareness.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import uuid


def generate_cdm_record(
    primary_sat_id: str,
    secondary_obj_id: str,
    miss_distance_km: float,
    relative_speed_km_s: float,
    collision_prob: float,
    altitude_km: float,
    tca_hours_from_now: float = 12.0,
) -> Dict[str, Any]:
    """
    Synthesize a valid CCSDS-formatted Conjunction Data Message dictionary.
    """
    tca_dt = datetime.now(timezone.utc) + timedelta(hours=tca_hours_from_now)
    message_id = f"CDM-AEGIS-{uuid.uuid4().hex[:8].upper()}"

    return {
        "ccsds_cdm_vers": "1.0",
        "message_id": message_id,
        "creation_date": datetime.now(timezone.utc).isoformat(),
        "originator": "AEGIS-LEO SSA Autonomous Shield / Team AYAZ",
        "tca": tca_dt.isoformat(),
        "miss_distance_m": round(miss_distance_km * 1000.0, 2),
        "relative_speed_m_s": round(relative_speed_km_s * 1000.0, 2),
        "collision_probability": collision_prob,
        "primary_object": {
            "object_id": primary_sat_id,
            "object_type": "PAYLOAD",
            "altitude_km": altitude_km,
        },
        "secondary_object": {
            "object_id": secondary_obj_id,
            "object_type": "DEBRIS",
            "altitude_km": altitude_km,
        },
        "covariance_matrix_rtn_m2": {
            "c_rr": 450.0,
            "c_tt": 18500.0,
            "c_nn": 1200.0,
        },
        "alert_status": "CRITICAL" if collision_prob > 1e-4 else "NOMINAL",
    }
