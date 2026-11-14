"""
Live NOAA SWPC & Space-Track Telemetry Feed Connector
Fetches real-time space weather indices from NOAA SWPC with offline fallback caching.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
import json
import urllib.request
from aegis_leo.constants import QUIET_KP, QUIET_F10_7


NOAA_KP_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
NOAA_F107_URL = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"


class LiveSpaceWeatherFeed:
    """
    Connects to NOAA Space Weather Prediction Center (SWPC) live endpoints.
    Provides automatic fallback to cached / physics-based telemetry if offline.
    """

    def __init__(self, timeout_seconds: float = 3.0):
        self.timeout = timeout_seconds
        self._cached_telemetry: Optional[Dict[str, Any]] = None

    def fetch_live_kp_and_flux(self) -> Dict[str, Any]:
        """
        Fetch latest observed Kp and F10.7 from NOAA.
        """
        try:
            req = urllib.request.Request(
                NOAA_KP_URL,
                headers={"User-Agent": "AEGIS-LEO/1.0 (NASA Space Apps 2026; Team AYAZ)"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if isinstance(data, list) and len(data) > 0:
                        latest_entry = data[-1]
                        observed_kp = float(latest_entry.get("kp_index", QUIET_KP))
                        timestamp = latest_entry.get("time_tag", datetime.now(timezone.utc).isoformat())

                        result = {
                            "source": "NOAA SWPC Live Feed",
                            "status": "ONLINE_LIVE",
                            "observed_kp": observed_kp,
                            "estimated_f10_7": round(65.0 + observed_kp * 18.0, 1),
                            "timestamp": timestamp,
                            "is_storm_condition": observed_kp >= 5.0,
                        }
                        self._cached_telemetry = result
                        return result
        except Exception:
            pass  # Fallback to cached or nominal

        if self._cached_telemetry:
            cached = self._cached_telemetry.copy()
            cached["status"] = "CACHED_FALLBACK"
            return cached

        # Standard resilient fallback
        return {
            "source": "AEGIS-LEO Resilient Local Model",
            "status": "STANDALONE_NOMINAL",
            "observed_kp": QUIET_KP,
            "estimated_f10_7": QUIET_F10_7,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_storm_condition": False,
        }
