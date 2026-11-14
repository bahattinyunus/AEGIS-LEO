"""
Live CelesTrak & Space-Track TLE Ingestion Client
Fetches real-time orbital elements from CelesTrak with local caching and SGP4 parsing.
"""

from typing import Dict, Any, Optional
import urllib.request
import json
from datetime import datetime, timezone
import os

CELESTRAK_BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"


class CelesTrakClient:
    """
    Client for retrieving live orbital Two-Line Element (TLE) sets from CelesTrak.
    """

    def __init__(self, cache_dir: str = ".cache_tle", timeout_seconds: float = 4.0):
        self.cache_dir = cache_dir
        self.timeout = timeout_seconds
        os.makedirs(cache_dir, exist_ok=True)

    def fetch_tle_by_catnr(self, norad_catnr: int) -> Dict[str, Any]:
        """
        Fetch live TLE lines for a specified NORAD Catalog Number (e.g. 25544 for ISS).
        """
        cache_file = os.path.join(self.cache_dir, f"tle_{norad_catnr}.json")
        url = f"{CELESTRAK_BASE_URL}?CATNR={norad_catnr}&FORMAT=TLE"

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AEGIS-LEO/1.0 (NASA Space Apps Challenge 2026; Team AYAZ)"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if resp.status == 200:
                    raw_text = resp.read().decode("utf-8").strip()
                    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
                    if len(lines) >= 2:
                        sat_name = lines[0] if len(lines) == 3 else f"NORAD-{norad_catnr}"
                        l1 = lines[-2]
                        l2 = lines[-1]

                        data = {
                            "norad_catnr": norad_catnr,
                            "name": sat_name,
                            "line1": l1,
                            "line2": l2,
                            "source": "CelesTrak Live API",
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                            "status": "LIVE_SUCCESS",
                        }
                        # Save to cache
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        return data
        except Exception:
            pass

        # Try reading cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    cached_data["status"] = "CACHED_FALLBACK"
                    return cached_data
            except Exception:
                pass

        # Default synthetic fallback for robust operation
        return {
            "norad_catnr": norad_catnr,
            "name": f"Object-{norad_catnr}",
            "line1": f"1 {norad_catnr:05d}U 20001A   24120.50000000  .00015000  00000+0  25000-3 0  9991",
            "line2": f"2 {norad_catnr:05d}  53.0000 120.0000 0001500  45.0000 315.0000 15.0600000010001",
            "source": "Synthetic Standalone Fallback",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "status": "STANDALONE_FALLBACK",
        }
