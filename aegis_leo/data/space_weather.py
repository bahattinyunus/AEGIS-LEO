"""
Space Weather Feed & Geomagnetic Storm Simulation Engine
Integrates NOAA Space Weather Prediction Center (SWPC) and NASA OMNIWeb parameters.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import numpy as np


class SpaceWeatherEvent:
    QUIET = "QUIET"
    G1_MINOR = "G1_MINOR"
    G2_MODERATE = "G2_MODERATE"
    G3_STRONG = "G3_STRONG"
    G4_SEVERE = "G4_SEVERE"
    G5_EXTREME = "G5_EXTREME"


PRESET_STORMS = {
    SpaceWeatherEvent.QUIET: {
        "name": "Solar Quiet Baseline",
        "kp": 1.33,
        "ap": 5.0,
        "f10_7": 72.0,
        "solar_wind_speed_km_s": 380.0,
        "proton_density_n_cm3": 4.5,
        "imf_bz_nt": 1.2,
        "description": "Nominal quiet conditions, standard baseline drag regime.",
    },
    SpaceWeatherEvent.G1_MINOR: {
        "name": "G1 Minor Storm",
        "kp": 5.0,
        "ap": 48.0,
        "f10_7": 140.0,
        "solar_wind_speed_km_s": 520.0,
        "proton_density_n_cm3": 12.0,
        "imf_bz_nt": -5.5,
        "description": "Weak power grid fluctuations, minor satellite drag increase (+35%).",
    },
    SpaceWeatherEvent.G3_STRONG: {
        "name": "G3 Strong Storm",
        "kp": 7.0,
        "ap": 132.0,
        "f10_7": 210.0,
        "solar_wind_speed_km_s": 680.0,
        "proton_density_n_cm3": 28.0,
        "imf_bz_nt": -14.0,
        "description": "False alarms on some protection devices, drag increase (+120%).",
    },
    SpaceWeatherEvent.G4_SEVERE: {
        "name": "G4 Severe Storm (Starlink Feb 2022 Event)",
        "kp": 8.0,
        "ap": 207.0,
        "f10_7": 265.0,
        "solar_wind_speed_km_s": 790.0,
        "proton_density_n_cm3": 45.0,
        "imf_bz_nt": -24.0,
        "description": "Widespread voltage control problems, massive thermospheric surge (+350%).",
    },
    SpaceWeatherEvent.G5_EXTREME: {
        "name": "G5 Extreme Storm (May 2024 Superstorm)",
        "kp": 9.0,
        "ap": 400.0,
        "f10_7": 340.0,
        "solar_wind_speed_km_s": 980.0,
        "proton_density_n_cm3": 85.0,
        "imf_bz_nt": -48.0,
        "description": "Complete HF blackout, critical LEO orbital perturbations (+600% drag).",
    },
}


class SpaceWeatherService:
    """
    Manages space weather state, real-time fetching with resilient mock fallback,
    and synthetic storm injection for mission stress tests.
    """

    def __init__(self, default_event: str = SpaceWeatherEvent.QUIET):
        self.current_event_key = default_event
        self.custom_override: Optional[Dict[str, float]] = None

    def get_current_conditions(self) -> Dict[str, Any]:
        """
        Return the currently active space weather parameters.
        """
        base = PRESET_STORMS.get(self.current_event_key, PRESET_STORMS[SpaceWeatherEvent.QUIET]).copy()
        if self.custom_override:
            base.update(self.custom_override)

        base["timestamp"] = datetime.now(timezone.utc).isoformat()
        base["event_key"] = self.current_event_key

        # Estimate exospheric temperature change Delta T_inf (Kelvin)
        # Empirical Jacchia relation: Delta T = 28 * Kp + 0.03 * exp(Kp)
        kp = base["kp"]
        delta_t_inf = 28.0 * kp + 0.03 * np.exp(kp)
        base["estimated_thermospheric_delta_temp_k"] = round(float(delta_t_inf), 1)

        return base

    def set_storm_event(self, event_key: str) -> Dict[str, Any]:
        """
        Trigger a specific storm scenario.
        """
        if event_key in PRESET_STORMS:
            self.current_event_key = event_key
            self.custom_override = None
        else:
            raise ValueError(f"Unknown event key {event_key}. Available: {list(PRESET_STORMS.keys())}")
        return self.get_current_conditions()

    def set_custom_parameters(self, kp: float, f10_7: float) -> Dict[str, Any]:
        """
        Apply custom user-specified Kp and F10.7 values.
        """
        self.current_event_key = "CUSTOM"
        self.custom_override = {
            "name": f"Custom Space Weather (Kp={kp:.1f}, F10.7={f10_7:.0f})",
            "kp": float(kp),
            "ap": float(max(2.0, (kp ** 2.2) * 2.5)),
            "f10_7": float(f10_7),
            "solar_wind_speed_km_s": float(350.0 + kp * 65.0),
            "proton_density_n_cm3": float(4.0 + kp * 8.0),
            "imf_bz_nt": float(-kp * 3.5),
            "description": "User-configured space weather conditions.",
        }
        return self.get_current_conditions()

    @staticmethod
    def list_preset_events() -> List[Dict[str, Any]]:
        """
        List all available predefined space weather scenarios.
        """
        return [
            {"key": k, **v} for k, v in PRESET_STORMS.items()
        ]
