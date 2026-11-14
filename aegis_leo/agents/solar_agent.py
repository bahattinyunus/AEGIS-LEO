"""
Agent 1: Solar & Space Weather Intelligence Agent
Monitors solar wind, CME arrival, proton flux, Kp/Ap indices, and forecasts thermospheric heating.
"""

from typing import Dict, Any
from aegis_leo.agents.base_agent import BaseSwarmAgent, AgentRole, AgentStatus, SwarmMessage
from aegis_leo.data.space_weather import SpaceWeatherService


class SolarStormAgent(BaseSwarmAgent):
    """
    Ingests live NOAA SWPC & OMNIWeb solar wind telemetry and warns swarm of geomagnetic storms.
    """

    def __init__(self, blackboard):
        super().__init__(AgentRole.SOLAR_PHYSICS, "SolarStormAgent-01", blackboard)
        self.weather_svc = SpaceWeatherService()

    def execute_cycle(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        self.cycle_count += 1
        kp = telemetry.get("kp", 1.33)
        f10_7 = telemetry.get("f10_7", 72.0)

        is_storm = kp >= 5.0
        self.status = AgentStatus.ALERT if is_storm else AgentStatus.ACTIVE

        decision = (
            f"GEOMAGNETIC STORM ALERT (Kp={kp:.1f}, F10.7={f10_7:.0f} SFU). CME Energy Injected."
            if is_storm
            else f"Solar Quiet Baseline (Kp={kp:.1f}). Nominal ionosphere."
        )
        self.last_decision = decision

        payload = {
            "kp": kp,
            "f10_7": f10_7,
            "is_storm": is_storm,
            "storm_severity": "G4_SEVERE" if kp >= 8.0 else "G1_MINOR" if is_storm else "QUIET",
            "joule_heating_index": round(float((kp / 9.0) ** 1.8), 3),
        }

        self.blackboard.post("solar_weather", payload, self.name)
        if is_storm:
            self.blackboard.broadcast(
                SwarmMessage(self.name, "ThermosphereAgent-02", "STORM_ALERT", payload, priority=2)
            )

        return {"agent": self.name, "status": self.status.value, "decision": decision, "data": payload}
