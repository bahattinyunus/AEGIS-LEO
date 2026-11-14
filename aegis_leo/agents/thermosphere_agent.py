"""
Agent 2: Thermosphere Dynamics & Drag Physics Agent
Solves dynamic density field rho(r, t) and projects along-track ephemeris divergence.
"""

from typing import Dict, Any
from aegis_leo.agents.base_agent import BaseSwarmAgent, AgentRole, AgentStatus, SwarmMessage
from aegis_leo.physics.atmosphere import compute_atmospheric_density


class ThermosphereAgent(BaseSwarmAgent):
    """
    Computes real-time dynamic upper-atmosphere expansion and drag perturbations.
    """

    def __init__(self, blackboard):
        super().__init__(AgentRole.THERMOSPHERE_DYNAMICS, "ThermosphereAgent-02", blackboard)

    def execute_cycle(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        self.cycle_count += 1
        alt = telemetry.get("altitude_km", 550.0)
        kp = telemetry.get("kp", 1.33)
        f10_7 = telemetry.get("f10_7", 72.0)

        rho = compute_atmospheric_density(alt, kp=kp, f10_7=f10_7)
        rho_quiet = compute_atmospheric_density(alt, kp=1.33, f10_7=72.0)
        multiplier = rho / max(rho_quiet, 1e-25)

        along_track_drift_km = round(float((kp - 2.0) * 2.3 if kp >= 5.0 else 0.2), 2)
        self.status = AgentStatus.ALERT if multiplier > 2.5 else AgentStatus.ACTIVE

        decision = (
            f"Thermosphere expansion: {multiplier:.1f}x surge at {alt:.0f} km. "
            f"Along-track drift: +{along_track_drift_km} km. Static CDMs corrupted."
        )
        self.last_decision = decision

        payload = {
            "density_kg_m3": rho,
            "density_multiplier": round(multiplier, 2),
            "along_track_drift_km": along_track_drift_km,
            "ballistic_deceleration_scale": round(multiplier * 1.25, 2),
        }

        self.blackboard.post("atmospheric_state", payload, self.name)
        self.blackboard.broadcast(
            SwarmMessage(self.name, "RiskAssessmentAgent-03", "UPDATED_DENSITY_FIELD", payload)
        )

        return {"agent": self.name, "status": self.status.value, "decision": decision, "data": payload}
