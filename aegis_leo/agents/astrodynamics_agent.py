"""
Agent 4: Astrodynamics & Maneuver Trajectory Agent
Synthesizes Vis-Viva phasing burns, continuous low-thrust arcs, and optimal Delta-v trajectories.
"""

from typing import Dict, Any
from aegis_leo.agents.base_agent import BaseSwarmAgent, AgentRole, AgentStatus, SwarmMessage
from aegis_leo.ai.edge_optimizer import EdgeManeuverOptimizer


class AstrodynamicsAgent(BaseSwarmAgent):
    """
    Computes optimal, fuel-minimizing orbital burn vectors for evasive action.
    """

    def __init__(self, blackboard):
        super().__init__(AgentRole.ASTRODYNAMICS, "AstrodynamicsAgent-04", blackboard)
        self.optimizer = EdgeManeuverOptimizer()

    def execute_cycle(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        self.cycle_count += 1
        alt = telemetry.get("altitude_km", 550.0)
        inc = telemetry.get("inclination_deg", 53.0)
        miss_km = telemetry.get("miss_distance_km", 0.45)
        kp = telemetry.get("kp", 1.33)

        risk_state = self.blackboard.get("conjunction_risk", {})
        is_threat = risk_state.get("risk_level") in ["CRITICAL", "HIGH"]

        if is_threat or kp >= 5.0:
            plan = self.optimizer.plan_autonomous_avoidance(
                current_altitude_km=alt,
                inclination_deg=inc,
                initial_miss_distance_km=miss_km,
                storm_kp=kp,
            )
            pb = plan["proactive_burn"]
            decision = (
                f"Evasive phasing burn synthesized: Δv={pb['delta_v_m_s']:.3f} m/s, "
                f"fuel={pb['propellant_consumed_kg']:.4f} kg, clearance={pb['resulting_separation_km']:.2f} km."
            )
            self.status = AgentStatus.VOTING
            payload = {
                "maneuver_needed": True,
                "strategy": pb["strategy"],
                "delta_v_m_s": pb["delta_v_m_s"],
                "propellant_kg": pb["propellant_consumed_kg"],
                "resulting_clearance_km": pb["resulting_separation_km"],
                "fuel_savings_pct": plan["efficiency_metrics"]["fuel_savings_percentage"],
            }
        else:
            decision = "Nominal orbit maintained. No maneuver required."
            self.status = AgentStatus.ACTIVE
            payload = {"maneuver_needed": False, "delta_v_m_s": 0.0}

        self.last_decision = decision
        self.blackboard.post("maneuver_plan", payload, self.name)
        if payload.get("maneuver_needed"):
            self.blackboard.broadcast(
                SwarmMessage(self.name, "MissionCommanderAgent-09", "MANEUVER_PROPOSAL", payload, priority=2)
            )

        return {"agent": self.name, "status": self.status.value, "decision": decision, "data": payload}
