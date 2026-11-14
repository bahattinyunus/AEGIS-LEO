"""
Agent 3: A-DRI Risk Assessment & Conjunction Watchdog Agent
Continuously calculates multi-factor A-DRI scores and triggers swarm evasion alerts.
"""

from typing import Dict, Any
from aegis_leo.agents.base_agent import BaseSwarmAgent, AgentRole, AgentStatus, SwarmMessage
from aegis_leo.risk.adri import ADRICalculator, RiskLevel


class RiskAssessmentAgent(BaseSwarmAgent):
    """
    Evaluates real-time A-DRI scores and flags critical conjunctions.
    """

    def __init__(self, blackboard):
        super().__init__(AgentRole.RISK_ASSESSMENT, "RiskAssessmentAgent-03", blackboard)
        self.adri_calc = ADRICalculator()

    def execute_cycle(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        self.cycle_count += 1
        miss_km = telemetry.get("miss_distance_km", 0.45)
        alt = telemetry.get("altitude_km", 550.0)
        inc = telemetry.get("inclination_deg", 53.0)
        kp = telemetry.get("kp", 1.33)
        f10_7 = telemetry.get("f10_7", 72.0)

        # Retrieve drift from blackboard if available
        atm_state = self.blackboard.get("atmospheric_state", {})
        drift_km = atm_state.get("along_track_drift_km", 0.2)

        risk_res = self.adri_calc.evaluate_risk(
            miss_distance_km=miss_km,
            altitude_km=alt,
            inclination_deg=inc,
            kp=kp,
            f10_7=f10_7,
            along_track_disp_km=drift_km,
        )

        level = risk_res["risk_level"]
        score = risk_res["a_dri"]
        self.status = AgentStatus.ALERT if level in [RiskLevel.CRITICAL.value, RiskLevel.HIGH.value] else AgentStatus.ACTIVE

        decision = f"A-DRI={score:.3f} [{level}]. Action command: {risk_res['action_command']}."
        self.last_decision = decision

        payload = {
            "a_dri": score,
            "risk_level": level,
            "action_command": risk_res["action_command"],
            "miss_distance_km": miss_km,
            "c_prob_raw": risk_res["components"]["c_prob_raw"],
        }

        self.blackboard.post("conjunction_risk", payload, self.name)
        if level in [RiskLevel.CRITICAL.value, RiskLevel.HIGH.value]:
            self.blackboard.broadcast(
                SwarmMessage(self.name, "AstrodynamicsAgent-04", "CONJUNCTION_THREAT_DETECTED", payload, priority=3)
            )

        return {"agent": self.name, "status": self.status.value, "decision": decision, "data": payload}
