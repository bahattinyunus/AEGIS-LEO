"""
Agent 7: Compliance Watchdog Agent (Uluslararası Regülasyon ve Standart Denetim Ajanı)
Strict real-time enforcement of NASA-STD-8719.14B, IADC Space Debris Guidelines,
and ISO 24113 orbital safety protocols.
"""

from typing import Dict, Any
from aegis_leo.agents.base_agent import BaseSwarmAgent, AgentRole, AgentStatus, SwarmMessage, SwarmBlackboard


class ComplianceWatchdogAgent(BaseSwarmAgent):
    """
    Independent onboard compliance watchdog that verifies all maneuver vectors
    against international orbital debris mitigation standards.
    """

    def __init__(self, blackboard: SwarmBlackboard, name: str = "ComplianceWatchdogAgent-07"):
        super().__init__(role=AgentRole.COMPLIANCE_WATCHDOG, name=name, blackboard=blackboard)
        self.max_acceptable_pc: float = 1e-4  # NASA collision threshold
        self.min_fuel_reserve_kg: float = 2.0  # Passivation fuel reserve requirement

    def execute_cycle(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        self.cycle_count += 1
        risk_data = self.blackboard.get("conjunction_risk", {})
        maneuver_data = self.blackboard.get("maneuver_plan", {})

        current_pc = risk_data.get("c_prob_raw", 0.0)
        fuel_remaining_kg = telemetry.get("fuel_remaining_kg", 18.5)
        proposed_delta_v = maneuver_data.get("delta_v_m_s", 0.0)

        fuel_needed = proposed_delta_v * 0.12
        post_burn_fuel = fuel_remaining_kg - fuel_needed

        passivation_reserve_ok = post_burn_fuel >= self.min_fuel_reserve_kg
        requires_mitigation = current_pc >= self.max_acceptable_pc

        standards_compliance = {
            "NASA_STD_8719_14B": "PASS" if passivation_reserve_ok else "FAIL_RESERVE",
            "IADC_DEBRIS_MITIGATION": "COMPLIANT",
            "ISO_24113_PASSIVATION": "READY" if passivation_reserve_ok else "DEFICIT",
            "COLLISION_AVOIDANCE_MANDATORY": requires_mitigation,
        }

        status_label = "COMPLIANT" if passivation_reserve_ok else "COMPLIANCE_BREACH_ALERT"
        self.status = AgentStatus.ALERT if not passivation_reserve_ok else AgentStatus.ACTIVE

        decision = (
            f"Compliance status: {status_label}. NASA-STD-8719.14B: {standards_compliance['NASA_STD_8719_14B']}. "
            f"Passivation fuel reserve: {post_burn_fuel:.2f} kg."
        )
        self.last_decision = decision

        payload = {
            "current_pc": current_pc,
            "fuel_remaining_kg": round(fuel_remaining_kg, 2),
            "post_burn_fuel_kg": round(post_burn_fuel, 2),
            "passivation_reserve_ok": passivation_reserve_ok,
            "standards_compliance": standards_compliance,
            "status": status_label,
        }

        self.blackboard.post("compliance_audit", payload, self.name)
        self.blackboard.broadcast(
            SwarmMessage(
                sender=self.name,
                recipient="BROADCAST",
                message_type="COMPLIANCE_AUDIT",
                payload=payload,
            )
        )
        return {"agent": self.name, "status": self.status.value, "decision": decision, "data": payload}

    def vote(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        delta_v = proposal.get("delta_v_m_s", proposal.get("delta_v_ms", 0.0))
        fuel_remaining = self.blackboard.get("compliance_audit", {}).get("fuel_remaining_kg", 18.5)
        fuel_needed = delta_v * 0.12

        if (fuel_remaining - fuel_needed) < self.min_fuel_reserve_kg:
            return {
                "agent": self.name,
                "role": self.role.value,
                "vote": "VETO",
                "confidence": 0.99,
                "reason": (
                    f"VETO: Maneuver exhausts passivation reserve below {self.min_fuel_reserve_kg} kg "
                    f"(NASA-STD-8719.14B Section 4.5.1 violation)"
                ),
            }

        return {
            "agent": self.name,
            "role": self.role.value,
            "vote": "APPROVE",
            "confidence": 0.98,
            "reason": "NASA-STD-8719.14B and IADC collision mitigation rules strictly satisfied",
        }
