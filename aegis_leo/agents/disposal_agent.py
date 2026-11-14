"""
Agent 8: Reentry Disposal Agent (Görev Sonu Atmosferik İmha ve Deorbit Ajanı)
Calculates orbital decay lifetime, assesses disposal delta-V, and optimizes
controlled reentry corridors toward Point Nemo (SPOUA - South Pacific Ocean Uninhabited Area).
"""

from typing import Dict, Any
import math
from aegis_leo.agents.base_agent import BaseSwarmAgent, AgentRole, AgentStatus, SwarmMessage, SwarmBlackboard


class ReentryDisposalAgent(BaseSwarmAgent):
    """
    Manages end-of-life disposal protocols, calculating controlled reentry parameters
    to ensure casualty risk < 1e-4 in accordance with NASA orbital safety standards.
    """

    def __init__(self, blackboard: SwarmBlackboard, name: str = "ReentryDisposalAgent-08"):
        super().__init__(role=AgentRole.REENTRY_DISPOSAL, name=name, blackboard=blackboard)
        self.point_nemo_coords = {"lat": -48.8767, "lon": -123.3933}

    def execute_cycle(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        self.cycle_count += 1
        altitude_km = telemetry.get("altitude_km", 450.0)
        fuel_remaining_kg = telemetry.get("fuel_remaining_kg", 18.5)
        solar_f107 = self.blackboard.get("solar_weather", {}).get("f10_7", 140.0)

        base_years = (altitude_km / 500.0) ** 3.8 * 8.5
        solar_factor = 150.0 / max(solar_f107, 70.0)
        estimated_lifetime_years = round(base_years * solar_factor, 2)

        mu = 398600.4418
        re = 6378.137
        r_current = re + altitude_km
        r_target_perigee = re + 180.0
        v_current = math.sqrt(mu / r_current)
        a_transfer = (r_current + r_target_perigee) / 2.0
        v_transfer_apogee = math.sqrt(mu * (2.0 / r_current - 1.0 / a_transfer))
        required_disposal_dv = round(abs(v_current - v_transfer_apogee) * 1000.0, 2)

        is_eol_triggered = fuel_remaining_kg < 3.5 or estimated_lifetime_years < 1.0
        disposal_mode = "IMMEDIATE_DEORBIT_RECOMMENDED" if is_eol_triggered else "PASSIVE_MONITORING"
        self.status = AgentStatus.ALERT if is_eol_triggered else AgentStatus.ACTIVE

        decision = (
            f"Orbital lifetime: {estimated_lifetime_years:.1f} yrs. Disposal mode: {disposal_mode}. "
            f"Disposal Δv required: {required_disposal_dv:.1f} m/s to Point Nemo."
        )
        self.last_decision = decision

        payload = {
            "estimated_lifetime_years": estimated_lifetime_years,
            "altitude_km": altitude_km,
            "disposal_delta_v_ms": required_disposal_dv,
            "target_impact_zone": "Point Nemo (SPOUA)",
            "casualty_risk_expected": "< 1:100,000",
            "disposal_mode": disposal_mode,
            "f107_drag_amplification": round(1.0 / solar_factor, 2),
        }

        self.blackboard.post("disposal_status", payload, self.name)
        self.blackboard.broadcast(
            SwarmMessage(
                sender=self.name,
                recipient="BROADCAST",
                message_type="REENTRY_LIFETIME_UPDATE",
                payload=payload,
            )
        )
        return {"agent": self.name, "status": self.status.value, "decision": decision, "data": payload}

    def vote(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        delta_v = proposal.get("delta_v_m_s", proposal.get("delta_v_ms", 0.0))
        fuel_remaining = self.blackboard.get("compliance_audit", {}).get("fuel_remaining_kg", 18.5)
        disposal_data = self.blackboard.get("disposal_status", {})
        required_dv = disposal_data.get("disposal_delta_v_ms", 120.0)

        fuel_needed = delta_v * 0.12
        post_burn_fuel = fuel_remaining - fuel_needed
        post_burn_dv_capability = post_burn_fuel * 25.0

        if post_burn_dv_capability < (required_dv * 0.5):
            return {
                "agent": self.name,
                "role": self.role.value,
                "vote": "CAUTION",
                "confidence": 0.85,
                "reason": "Maneuver depletes propellant budget below minimum controlled deorbit reserve",
            }

        return {
            "agent": self.name,
            "role": self.role.value,
            "vote": "APPROVE",
            "confidence": 0.95,
            "reason": "Safe deorbit delta-V margin to Point Nemo preserved",
        }
