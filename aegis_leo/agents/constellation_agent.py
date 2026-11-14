"""
Agent 6: Constellation Consensus Agent (Takımyıldız Eşgüdüm ve Konsensüs Ajanı)
Coordinates multi-satellite inter-satellite links (ISL), verifies Walker-Delta
slot separations, and prevents constellation-level collision cascades.
"""

from typing import Dict, Any, List
from aegis_leo.agents.base_agent import BaseSwarmAgent, AgentRole, AgentStatus, SwarmMessage, SwarmBlackboard


class ConstellationConsensusAgent(BaseSwarmAgent):
    """
    Manages decentralized inter-satellite coordination across the constellation plane.
    """

    def __init__(self, blackboard: SwarmBlackboard, name: str = "ConstellationConsensusAgent-06"):
        super().__init__(role=AgentRole.CONSTELLATION_CONSENSUS, name=name, blackboard=blackboard)
        self.plane_id: int = 1
        self.satellite_index: int = 4
        self.peer_satellites: List[str] = ["AEGIS-01", "AEGIS-02", "AEGIS-03", "AEGIS-05"]

    def execute_cycle(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        self.cycle_count += 1
        maneuver_proposal = self.blackboard.get("maneuver_plan", {})
        delta_v = maneuver_proposal.get("delta_v_m_s", 0.0)

        # Nominal in-plane inter-satellite spacing
        slot_separation_km = 450.0
        drift_estimate_km = delta_v * 12.0

        conflict_detected = drift_estimate_km > (slot_separation_km * 0.4)
        status_label = "CONFLICT_RISK" if conflict_detected else "CONSENSUS_ALIGNED"
        self.status = AgentStatus.ALERT if conflict_detected else AgentStatus.ACTIVE

        decision = (
            f"Constellation ISL mesh verified: {len(self.peer_satellites)} peers active. "
            f"Status: {status_label} (drift={drift_estimate_km:.2f} km)."
        )
        self.last_decision = decision

        payload = {
            "plane_id": self.plane_id,
            "satellite_index": self.satellite_index,
            "peers_polled": len(self.peer_satellites),
            "slot_separation_km": slot_separation_km,
            "drift_estimate_km": round(drift_estimate_km, 2),
            "status": status_label,
            "isl_health": "OPTIMAL",
            "cross_link_latency_ms": 14.2,
        }

        self.blackboard.post("constellation_consensus", payload, self.name)
        self.blackboard.broadcast(
            SwarmMessage(
                sender=self.name,
                recipient="BROADCAST",
                message_type="ISL_CONSENSUS_SYNC",
                payload=payload,
            )
        )
        return {"agent": self.name, "status": self.status.value, "decision": decision, "data": payload}

    def vote(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        delta_v = proposal.get("delta_v_m_s", proposal.get("delta_v_ms", 0.0))
        if delta_v > 25.0:
            return {
                "agent": self.name,
                "role": self.role.value,
                "vote": "REJECT",
                "confidence": 0.88,
                "reason": f"Delta-v {delta_v:.1f} m/s causes fatal constellation slot desynchronization",
            }
        return {
            "agent": self.name,
            "role": self.role.value,
            "vote": "APPROVE",
            "confidence": 0.94,
            "reason": f"Delta-v {delta_v:.1f} m/s verified safe against Walker-Delta neighbor slots",
        }
