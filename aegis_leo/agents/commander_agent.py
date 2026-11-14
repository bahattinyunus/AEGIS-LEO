"""
Agent 9: Mission Commander Agent (Ana Görev Komuta ve Karar Verme Ajanı)
Aggregates recommendations from all 8 specialist agents, executes Byzantine
consensus voting, enforces failsafe overrides, and authorizes orbital burns.
"""

from typing import Dict, Any, List
import hashlib
from datetime import datetime, timezone
from aegis_leo.agents.base_agent import BaseSwarmAgent, AgentRole, AgentStatus, SwarmMessage, SwarmBlackboard


class MissionCommanderAgent(BaseSwarmAgent):
    """
    Chief decision-making agent that conducts voting cycles and generates
    signed flight-software execution commands.
    """

    def __init__(self, blackboard: SwarmBlackboard, name: str = "MissionCommanderAgent-09"):
        super().__init__(role=AgentRole.MISSION_COMMANDER, name=name, blackboard=blackboard)
        self.consensus_threshold: float = 0.70
        self.agent_weights = {
            "SolarStormAgent-01": 1.0,
            "ThermosphereAgent-02": 1.0,
            "RiskAssessmentAgent-03": 1.5,
            "AstrodynamicsAgent-04": 1.5,
            "EdgeInferenceAgent-05": 1.2,
            "ConstellationConsensusAgent-06": 1.0,
            "ComplianceWatchdogAgent-07": 2.0,
            "ReentryDisposalAgent-08": 1.0,
            "MissionCommanderAgent-09": 1.0,
        }

    def execute_cycle(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        self.cycle_count += 1
        decision = "Commander nominal. Ready to evaluate swarm consensus."
        self.last_decision = decision
        return {
            "agent": self.name,
            "status": self.status.value,
            "decision": decision,
            "data": {"status": "READY_FOR_CONSENSUS"},
        }

    def evaluate_consensus(self, proposal: Dict[str, Any], votes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate consensus from all agents' votes and issue final mission directive.
        """
        total_weight = 0.0
        approve_weight = 0.0
        veto_triggered = False
        veto_reason = None
        rejections: List[str] = []

        for v in votes:
            agent_name = v.get("agent", "")
            vote_val = v.get("vote", "REJECT")
            weight = self.agent_weights.get(agent_name, 1.0)
            total_weight += weight

            if vote_val == "VETO":
                veto_triggered = True
                veto_reason = v.get("reason", "Veto invoked without detailed reason")
            elif vote_val in ["APPROVE", "OPTIMAL", "URGENT_APPROVAL"]:
                approve_weight += weight
            elif vote_val in ["CAUTION", "URGENT_EVADE"]:
                approve_weight += weight * 0.7
            else:
                rejections.append(f"{agent_name}: {v.get('reason', 'Rejected')}")

        approval_ratio = (approve_weight / total_weight) if total_weight > 0 else 0.0
        is_consensus_reached = (approval_ratio >= self.consensus_threshold) and not veto_triggered

        if veto_triggered:
            command_directive = "ABORT_VETOED"
            summary_status = f"VETO BY WATCHDOG: {veto_reason}"
        elif is_consensus_reached:
            command_directive = "EXECUTE_AUTONOMOUS_MANEUVER"
            summary_status = f"CONSENSUS REACHED ({approval_ratio*100:.1f}% approval)"
        else:
            command_directive = "HOLD_OBSERVE"
            summary_status = f"INSUFFICIENT CONSENSUS ({approval_ratio*100:.1f}% < {self.consensus_threshold*100:.0f}%)"

        timestamp_str = datetime.now(timezone.utc).isoformat()
        signature_material = f"{command_directive}_{proposal}_{timestamp_str}"
        cmd_signature = hashlib.sha256(signature_material.encode("utf-8")).hexdigest()[:16].upper()

        decision_payload = {
            "commander": self.name,
            "directive": command_directive,
            "status": summary_status,
            "consensus_reached": is_consensus_reached,
            "approval_ratio": round(approval_ratio, 3),
            "approval_percentage": round(approval_ratio * 100, 1),
            "total_votes": len(votes),
            "veto_triggered": veto_triggered,
            "veto_reason": veto_reason,
            "rejections": rejections,
            "authorization_signature": f"AEGIS-SIG-{cmd_signature}",
            "execution_timestamp": timestamp_str,
            "authorized_maneuver": proposal if is_consensus_reached else None,
        }

        self.blackboard.post("commander_directive", decision_payload, self.name)
        self.blackboard.broadcast(
            SwarmMessage(
                sender=self.name,
                recipient="BROADCAST",
                message_type="MISSION_COMMAND_DIRECTIVE",
                payload={"directive": command_directive, "signature": decision_payload["authorization_signature"]},
                priority=10,
            )
        )
        self.last_decision = command_directive
        return decision_payload

    def vote(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "agent": self.name,
            "role": self.role.value,
            "vote": "APPROVE",
            "confidence": 0.99,
            "reason": "Commander nominal operational readiness verified",
        }
