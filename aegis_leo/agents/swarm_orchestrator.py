"""
AEGIS-LEO Orbital Swarm Orchestrator (Otonom Yörünge Sürü Zekası Orkestratörü)
Coordinates the 9 specialized onboard flight agents, manages asynchronous
telemetry exchange through the shared blackboard, and executes consensus voting.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from aegis_leo.agents.base_agent import SwarmBlackboard, SwarmMessage
from aegis_leo.agents.solar_agent import SolarStormAgent
from aegis_leo.agents.thermosphere_agent import ThermosphereAgent
from aegis_leo.agents.risk_agent import RiskAssessmentAgent
from aegis_leo.agents.astrodynamics_agent import AstrodynamicsAgent
from aegis_leo.agents.edge_inference_agent import EdgeInferenceAgent
from aegis_leo.agents.constellation_agent import ConstellationConsensusAgent
from aegis_leo.agents.compliance_agent import ComplianceWatchdogAgent
from aegis_leo.agents.disposal_agent import ReentryDisposalAgent
from aegis_leo.agents.commander_agent import MissionCommanderAgent


class OrbitalSwarmOrchestrator:
    """
    Master swarm controller orchestrating 9 onboard autonomous flight agents.
    """

    def __init__(self):
        self.blackboard = SwarmBlackboard()
        self.agents = {
            "A1_SOLAR": SolarStormAgent(self.blackboard),
            "A2_THERMOSPHERE": ThermosphereAgent(self.blackboard),
            "A3_RISK": RiskAssessmentAgent(self.blackboard),
            "A4_ASTRODYNAMICS": AstrodynamicsAgent(self.blackboard),
            "A5_EDGE_INFERENCE": EdgeInferenceAgent(self.blackboard),
            "A6_CONSTELLATION": ConstellationConsensusAgent(self.blackboard),
            "A7_COMPLIANCE": ComplianceWatchdogAgent(self.blackboard),
            "A8_DISPOSAL": ReentryDisposalAgent(self.blackboard),
            "A9_COMMANDER": MissionCommanderAgent(self.blackboard),
        }
        self.cycle_count = 0
        self.history: List[Dict[str, Any]] = []

    def get_agent_status(self) -> List[Dict[str, Any]]:
        """
        Returns status summary for all 9 agents.
        """
        status_list = []
        for code, agent in self.agents.items():
            status_list.append({
                "code": code,
                "name": agent.name,
                "role": agent.role.value,
                "status": agent.status.value,
                "cycles": agent.cycle_count,
                "last_decision": agent.last_decision or "STANDBY",
            })
        return status_list

    def run_consensus_cycle(self, telemetry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes a complete 9-agent autonomous operational and consensus cycle.
        """
        self.cycle_count += 1
        if telemetry is None:
            telemetry = {
                "satellite_id": "AEGIS-ALPHA",
                "altitude_km": 448.2,
                "inclination_deg": 53.0,
                "miss_distance_km": 0.082,
                "relative_speed_km_s": 11.2,
                "tca_hours": 3.4,
                "fuel_remaining_kg": 18.2,
                "kp": 7.2,
                "f10_7": 245.0,
            }
        else:
            # normalize keys if callers use kp_index or f107_flux
            if "kp_index" in telemetry and "kp" not in telemetry:
                telemetry["kp"] = telemetry["kp_index"]
            if "f107_flux" in telemetry and "f10_7" not in telemetry:
                telemetry["f10_7"] = telemetry["f107_flux"]
            if "miss_distance_m" in telemetry and "miss_distance_km" not in telemetry:
                telemetry["miss_distance_km"] = telemetry["miss_distance_m"] / 1000.0

        self.blackboard.post("raw_telemetry", telemetry, "TELEMETRY_BUS")

        # Phase 1: Environment & Space Weather Sensing (A1, A2)
        solar_res = self.agents["A1_SOLAR"].execute_cycle(telemetry)
        thermo_res = self.agents["A2_THERMOSPHERE"].execute_cycle(telemetry)

        # Phase 2: Collision Risk Evaluation (A3)
        risk_res = self.agents["A3_RISK"].execute_cycle(telemetry)

        # Phase 3: Astrodynamics Maneuver Synthesis & Edge AI Optimization (A4, A5)
        astro_res = self.agents["A4_ASTRODYNAMICS"].execute_cycle(telemetry)
        edge_res = self.agents["A5_EDGE_INFERENCE"].execute_cycle(telemetry)

        # Maneuver proposal under consideration
        maneuver_proposal = astro_res.get("data", {})

        # Phase 4: Constellation Coordination, Compliance & Disposal (A6, A7, A8)
        const_res = self.agents["A6_CONSTELLATION"].execute_cycle(telemetry)
        comp_res = self.agents["A7_COMPLIANCE"].execute_cycle(telemetry)
        disp_res = self.agents["A8_DISPOSAL"].execute_cycle(telemetry)

        # Phase 5: Swarm-wide Voting Cycle
        votes = []
        for code, agent in self.agents.items():
            vote = agent.vote(maneuver_proposal)
            votes.append(vote)

        # Phase 6: Commander Consensus Evaluation & Directive Generation (A9)
        commander_directive = self.agents["A9_COMMANDER"].evaluate_consensus(
            proposal=maneuver_proposal,
            votes=votes,
        )

        cycle_result = {
            "cycle_id": self.cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "telemetry": telemetry,
            "agent_results": {
                "A1_solar": solar_res,
                "A2_thermosphere": thermo_res,
                "A3_risk": risk_res,
                "A4_astrodynamics": astro_res,
                "A5_edge_ai": edge_res,
                "A6_constellation": const_res,
                "A7_compliance": comp_res,
                "A8_disposal": disp_res,
                "A9_commander": self.agents["A9_COMMANDER"].execute_cycle(telemetry),
            },
            "maneuver_proposal": maneuver_proposal,
            "swarm_votes": votes,
            "commander_directive": commander_directive,
            "recent_messages": self.blackboard.get_recent_messages(10),
        }

        self.history.append(cycle_result)
        if len(self.history) > 50:
            self.history.pop(0)

        return cycle_result
