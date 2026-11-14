"""
Unit tests for the AEGIS-LEO 9-Agent Swarm Intelligence Architecture.
"""

import pytest
from aegis_leo.agents import (
    SwarmBlackboard,
    SolarStormAgent,
    ThermosphereAgent,
    RiskAssessmentAgent,
    AstrodynamicsAgent,
    EdgeInferenceAgent,
    ConstellationConsensusAgent,
    ComplianceWatchdogAgent,
    ReentryDisposalAgent,
    MissionCommanderAgent,
    OrbitalSwarmOrchestrator,
)


def test_swarm_blackboard():
    bb = SwarmBlackboard()
    bb.post("kp_index", 7.5, "TEST_AGENT")
    assert bb.get("kp_index") == 7.5
    assert bb.get("missing_key", 42) == 42


def test_solar_agent():
    bb = SwarmBlackboard()
    agent = SolarStormAgent(bb)
    res = agent.execute_cycle({"kp": 8.0, "f10_7": 260.0})
    assert res["status"] == "ALERT"
    assert res["data"]["storm_severity"] == "G4_SEVERE"
    assert res["data"]["is_storm"] is True
    vote = agent.vote({})
    assert vote["vote"] in ["APPROVE", "URGENT_APPROVAL"]


def test_thermosphere_agent():
    bb = SwarmBlackboard()
    agent = ThermosphereAgent(bb)
    res = agent.execute_cycle({"altitude_km": 400.0, "kp": 7.0, "f10_7": 200.0})
    assert "density_kg_m3" in res["data"]
    assert res["data"]["density_kg_m3"] > 0
    vote = agent.vote({})
    assert vote["vote"] in ["APPROVE", "OPTIMAL"]


def test_risk_agent():
    bb = SwarmBlackboard()
    agent = RiskAssessmentAgent(bb)
    res = agent.execute_cycle({"miss_distance_km": 0.045, "altitude_km": 500.0, "kp": 6.5})
    assert res["data"]["risk_level"] in ["CRITICAL", "HIGH", "MODERATE", "LOW"]
    assert res["data"]["a_dri"] > 0
    vote = agent.vote({})
    assert vote["vote"] in ["APPROVE", "URGENT_EVADE"]


def test_astrodynamics_agent():
    bb = SwarmBlackboard()
    agent = AstrodynamicsAgent(bb)
    # Simulate a critical risk state on blackboard
    bb.post("conjunction_risk", {"risk_level": "CRITICAL"}, "RiskAssessmentAgent-03")
    res = agent.execute_cycle({"altitude_km": 450.0, "miss_distance_km": 0.05, "kp": 6.0})
    assert res["data"]["maneuver_needed"] is True
    assert res["data"]["delta_v_m_s"] > 0
    vote = agent.vote(res["data"])
    assert vote["vote"] == "APPROVE"


def test_edge_inference_agent():
    bb = SwarmBlackboard()
    agent = EdgeInferenceAgent(bb)
    res = agent.execute_cycle({"altitude_km": 450.0, "kp": 5.0, "f10_7": 150.0})
    assert res["data"]["tmr_hardware_resilience_pct"] > 80.0
    assert "pinn_predicted_density_mult" in res["data"]
    vote = agent.vote({})
    assert vote["vote"] == "APPROVE"


def test_constellation_agent():
    bb = SwarmBlackboard()
    agent = ConstellationConsensusAgent(bb)
    bb.post("maneuver_plan", {"delta_v_m_s": 3.5}, "AstrodynamicsAgent-04")
    res = agent.execute_cycle({})
    assert res["data"]["status"] in ["CONSENSUS_ALIGNED", "CONFLICT_RISK"]
    vote_safe = agent.vote({"delta_v_m_s": 5.0})
    assert vote_safe["vote"] == "APPROVE"
    vote_excess = agent.vote({"delta_v_m_s": 30.0})
    assert vote_excess["vote"] == "REJECT"


def test_compliance_agent():
    bb = SwarmBlackboard()
    agent = ComplianceWatchdogAgent(bb)
    res = agent.execute_cycle({"fuel_remaining_kg": 20.0})
    assert res["data"]["passivation_reserve_ok"] is True
    vote_pass = agent.vote({"delta_v_m_s": 2.0})
    assert vote_pass["vote"] == "APPROVE"

    bb.post("compliance_audit", {"fuel_remaining_kg": 2.5}, "ComplianceWatchdogAgent-07")
    vote_veto = agent.vote({"delta_v_m_s": 10.0})  # 10 * 0.12 = 1.2 kg -> 2.5 - 1.2 = 1.3 < 2.0 kg
    assert vote_veto["vote"] == "VETO"


def test_disposal_agent():
    bb = SwarmBlackboard()
    agent = ReentryDisposalAgent(bb)
    res = agent.execute_cycle({"altitude_km": 450.0, "fuel_remaining_kg": 18.0})
    assert res["data"]["estimated_lifetime_years"] > 0
    assert res["data"]["disposal_delta_v_ms"] > 0
    assert res["data"]["target_impact_zone"] == "Point Nemo (SPOUA)"


def test_commander_agent():
    bb = SwarmBlackboard()
    agent = MissionCommanderAgent(bb)
    proposal = {"delta_v_m_s": 4.2, "strategy": "PROACTIVE_PHASING_BURN"}
    votes = [
        {"agent": "SolarStormAgent-01", "vote": "APPROVE"},
        {"agent": "ThermosphereAgent-02", "vote": "APPROVE"},
        {"agent": "RiskAssessmentAgent-03", "vote": "APPROVE"},
        {"agent": "AstrodynamicsAgent-04", "vote": "APPROVE"},
        {"agent": "EdgeInferenceAgent-05", "vote": "APPROVE"},
        {"agent": "ConstellationConsensusAgent-06", "vote": "APPROVE"},
        {"agent": "ComplianceWatchdogAgent-07", "vote": "APPROVE"},
        {"agent": "ReentryDisposalAgent-08", "vote": "APPROVE"},
        {"agent": "MissionCommanderAgent-09", "vote": "APPROVE"},
    ]
    res = agent.evaluate_consensus(proposal, votes)
    assert res["consensus_reached"] is True
    assert res["directive"] == "EXECUTE_AUTONOMOUS_MANEUVER"
    assert res["authorization_signature"].startswith("AEGIS-SIG-")


def test_orbital_swarm_orchestrator_cycle():
    orchestrator = OrbitalSwarmOrchestrator()
    status = orchestrator.get_agent_status()
    assert len(status) == 9

    result = orchestrator.run_consensus_cycle()
    assert result["cycle_id"] == 1
    assert len(result["agent_results"]) == 9
    assert len(result["swarm_votes"]) == 9
    assert result["commander_directive"]["consensus_reached"] is True
    assert "AEGIS-SIG-" in result["commander_directive"]["authorization_signature"]
