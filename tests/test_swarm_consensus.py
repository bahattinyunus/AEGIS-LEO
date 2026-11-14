"""
Tests for Swarm Quorum & Byzantine-Tolerant Autonomous Avoidance Consensus.
"""

import pytest
from aegis_leo.agents.swarm_consensus import (
    SwarmPeerNode,
    ConjunctionProposal,
    SwarmConsensusCoordinator,
    ConsensusState,
)


def test_peer_node_cost_calculation():
    p1 = SwarmPeerNode("SAT-1", 1, 1, remaining_delta_v_m_s=50.0, battery_state_of_charge_pct=95.0)
    p2 = SwarmPeerNode("SAT-2", 1, 2, remaining_delta_v_m_s=10.0, battery_state_of_charge_pct=40.0)

    # p1 has more fuel & better battery -> should have lower maneuver penalty
    cost1 = p1.compute_maneuver_cost_weight(0.5)
    cost2 = p2.compute_maneuver_cost_weight(0.5)
    assert cost1 < cost2


def test_proposal_digest_and_tamper_detection():
    prop = ConjunctionProposal(
        proposal_id="PROP-01",
        target_primary_id="SAT-A",
        target_secondary_id="SAT-B",
        miss_distance_m=65.0,
        time_to_closest_approach_s=1800.0,
        candidate_maneuver_satellite="SAT-A",
        allocated_delta_v_vector_m_s=[0.0, 0.4, 0.1],
        projected_post_miss_m=2000.0,
    )
    digest = prop.calculate_digest()
    assert len(digest) == 16

    # Tampering with vector changes digest
    prop.allocated_delta_v_vector_m_s[1] = 0.99
    tampered_digest = prop.calculate_digest()
    assert digest != tampered_digest


def test_swarm_consensus_coordinator_arbitration():
    coord = SwarmConsensusCoordinator()
    assert coord.total_nodes >= 7
    assert coord.max_byzantine_faulty_nodes >= 2
    assert coord.quorum_threshold >= 5

    res = coord.arbitrate_conjunction(
        sat_a_id="SAT-WALKER-A1",
        sat_b_id="SAT-WALKER-A2",
        initial_miss_m=55.0,
        tca_s=3600.0,
        nominal_dv_req_m_s=0.35,
    )

    assert res.agreement_state == ConsensusState.MANEUVER_COMMITTED
    assert res.quorum_percentage >= 70.0
    assert res.elected_satellite_to_maneuver in ["SAT-WALKER-A1", "SAT-WALKER-A2"]
    assert res.holding_satellite in ["SAT-WALKER-A1", "SAT-WALKER-A2"]
    assert res.elected_satellite_to_maneuver != res.holding_satellite
    assert res.secondary_risk_cleared is True
    assert len(res.audit_trail) >= 3

    d = res.to_dict()
    assert d["agreement_state"] == "MANEUVER_COMMITTED"
    assert "quorum_percentage" in d
