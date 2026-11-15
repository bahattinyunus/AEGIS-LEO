"""Tests for End-to-End Integrated Autonomous Mission Pipeline."""

from aegis_leo.data.catalog import list_satellites
from aegis_leo.simulation.mission_pipeline import MissionPipelineConfig, run_end_to_end_mission


def test_end_to_end_mission_pipeline():
    sats = list_satellites()
    sat_id = sats[0]["id"]
    rep = run_end_to_end_mission(
        MissionPipelineConfig(
            satellite_id=sat_id,
            storm_preset="g4_severe",
            miss_distance_km=0.85,
            link_band="X_BAND",
            pass_duration_hours=6.0,
        )
    )
    d = rep.to_dict()
    assert 0.0 <= d["mission_readiness_score"] <= 100.0
    assert d["overall_status"] in {"MISSION_READY", "CAUTION", "CRITICAL_REVIEW"}
    assert d["ground_comms"]["total_passes"] >= 0
    assert d["conjunction_risk"]["foster_pc"] >= 0.0
    assert d["swarm_consensus"]["weighted_approval"] >= 0.0
    assert "nasa_std_8719_compliant" in d["compliance"]
    assert d["attitude_adcs"]["burn_attitude_ready"] is True
    assert d["thermal_control"]["all_nodes_within_limits"] is True
    assert d["pqc_security"]["authentic_command_verified"] is True
    assert d["pqc_security"]["replay_attack_blocked"] is True
