import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "AEGIS-LEO" in data["system"]


def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ONLINE"


def test_space_weather_endpoints():
    res_curr = client.get("/api/space-weather/current")
    assert res_curr.status_code == 200
    assert "kp" in res_curr.json()

    res_storm = client.post("/api/space-weather/simulate-storm", json={"event_key": "G4_SEVERE"})
    assert res_storm.status_code == 200
    assert res_storm.json()["kp"] == 8.0


def test_physics_density_endpoint():
    response = client.get("/api/physics/density?altitude_km=550&kp=8.0&f10_7=240")
    assert response.status_code == 200
    data = response.json()
    assert data["storm_expansion_multiplier"] > 1.0


def test_conjunction_risk_endpoint():
    payload = {
        "satellite_id": "STARLINK_LEO",
        "debris_id": "COSMOS_2251_DEB",
        "miss_distance_km": 0.45,
        "include_storm": True,
        "storm_kp": 8.0,
        "storm_f10_7": 265.0,
    }
    response = client.post("/api/risk/evaluate-conjunction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "a_dri_assessment" in data
    assert 0.0 <= data["a_dri_assessment"]["a_dri"] <= 1.0


def test_optimize_maneuver_endpoint():
    payload = {
        "satellite_id": "STARLINK_LEO",
        "miss_distance_km": 0.45,
        "storm_kp": 8.0,
        "storm_f10_7": 265.0,
        "along_track_disp_km": 14.2,
    }
    response = client.post("/api/edge-ai/optimize-maneuver", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "AUTONOMOUS_BURN_PLANNED"
    assert "proactive_burn" in data


def test_compliance_endpoint():
    payload = {
        "satellite_id": "STARLINK_LEO",
        "conjunction_pc": 5e-5,
        "has_passivation_valves": True,
    }
    response = client.post("/api/compliance/audit", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "overall_status" in data


def test_dashboard_endpoint():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "AEGIS-LEO" in response.text


def test_swarm_endpoints():
    # Test GET status
    resp_status = client.get("/api/swarm/status")
    assert resp_status.status_code == 200
    data_status = resp_status.json()
    assert data_status["agent_count"] == 9
    assert len(data_status["swarm_roster"]) == 9

    # Test POST consensus cycle
    payload = {
        "satellite_id": "AEGIS-ALPHA",
        "altitude_km": 450.0,
        "miss_distance_km": 0.075,
        "kp": 8.0,
        "f10_7": 260.0,
        "fuel_remaining_kg": 18.0,
    }
    resp_cycle = client.post("/api/swarm/consensus-cycle", json=payload)
    assert resp_cycle.status_code == 200
    data_cycle = resp_cycle.json()
    assert data_cycle["commander_directive"]["consensus_reached"] is True
    assert "AEGIS-SIG-" in data_cycle["commander_directive"]["authorization_signature"]
    assert len(data_cycle["swarm_votes"]) == 9


def test_lifetime_api_endpoint():
    payload = {
        "altitude_km": 350.0,
        "mass_kg": 200.0,
        "drag_area_m2": 2.0,
        "cd": 2.2,
        "max_duration_years": 10.0,
    }
    resp = client.post("/api/lifetime/decay-prediction", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["fcc_5year_compliant"] is True
    assert data["estimated_lifetime_years"] < 5.0


def test_electrodynamic_tether_api_endpoint():
    resp = client.get("/api/adr/electrodynamic-tether?satellite_mass_kg=250&initial_altitude_km=750")
    assert resp.status_code == 200
    data = resp.json()
    assert "methods" in data
    assert "ELECTRODYNAMIC_TETHER" in data["methods"]


def test_laser_ablation_api_endpoint():
    payload = {
        "target_id": "DEBRIS-TEST-01",
        "debris_mass_kg": 0.9,
        "debris_diameter_cm": 14.0,
        "perigee_km": 420.0,
        "apogee_km": 720.0,
        "laser_power_kw": 12.0,
        "engagement_duration_s": 40.0,
    }
    resp = client.post("/api/adr/laser-ablation", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "PASS_COMPLETED"
    assert data["delta_v_m_s"] > 0


def test_swarm_bft_consensus_api_endpoint():
    payload = {
        "sat_a_id": "SAT-WALKER-A1",
        "sat_b_id": "SAT-WALKER-A2",
        "initial_miss_m": 45.0,
        "tca_s": 1800.0,
        "nominal_dv_req_m_s": 0.45,
    }
    resp = client.post("/api/swarm/bft-consensus", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["agreement_state"] == "MANEUVER_COMMITTED"
    assert data["quorum_percentage"] >= 70.0


def test_pass_schedule_api_endpoint():
    resp = client.post("/api/comms/pass-schedule", json={
        "altitude_km": 550.0,
        "inclination_deg": 53.0,
        "duration_hours": 6.0,
        "band": "X_BAND",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_passes"] > 0
    assert data["total_data_downlinked_mb"] > 0


def test_eclipse_power_api_endpoint():
    resp = client.post("/api/physics/eclipse-power", json={
        "altitude_km": 550.0,
        "inclination_deg": 97.6,
        "day_of_year": 172,
        "bus_load_w": 160.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["power_margin_positive"] is True
    assert len(data["timeline"]) == 120


def test_end_to_end_mission_api_endpoint():
    sats = client.get("/api/catalog/satellites").json()
    sat_id = sats[0]["id"]
    resp = client.post("/api/mission/end-to-end", json={
        "satellite_id": sat_id,
        "storm_preset": "g4_severe",
        "miss_distance_km": 0.85,
        "link_band": "X_BAND",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["mission_readiness_score"] <= 100.0
    assert "swarm_consensus" in data
    assert "compliance" in data


def test_wave2_rest_endpoints():
    r_bplane = client.post("/api/risk/foster-bplane", json={"altitude_km": 550.0, "miss_distance_km": 0.35})
    assert r_bplane.status_code == 200
    assert r_bplane.json()["pc_foster"] >= 0.0

    r_radar = client.post("/api/sensors/radar-detect?range_km=15.0&rcs_m2=0.5")
    assert r_radar.status_code == 200
    assert r_radar.json()["detection_probability_pd"] >= 0.0

    r_mpc = client.post("/api/ai/mpc-avoidance?ellipsoid_radius_km=0.5")
    assert r_mpc.status_code == 200
    assert r_mpc.json()["total_delta_v_m_s"] >= 0.0

    r_iod = client.post("/api/navigation/iod-gauss?altitude_km=550.0&inclination_deg=53.0")
    assert r_iod.status_code == 200
    assert "orbital_elements" in r_iod.json()

    r_ekf = client.post("/api/navigation/ekf-od?altitude_km=550.0&kp=5.0")
    assert r_ekf.status_code == 200

    r_vleo = client.post("/api/ai/vleo-drag-makeup?altitude_km=220.0")
    assert r_vleo.status_code == 200
    assert "l1_altitude_error_rms_m" in r_vleo.json()

    r_fed = client.post("/api/ai/federated-swarm?num_satellites=6&rounds=2")
    assert r_fed.status_code == 200
    assert "fedprox_final_loss" in r_fed.json()

    sats = client.get("/api/catalog/satellites").json()
    r_odar = client.post(f"/api/compliance/odar-report?satellite_id={sats[0]['id']}")
    assert r_odar.status_code == 200
    assert "ORBITAL DEBRIS ASSESSMENT REPORT" in r_odar.json()["odar_markdown"].upper()


def test_wave3_v2_rest_endpoints():
    r_adcs = client.post("/api/astrodynamics/adcs-slew", json={
        "target_roll_deg": 0.0,
        "target_pitch_deg": 35.0,
        "target_yaw_deg": -20.0,
        "altitude_km": 550.0,
        "kp_storm": 6.0,
        "duration_s": 90.0,
    })
    assert r_adcs.status_code == 200
    assert r_adcs.json()["burn_attitude_ready"] is True

    r_thm = client.post("/api/physics/thermal-analysis", json={
        "altitude_km": 550.0,
        "beta_angle_deg": 22.0,
        "kp_storm": 5.0,
        "avionics_load_w": 85.0,
        "burn_active": True,
    })
    assert r_thm.status_code == 200
    assert r_thm.json()["all_nodes_within_limits"] is True

    r_pqc = client.post("/api/security/pqc-verify", json={
        "spacecraft_id": "STARLINK_LEO",
        "command_op": "EXECUTE_PROACTIVE_BURN",
        "delta_v_m_s": 0.325,
    })
    assert r_pqc.status_code == 200
    assert r_pqc.json()["authentic_command_verified"] is True
    assert r_pqc.json()["replay_attack_blocked"] is True

    r_tour = client.post("/api/adr/lambert-tour", json={
        "servicer_dry_mass_kg": 450.0,
        "initial_fuel_kg": 180.0,
        "isp_s": 320.0,
        "start_altitude_km": 750.0,
        "max_targets": 4,
    })
    assert r_tour.status_code == 200
    assert r_tour.json()["targets_visited"] == 4




