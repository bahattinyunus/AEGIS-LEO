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
