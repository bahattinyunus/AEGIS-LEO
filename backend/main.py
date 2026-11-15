"""
AEGIS-LEO Backend Server
FastAPI application providing Space Situational Awareness, Space Weather Drag Physics,
A-DRI Risk Assessment, and Edge-AI Autonomous Avoidance APIs.
"""

from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import torch
import numpy as np

from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    QUIET_KP,
    QUIET_F10_7,
    STORM_G4_KP,
    STORM_G5_KP,
)
from aegis_leo.physics.atmosphere import compute_atmospheric_density, compute_ballistic_coefficient
from aegis_leo.physics.propagator import OrbitalPropagator
from aegis_leo.physics.coordinates import kepler_to_cartesian, cartesian_to_kepler
from aegis_leo.risk.adri import ADRICalculator
from aegis_leo.risk.collision import compute_collision_probability
from aegis_leo.ai.pinn import get_pretrained_edge_pinn
from aegis_leo.ai.edge_optimizer import EdgeManeuverOptimizer
from aegis_leo.ai.quantizer import benchmark_edge_inference
from aegis_leo.data.space_weather import SpaceWeatherService, SpaceWeatherEvent
from aegis_leo.data.catalog import list_satellites, list_debris, get_satellite_by_id
from aegis_leo.data.cdm import generate_cdm_record
from aegis_leo.compliance.nasa_std_8719 import NASAComplianceAuditor
from aegis_leo.physics.atmosphere_models import compare_atmospheric_models
from aegis_leo.physics.ground_track import calculate_ground_track, analyze_communication_blackouts, GROUND_STATIONS
from aegis_leo.screening.constellation_screen import (
    generate_walker_delta_constellation,
    generate_synthetic_debris_cloud,
    ConstellationConjunctionScreener,
)
from aegis_leo.ai.pareto_optimizer import ParetoManeuverOptimizer
from aegis_leo.data.live_feeds import LiveSpaceWeatherFeed
from aegis_leo.data.celestrak_live import CelesTrakClient
from aegis_leo.ai.radiation_fault_simulator import RadiationFaultSimulator
from aegis_leo.physics.deorbit_simulation import compute_deorbit_burn, simulate_reentry_corridor, POINT_NEMO
from aegis_leo.risk.covariance_visualizer import generate_covariance_ellipsoid_mesh
from aegis_leo.agents.swarm_orchestrator import OrbitalSwarmOrchestrator
from aegis_leo.physics.orbit_lifetime import simulate_orbit_decay, evaluate_deorbit_compliance_matrix
from aegis_leo.adr.electrodynamic_tether import simulate_electrodynamic_tether_decay, compare_all_deorbit_methods
from aegis_leo.adr.laser_ablation import simulate_laser_debris_engagement, LaserStationSpecs, DebrisTargetSpecs, LaserPlatformType
from aegis_leo.agents.swarm_consensus import SwarmConsensusCoordinator
from aegis_leo.physics.pass_scheduler import LinkBand, schedule_ground_passes
from aegis_leo.physics.eclipse_power import simulate_eclipse_and_power
from aegis_leo.simulation.mission_pipeline import MissionPipelineConfig, run_end_to_end_mission
from aegis_leo.risk.foster_bplane import compute_bplane_conjunction_risk
from aegis_leo.compliance.odar_generator import generate_odar_report
from aegis_leo.navigation.sensor_fusion import AODSensorFusionEngine
from aegis_leo.navigation.iod_gauss import generate_synthetic_optical_observations, initial_orbit_determination
from aegis_leo.sensors.laser_radar import SpaceborneRadar
from aegis_leo.ai.mpc_avoidance import SCPAvoidancePlanner, AvoidanceEllipsoid
from aegis_leo.ai.drag_makeup import compare_admu_controllers
from aegis_leo.ai.federated_swarm import compare_fedavg_vs_fedprox
from aegis_leo.astrodynamics.attitude_adcs import simulate_attitude_slew
from aegis_leo.astrodynamics.lambert_tour import plan_multi_debris_adr_tour
from aegis_leo.physics.thermal_analyzer import simulate_spacecraft_thermal_balance
from aegis_leo.security.pqc_telecommand import run_pqc_telecommand_audit


app = FastAPI(
    title="AEGIS-LEO Mission Control API",
    description="Autonomous Edge-AI & Physics-Informed Orbital Safety Shield API (NASA Space Apps Challenge 2026)",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services & instances
weather_service = SpaceWeatherService()
adri_calc = ADRICalculator()
edge_pinn = get_pretrained_edge_pinn()
compliance_auditor = NASAComplianceAuditor()
live_weather_feed = LiveSpaceWeatherFeed()
constellation_screener = ConstellationConjunctionScreener()


# ------------------ Request / Response Schemas ------------------ #

class StormSimulationRequest(BaseModel):
    event_key: Optional[str] = Field("G4_SEVERE", description="Storm preset (G1_MINOR, G3_STRONG, G4_SEVERE, G5_EXTREME, QUIET)")
    custom_kp: Optional[float] = Field(None, ge=0.0, le=9.0, description="Optional custom Kp index (0 to 9)")
    custom_f10_7: Optional[float] = Field(None, ge=50.0, le=450.0, description="Optional custom F10.7 flux")


class ConjunctionEvalRequest(BaseModel):
    satellite_id: str = "STARLINK_LEO"
    debris_id: str = "COSMOS_2251_DEB"
    miss_distance_km: float = 0.45
    include_storm: bool = True
    storm_kp: float = 8.0
    storm_f10_7: float = 265.0


class ManeuverPlanRequest(BaseModel):
    satellite_id: str = "STARLINK_LEO"
    miss_distance_km: float = 0.45
    storm_kp: float = 8.0
    storm_f10_7: float = 265.0
    along_track_disp_km: float = 14.2


class ComplianceAuditRequest(BaseModel):
    satellite_id: str = "STARLINK_LEO"
    conjunction_pc: float = 2.4e-4
    has_passivation_valves: bool = True


# ------------------ API Endpoints ------------------ #

@app.get("/")
def root():
    return {
        "system": "AEGIS-LEO: Autonomous Edge-AI & Physics-Informed Orbital Safety Shield",
        "team": "Team AYAZ (NASA Space Apps Challenge 2026)",
        "lead": "Bahattin Yunus Çetin (@bahattinyunus)",
        "version": "1.0.0",
        "docs_url": "/docs",
        "hud_dashboard": "/dashboard",
    }


@app.get("/api/health")
def health_check():
    return {"status": "ONLINE", "subsystems": {"physics": "OK", "ai_pinn": "OK", "weather_feed": "OK"}}


# --- Space Weather Endpoints ---

@app.get("/api/space-weather/current")
def get_current_space_weather():
    return weather_service.get_current_conditions()


@app.get("/api/space-weather/presets")
def get_space_weather_presets():
    return weather_service.list_preset_events()


@app.post("/api/space-weather/simulate-storm")
def simulate_storm(req: StormSimulationRequest):
    if req.custom_kp is not None and req.custom_f10_7 is not None:
        return weather_service.set_custom_parameters(req.custom_kp, req.custom_f10_7)
    if req.event_key:
        return weather_service.set_storm_event(req.event_key)
    return weather_service.get_current_conditions()


# --- Catalog Endpoints ---

@app.get("/api/catalog/satellites")
def get_satellites():
    return list_satellites()


@app.get("/api/catalog/debris")
def get_debris_objects():
    return list_debris()


# --- Physics & Propagation Endpoints ---

@app.get("/api/physics/density")
def get_thermospheric_density(
    altitude_km: float = Query(550.0, ge=100.0, le=1200.0),
    kp: float = Query(QUIET_KP, ge=0.0, le=9.0),
    f10_7: float = Query(QUIET_F10_7, ge=50.0, le=450.0),
):
    density = compute_atmospheric_density(altitude_km, kp, f10_7)
    quiet_density = compute_atmospheric_density(altitude_km, QUIET_KP, QUIET_F10_7)
    multiplier = density / max(quiet_density, 1e-25)

    return {
        "altitude_km": altitude_km,
        "kp_index": kp,
        "f10_7_flux": f10_7,
        "density_kg_m3": density,
        "quiet_baseline_density_kg_m3": quiet_density,
        "storm_expansion_multiplier": round(multiplier, 2),
    }


@app.get("/api/physics/simulate-decay")
def simulate_orbital_decay(
    altitude_km: float = 550.0,
    duration_hours: float = 36.0,
    storm_kp: float = 8.0,
    storm_f10_7: float = 265.0,
    area_m2: float = 2.2,
    mass_kg: float = 260.0,
):
    # Initialize circular orbit at given altitude
    r0, v0 = kepler_to_cartesian(
        a_km=EARTH_RADIUS_KM + altitude_km,
        ecc=0.0001,
        inc_deg=53.0,
        raan_deg=0.0,
        argp_deg=0.0,
        nu_deg=0.0,
    )

    propagator = OrbitalPropagator(cd=2.2, area_m2=area_m2, mass_kg=mass_kg)
    divergence = propagator.simulate_storm_divergence(
        r0=r0,
        v0=v0,
        duration_hours=duration_hours,
        step_seconds=120.0,
        storm_kp=storm_kp,
        storm_f10_7=storm_f10_7,
    )

    return {
        "duration_hours": duration_hours,
        "satellite": {"mass_kg": mass_kg, "area_m2": area_m2, "initial_altitude_km": altitude_km},
        "storm_conditions": {"kp": storm_kp, "f10_7": storm_f10_7},
        "quiet_decay_meters": round(divergence["quiet_altitude_decay_m"], 2),
        "storm_decay_meters": round(divergence["storm_altitude_decay_m"], 2),
        "net_unmodeled_decay_meters": round(divergence["net_decay_meters"], 2),
        "along_track_displacement_km": round(divergence["along_track_displacement_km"], 2),
        "total_miss_divergence_km": round(divergence["total_divergence_km"], 2),
    }


# --- A-DRI Risk Scoring Endpoints ---

@app.post("/api/risk/evaluate-conjunction")
def evaluate_conjunction(req: ConjunctionEvalRequest):
    sat = get_satellite_by_id(req.satellite_id)
    if not sat:
        raise HTTPException(status_code=404, detail="Satellite not found")

    kp = req.storm_kp if req.include_storm else QUIET_KP
    f10_7 = req.storm_f10_7 if req.include_storm else QUIET_F10_7
    along_track = 14.2 if req.include_storm else 0.2

    risk_result = adri_calc.evaluate_risk(
        miss_distance_km=req.miss_distance_km,
        altitude_km=sat["altitude_km"],
        inclination_deg=sat["inclination_deg"],
        area_m2=sat["area_m2"],
        mass_kg=sat["mass_kg"],
        kp=kp,
        f10_7=f10_7,
        along_track_disp_km=along_track,
        remaining_propellant_kg=sat["remaining_propellant_kg"],
    )

    cdm = generate_cdm_record(
        primary_sat_id=sat["name"],
        secondary_obj_id=req.debris_id,
        miss_distance_km=req.miss_distance_km,
        relative_speed_km_s=11.4,
        collision_prob=risk_result["components"]["c_prob_raw"],
        altitude_km=sat["altitude_km"],
    )

    return {
        "satellite": sat["name"],
        "debris": req.debris_id,
        "a_dri_assessment": risk_result,
        "synthesized_cdm": cdm,
    }


# --- Edge-AI & Autonomous Maneuver Optimizer Endpoints ---

@app.post("/api/edge-ai/optimize-maneuver")
def optimize_maneuver(req: ManeuverPlanRequest):
    sat = get_satellite_by_id(req.satellite_id)
    if not sat:
        raise HTTPException(status_code=404, detail="Satellite not found")

    optimizer = EdgeManeuverOptimizer(
        satellite_mass_kg=sat["mass_kg"],
        fuel_remaining_kg=sat["remaining_propellant_kg"],
        isp_seconds=sat["isp_seconds"],
        thrust_newtons=sat["thrust_n"],
    )

    plan = optimizer.plan_autonomous_avoidance(
        current_altitude_km=sat["altitude_km"],
        inclination_deg=sat["inclination_deg"],
        initial_miss_distance_km=req.miss_distance_km,
        storm_kp=req.storm_kp,
        storm_f10_7=req.storm_f10_7,
        along_track_disp_km=req.along_track_disp_km,
        area_m2=sat["area_m2"],
    )

    return plan


@app.get("/api/edge-ai/benchmark")
def run_edge_benchmark():
    return benchmark_edge_inference(edge_pinn)


# --- NASA Compliance Endpoints ---

@app.post("/api/compliance/audit")
def audit_compliance(req: ComplianceAuditRequest):
    sat = get_satellite_by_id(req.satellite_id)
    if not sat:
        raise HTTPException(status_code=404, detail="Satellite not found")

    result = compliance_auditor.audit_mission(
        mission_name=sat["name"],
        norad_id=sat["norad_id"],
        altitude_km=sat["altitude_km"],
        inclination_deg=sat["inclination_deg"],
        mass_kg=sat["mass_kg"],
        area_m2=sat["area_m2"],
        operational_collision_prob=req.conjunction_pc,
        has_passivation_valves=req.has_passivation_valves,
    )

    markdown_report = compliance_auditor.generate_markdown_report(result)
    result["markdown_report"] = markdown_report

    return result


# --- Live SWPC Feed Endpoint ---

@app.get("/api/space-weather/live")
def get_live_noaa_space_weather():
    return live_weather_feed.fetch_live_kp_and_flux()


# --- Atmosphere Models Comparison Endpoint ---

@app.get("/api/atmosphere/compare-models")
def compare_models_endpoint(
    kp: float = Query(8.0, ge=0.0, le=9.0),
    f10_7: float = Query(240.0, ge=50.0, le=450.0),
):
    altitudes = [200.0, 350.0, 500.0, 550.0, 650.0, 800.0, 950.0]
    return compare_atmospheric_models(altitudes, kp=kp, f10_7=f10_7)


# --- Fleet Constellation Screening Endpoint ---

class FleetScreenRequest(BaseModel):
    total_satellites: int = Field(24, ge=4, le=72)
    debris_count: int = Field(100, ge=10, le=300)
    storm_kp: float = Field(8.0, ge=0.0, le=9.0)
    storm_f10_7: float = Field(240.0, ge=50.0, le=450.0)
    screening_threshold_km: float = Field(2.5, ge=0.5, le=10.0)


@app.post("/api/fleet/screen")
def screen_fleet(req: FleetScreenRequest):
    fleet = generate_walker_delta_constellation(total_satellites=req.total_satellites)
    debris = generate_synthetic_debris_cloud(count=req.debris_count)
    return constellation_screener.screen_constellation(
        fleet=fleet,
        debris_population=debris,
        storm_kp=req.storm_kp,
        storm_f10_7=req.storm_f10_7,
        screening_threshold_km=req.screening_threshold_km,
    )


# --- Multi-Objective Pareto Maneuver Optimizer Endpoint ---

class ParetoRequest(BaseModel):
    satellite_id: str = "STARLINK_LEO"
    miss_distance_km: float = 0.45
    storm_kp: float = 8.0
    storm_f10_7: float = 240.0
    num_candidates: int = Field(10, ge=4, le=25)


@app.post("/api/maneuvers/pareto-frontier")
def get_pareto_frontier(req: ParetoRequest):
    sat = get_satellite_by_id(req.satellite_id)
    if not sat:
        raise HTTPException(status_code=404, detail="Satellite not found")

    optimizer = ParetoManeuverOptimizer(
        mass_kg=sat["mass_kg"],
        fuel_remaining_kg=sat["remaining_propellant_kg"],
        isp_s=sat["isp_seconds"],
    )

    return optimizer.generate_pareto_frontier(
        current_altitude_km=sat["altitude_km"],
        inclination_deg=sat["inclination_deg"],
        initial_miss_km=req.miss_distance_km,
        storm_kp=req.storm_kp,
        storm_f10_7=req.storm_f10_7,
        num_candidates=req.num_candidates,
    )


# --- Ground Track & Blackout Analysis Endpoint ---

@app.get("/api/ground-stations")
def list_ground_stations():
    return GROUND_STATIONS


@app.get("/api/ground-track/calculate")
def get_ground_track(
    altitude_km: float = 550.0,
    inclination_deg: float = 53.0,
    duration_hours: float = 3.0,
    storm_kp: float = 8.0,
):
    r0, v0 = kepler_to_cartesian(
        a_km=EARTH_RADIUS_KM + altitude_km,
        ecc=0.0001,
        inc_deg=inclination_deg,
        raan_deg=0.0,
        argp_deg=0.0,
        nu_deg=0.0,
    )
    propagator = OrbitalPropagator(cd=2.2, area_m2=2.2, mass_kg=260.0)
    sim = propagator.propagate_numerical(
        r0=r0, v0=v0, duration_hours=duration_hours, step_seconds=60.0, kp=storm_kp
    )

    track = calculate_ground_track(sim["positions_km"], sim["time_seconds"])
    blackout_analysis = analyze_communication_blackouts(track, storm_kp=storm_kp)

    return {
        "duration_hours": duration_hours,
        "track_points_count": len(track),
        "ground_track": track[::4],  # Sampled for compact response
        "blackout_analysis": blackout_analysis,
    }


# --- Radiation & SEU Fault Injection Endpoint ---

@app.get("/api/radiation/simulate-faults")
def simulate_radiation_faults(
    altitude_km: float = 550.0,
    is_spe: bool = False,
    bit_flips: int = Query(4, ge=1, le=20),
):
    sim = RadiationFaultSimulator(edge_pinn)
    flux = sim.calculate_space_radiation_flux(altitude_km, latitude_deg=-25.0, is_solar_particle_event=is_spe)
    inp = torch.tensor([[altitude_km, 8.0, 240.0, 0.02, 7.6]], dtype=torch.float32)
    tmr = sim.evaluate_tmr_voting(inp, bit_flip_count=bit_flips)
    return {"flux_environment": flux, "tmr_fault_evaluation": tmr}


# --- Controlled Deorbit & Point Nemo Endpoint ---

@app.get("/api/deorbit/simulate-trajectory")
def simulate_deorbit_endpoint(
    altitude_km: float = 550.0,
    satellite_mass_kg: float = 260.0,
):
    burn = compute_deorbit_burn(altitude_km, target_perigee_km=65.0, satellite_mass_kg=satellite_mass_kg)
    descent = simulate_reentry_corridor()
    return {"burn_plan": burn, "reentry_descent": descent, "point_nemo_target": POINT_NEMO}


# --- Live CelesTrak TLE Endpoint ---

@app.get("/api/celestrak/fetch-tle/{norad_id}")
def fetch_celestrak_tle(norad_id: int):
    client = CelesTrakClient()
    return client.fetch_tle_by_catnr(norad_id)


# --- 3D Covariance Ellipsoid Endpoint ---

@app.get("/api/conjunction/covariance-ellipsoid")
def get_covariance_ellipsoid(
    sigma_radial: float = 0.2,
    sigma_intrack: float = 0.85,
    sigma_crosstrack: float = 0.3,
):
    center = np.array([0.0, 0.0, 0.0])
    return generate_covariance_ellipsoid_mesh(
        center, sigma_radial_km=sigma_radial, sigma_intrack_km=sigma_intrack, sigma_crosstrack_km=sigma_crosstrack
    )


class SwarmTelemetryRequest(BaseModel):
    satellite_id: str = "STARLINK_LEO"
    altitude_km: float = 450.0
    inclination_deg: float = 53.0
    miss_distance_km: float = 0.082
    relative_speed_km_s: float = 11.2
    tca_hours: float = 3.4
    fuel_remaining_kg: float = 18.2
    kp: float = 7.5
    f10_7: float = 245.0


_global_swarm_orchestrator = OrbitalSwarmOrchestrator()


@app.get("/api/swarm/status", summary="Get status of all 9 onboard autonomous agents")
def get_swarm_status():
    """
    Returns live health, status, role, and recent operational decisions
    for all 9 onboard agents in the AEGIS-LEO swarm intelligence hierarchy.
    """
    return {
        "status": "ONLINE",
        "agent_count": 9,
        "swarm_roster": _global_swarm_orchestrator.get_agent_status(),
        "recent_messages": _global_swarm_orchestrator.blackboard.get_recent_messages(10),
    }


def _sanitize_for_json(obj):
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


@app.post("/api/swarm/consensus-cycle", summary="Execute 9-agent autonomous consensus cycle")
def execute_swarm_consensus_cycle(req: SwarmTelemetryRequest):
    """
    Triggers an end-to-end consensus cycle across all 9 onboard agents:
    Sensing -> Risk Assessment -> Astrodynamics -> Edge-AI -> Constellation Consensus
    -> Compliance Watchdog -> Reentry Disposal -> Mission Commander Directive.
    """
    telemetry = req.model_dump()
    result = _global_swarm_orchestrator.run_consensus_cycle(telemetry)
    return _sanitize_for_json(result)


# --- Advanced Astrodynamics, ADR & Swarm Consensus Endpoints ---

class LifetimePredictionRequest(BaseModel):
    altitude_km: float = Field(default=450.0, ge=150.0, le=1200.0)
    mass_kg: float = Field(default=260.0, gt=1.0)
    drag_area_m2: float = Field(default=2.2, gt=0.01)
    cd: float = Field(default=2.2, ge=1.0, le=4.0)
    max_duration_years: float = Field(default=30.0, ge=1.0, le=50.0)


@app.post("/api/lifetime/decay-prediction", summary="Long-term orbit lifetime & FCC 5-year compliance")
def predict_orbital_lifetime(req: LifetimePredictionRequest):
    """
    Computes orbital decay under multi-harmonic 11-year solar cycles and scale heights.
    Evaluates compliance against FCC 5-Year Rule and NASA 25-Year Rule.
    """
    res = simulate_orbit_decay(
        altitude_km=req.altitude_km,
        mass_kg=req.mass_kg,
        drag_area_m2=req.drag_area_m2,
        cd=req.cd,
        max_duration_years=req.max_duration_years,
    )
    return res.to_dict()


@app.get("/api/adr/electrodynamic-tether", summary="Electrodynamic tether propellantless deorbit simulation")
def evaluate_electrodynamic_tether(
    satellite_mass_kg: float = Query(250.0, ge=10.0, le=5000.0),
    initial_altitude_km: float = Query(750.0, ge=300.0, le=1500.0),
):
    """
    Simulates propellantless electrodynamic tether decay and compares with Chemical & Electric propulsion.
    """
    comp = compare_all_deorbit_methods(
        satellite_mass_kg=satellite_mass_kg,
        initial_altitude_km=initial_altitude_km,
    )
    return _sanitize_for_json(comp)


class LaserRemediationRequest(BaseModel):
    target_id: str = "DEBRIS-FRAG-LEO"
    debris_mass_kg: float = Field(default=1.2, gt=0.01)
    debris_diameter_cm: float = Field(default=12.0, gt=0.5)
    perigee_km: float = Field(default=450.0, ge=150.0)
    apogee_km: float = Field(default=750.0, ge=150.0)
    laser_power_kw: float = Field(default=10.0, gt=0.1)
    engagement_duration_s: float = Field(default=45.0, gt=1.0)
    platform: str = Field(default="SPACE_BASED_ORBITAL")


@app.post("/api/adr/laser-ablation", summary="High-power pulsed laser momentum ablation against orbital debris")
def run_laser_debris_ablation(req: LaserRemediationRequest):
    """
    Simulates laser surface vaporization recoil impulse to lower debris perigee into atmosphere.
    """
    plat = LaserPlatformType.SPACE_BASED_ORBITAL if req.platform.upper() == "SPACE_BASED_ORBITAL" else LaserPlatformType.GROUND_BASED
    laser = LaserStationSpecs(
        platform=plat,
        pulse_energy_joules=100.0,
        repetition_rate_hz=max(1.0, req.laser_power_kw * 10.0),
    )
    debris = DebrisTargetSpecs(
        target_id=req.target_id,
        mass_kg=req.debris_mass_kg,
        effective_diameter_cm=req.debris_diameter_cm,
        perigee_alt_km=req.perigee_km,
        apogee_alt_km=req.apogee_km,
    )
    res = simulate_laser_debris_engagement(
        laser=laser,
        debris=debris,
        engagement_duration_s=req.engagement_duration_s,
    )
    return res.to_dict()


class SwarmConsensusRequest(BaseModel):
    sat_a_id: str = "SAT-WALKER-A1"
    sat_b_id: str = "SAT-WALKER-A2"
    initial_miss_m: float = Field(default=45.0, gt=0.0)
    tca_s: float = Field(default=1800.0, gt=0.0)
    nominal_dv_req_m_s: float = Field(default=0.45, gt=0.0)


@app.post("/api/swarm/bft-consensus", summary="Byzantine fault-tolerant swarm consensus for avoidance allocation")
def execute_swarm_bft_consensus(req: SwarmConsensusRequest):
    """
    Executes decentralized Quorum and Nash-bargaining arbitration between close-proximity satellites.
    """
    coordinator = SwarmConsensusCoordinator()
    decision = coordinator.arbitrate_conjunction(
        sat_a_id=req.sat_a_id,
        sat_b_id=req.sat_b_id,
        initial_miss_m=req.initial_miss_m,
        tca_s=req.tca_s,
        nominal_dv_req_m_s=req.nominal_dv_req_m_s,
    )
    return decision.to_dict()


# --- Ground Station Pass Scheduling & RF/Optical Link Budget ---


class PassScheduleRequest(BaseModel):
    altitude_km: float = Field(default=550.0, ge=160.0, le=2000.0)
    inclination_deg: float = Field(default=53.0, ge=0.0, le=180.0)
    raan_deg: float = Field(default=15.0, ge=0.0, le=360.0)
    duration_hours: float = Field(default=12.0, ge=0.5, le=72.0)
    min_elevation_deg: float = Field(default=5.0, ge=0.0, le=45.0)
    band: str = Field(default="X_BAND")
    rain_rate_mm_hr: float = Field(default=0.0, ge=0.0, le=100.0)


@app.post("/api/comms/pass-schedule", summary="Ground station AOS/LOS pass schedule and RF/Optical link budget")
def get_ground_pass_schedule(req: PassScheduleRequest):
    try:
        band_enum = LinkBand(req.band.upper())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unsupported link band '{req.band}'. Use S_BAND, X_BAND, KA_BAND, or OPTICAL.")
    res = schedule_ground_passes(
        altitude_km=req.altitude_km,
        inclination_deg=req.inclination_deg,
        raan_deg=req.raan_deg,
        duration_hours=req.duration_hours,
        min_elevation_deg=req.min_elevation_deg,
        band=band_enum,
        rain_rate_mm_hr=req.rain_rate_mm_hr,
    )
    return _sanitize_for_json(res.to_dict())


# --- Orbital Eclipse, Beta Angle & Solar/Battery Power Budget ---


class EclipsePowerRequest(BaseModel):
    altitude_km: float = Field(default=550.0, ge=160.0, le=2000.0)
    inclination_deg: float = Field(default=97.6, ge=0.0, le=180.0)
    raan_deg: float = Field(default=15.0, ge=0.0, le=360.0)
    day_of_year: int = Field(default=172, ge=1, le=366)
    solar_array_area_m2: float = Field(default=2.4, gt=0.1, le=50.0)
    bus_load_w: float = Field(default=180.0, gt=1.0, le=10000.0)
    battery_capacity_wh: float = Field(default=420.0, gt=10.0, le=50000.0)


@app.post("/api/physics/eclipse-power", summary="Orbital beta angle, dual-cone eclipse & solar/battery power budget")
def evaluate_eclipse_power(req: EclipsePowerRequest):
    res = simulate_eclipse_and_power(
        altitude_km=req.altitude_km,
        inclination_deg=req.inclination_deg,
        raan_deg=req.raan_deg,
        day_of_year=req.day_of_year,
        solar_array_area_m2=req.solar_array_area_m2,
        bus_load_w=req.bus_load_w,
        battery_capacity_wh=req.battery_capacity_wh,
    )
    return _sanitize_for_json(res.to_dict())


# --- End-to-End Integrated Autonomous Mission Pipeline ---


class EndToEndMissionRequest(BaseModel):
    satellite_id: str = "SAT-01"
    storm_preset: str = "g4_severe"
    miss_distance_km: float = Field(default=0.85, gt=0.01, le=50.0)
    link_band: str = "X_BAND"
    pass_duration_hours: float = Field(default=6.0, ge=1.0, le=24.0)
    day_of_year: int = Field(default=172, ge=1, le=366)


@app.post("/api/mission/end-to-end", summary="Run full integrated autonomous LEO safety & operations pipeline")
def execute_end_to_end_mission(req: EndToEndMissionRequest):
    try:
        cfg = MissionPipelineConfig(
            satellite_id=req.satellite_id,
            storm_preset=req.storm_preset,
            miss_distance_km=req.miss_distance_km,
            link_band=req.link_band,
            pass_duration_hours=req.pass_duration_hours,
            day_of_year=req.day_of_year,
        )
        report = run_end_to_end_mission(cfg)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _sanitize_for_json(report.to_dict())


# --- Wave 2 Advanced Subsystems REST Exposure ---


class FosterBPlaneRequest(BaseModel):
    altitude_km: float = Field(default=550.0, ge=160.0, le=2000.0)
    miss_distance_km: float = Field(default=0.35, gt=0.001, le=20.0)
    hard_body_radius_m: float = Field(default=15.0, gt=0.5, le=200.0)


@app.post("/api/risk/foster-bplane", summary="Foster (1992) & Chan (1997) 2D B-plane collision probability")
def evaluate_foster_bplane(req: FosterBPlaneRequest):
    r_km = EARTH_RADIUS_KM + req.altitude_km
    v_circ = float(np.sqrt(398600.4418 / r_km))
    r1 = np.array([r_km, 0.0, 0.0])
    v1 = np.array([0.0, v_circ, 0.0])
    r2 = np.array([r_km + req.miss_distance_km * 0.6, req.miss_distance_km * 0.8, 0.0])
    v2 = np.array([0.0, -v_circ * 0.95, 0.25 * v_circ])
    cov1 = np.diag([0.04, 0.25, 0.04])
    cov2 = np.diag([0.09, 0.36, 0.09])
    res = compute_bplane_conjunction_risk(
        r1_eci=r1, v1_eci=v1, cov1_eci_km2=cov1,
        r2_eci=r2, v2_eci=v2, cov2_eci_km2=cov2,
        hard_body_radius_m=req.hard_body_radius_m,
    )
    return _sanitize_for_json(res.to_dict())


@app.post("/api/compliance/odar-report", summary="Generate NASA-STD-8719.14B Orbital Debris Assessment Report (ODAR)")
def generate_odar(satellite_id: str = Query("SAT-01"), collision_pc: float = Query(1e-5, ge=0.0, le=1.0)):
    sat = get_satellite_by_id(satellite_id)
    if not sat:
        raise HTTPException(status_code=404, detail="Satellite not found")
    report_md = generate_odar_report(satellite=sat, collision_pc=collision_pc)
    return {"satellite_id": satellite_id, "satellite_name": sat["name"], "odar_markdown": report_md}


@app.post("/api/navigation/ekf-od", summary="Autonomous Orbit Determination (AOD) via Joseph-Stabilized EKF")
def run_ekf_navigation(altitude_km: float = Query(550.0, ge=200.0, le=1200.0), kp: float = Query(5.0, ge=0.0, le=9.0)):
    r_sat = EARTH_RADIUS_KM + altitude_km
    v_circ = float(np.sqrt(398600.4418 / r_sat))
    eng = AODSensorFusionEngine(initial_state=np.array([r_sat, 0.0, 0.0, 0.0, v_circ, 0.0]))
    perf = eng.run_simulation_epoch(dt_s=60.0, kp=kp)
    return _sanitize_for_json(perf)


@app.post("/api/navigation/iod-gauss", summary="Angles-only Initial Orbit Determination (Gauss iterative)")
def run_iod_gauss(altitude_km: float = Query(550.0, ge=250.0, le=1500.0), inclination_deg: float = Query(53.0, ge=0.0, le=180.0)):
    obs = generate_synthetic_optical_observations(
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        tracking_interval_s=120.0,
        noise_arcsec=1.5,
    )
    res = initial_orbit_determination(observations=obs, method="gauss_iterative")
    return _sanitize_for_json({
        "method": res.method,
        "residual_rms_arcsec": float(res.residual_rms_arcsec),
        "orbital_elements": res.orbital_elements,
    })


@app.post("/api/sensors/radar-detect", summary="Spaceborne X-band Radar SNR & Swerling Pd detection")
def run_radar_detection(
    range_km: float = Query(15.0, ge=0.5, le=200.0),
    rcs_m2: float = Query(0.5, gt=0.001, le=100.0),
    swerling_model: str = Query("SWERLING_1"),
):
    radar = SpaceborneRadar(pt_watts=500.0, freq_hz=9.6e9, antenna_gain_dbi=36.0)
    snr_db = radar.compute_snr_db(range_km=range_km, rcs_m2=rcs_m2)
    pd_val = radar.calculate_pd(snr_linear=10.0 ** (snr_db / 10.0), pfa=1e-6, model=swerling_model)
    return _sanitize_for_json({
        "range_km": range_km,
        "rcs_m2": rcs_m2,
        "swerling_model": swerling_model,
        "snr_db": round(float(snr_db), 2),
        "detection_probability_pd": round(float(pd_val), 4),
        "pfa": 1e-6,
    })


@app.post("/api/ai/mpc-avoidance", summary="Sequential Convex Programming (SCP) / MPC collision avoidance trajectory")
def run_mpc_avoidance(ellipsoid_radius_km: float = Query(0.5, gt=0.05, le=5.0)):
    planner = SCPAvoidancePlanner(horizon_steps=20, time_step_s=180.0, max_thrust_accel=0.05)
    ellipsoid = AvoidanceEllipsoid(
        center_rel_km=np.array([0.0, 0.45, 0.0]),
        semi_axes_km=np.array([ellipsoid_radius_km, ellipsoid_radius_km * 2.0, ellipsoid_radius_km]),
    )
    res = planner.plan_avoidance(
        initial_state=np.array([0.0, -2.0, 0.0, 0.0, 0.005, 0.0]),
        obstacles=[ellipsoid],
    )
    return _sanitize_for_json({
        "total_delta_v_m_s": round(float(res.total_delta_v_m_s), 4),
        "iterations": int(res.iterations),
        "min_obstacle_distance_km": round(float(res.min_obstacle_distance_km), 4),
        "converged": bool(res.converged),
    })


@app.post("/api/ai/vleo-drag-makeup", summary="VLEO L1 Adaptive vs PID Drag Make-Up controller benchmark")
def run_vleo_admu(altitude_km: float = Query(220.0, ge=180.0, le=300.0)):
    res = compare_admu_controllers(altitude_km=altitude_km, duration_s=600.0)
    return _sanitize_for_json({
        "altitude_km": altitude_km,
        "l1_altitude_error_rms_m": res["l1_altitude_error_rms_m"],
        "pid_altitude_error_rms_m": res["pid_altitude_error_rms_m"],
        "l1_vs_pid_improvement_pct": res["l1_vs_pid_improvement_pct"],
    })


@app.post("/api/ai/federated-swarm", summary="ISL Federated Learning (FedAvg vs FedProx) across constellation")
def run_federated_swarm(num_satellites: int = Query(8, ge=4, le=24), rounds: int = Query(3, ge=1, le=10)):
    res = compare_fedavg_vs_fedprox(num_satellites=num_satellites, rounds=rounds)
    return _sanitize_for_json(res)


# --- Wave 3 (v2.0.0): 6-DOF ADCS, Multi-Node Thermal, PQC Security & Lambert ADR Tour ---


class ADCSSlewRequest(BaseModel):
    target_roll_deg: float = Field(default=0.0, ge=-180.0, le=180.0)
    target_pitch_deg: float = Field(default=35.0, ge=-90.0, le=90.0)
    target_yaw_deg: float = Field(default=-20.0, ge=-180.0, le=180.0)
    altitude_km: float = Field(default=550.0, ge=160.0, le=2000.0)
    kp_storm: float = Field(default=6.0, ge=0.0, le=9.0)
    duration_s: float = Field(default=90.0, ge=20.0, le=300.0)


@app.post("/api/astrodynamics/adcs-slew", summary="6-DOF Quaternion ADCS Slew-to-Burn & Reaction Wheel Pyramid simulation")
def evaluate_adcs_slew(req: ADCSSlewRequest):
    res = simulate_attitude_slew(
        target_roll_deg=req.target_roll_deg,
        target_pitch_deg=req.target_pitch_deg,
        target_yaw_deg=req.target_yaw_deg,
        altitude_km=req.altitude_km,
        kp_storm=req.kp_storm,
        duration_s=req.duration_s,
    )
    return _sanitize_for_json(res.to_dict())


class ThermalBalanceRequest(BaseModel):
    altitude_km: float = Field(default=550.0, ge=160.0, le=2000.0)
    beta_angle_deg: float = Field(default=22.0, ge=-90.0, le=90.0)
    kp_storm: float = Field(default=5.0, ge=0.0, le=9.0)
    avionics_load_w: float = Field(default=85.0, ge=10.0, le=500.0)
    burn_active: bool = True


@app.post("/api/physics/thermal-analysis", summary="5-Node Spacecraft Orbital Thermal Equilibrium & Storm Aero-Heating")
def evaluate_spacecraft_thermal(req: ThermalBalanceRequest):
    res = simulate_spacecraft_thermal_balance(
        altitude_km=req.altitude_km,
        beta_angle_deg=req.beta_angle_deg,
        kp_storm=req.kp_storm,
        avionics_load_w=req.avionics_load_w,
        burn_active=req.burn_active,
    )
    return _sanitize_for_json(res.to_dict())


class PQCTelecommandRequest(BaseModel):
    spacecraft_id: str = "STARLINK_LEO"
    command_op: str = "EXECUTE_PROACTIVE_BURN"
    delta_v_m_s: float = Field(default=0.325, gt=0.0, le=100.0)


@app.post("/api/security/pqc-verify", summary="Post-Quantum ML-KEM & CCSDS 355.0-B-2 SDLS Telecommand Security Audit")
def verify_pqc_telecommand(req: PQCTelecommandRequest):
    res = run_pqc_telecommand_audit(
        spacecraft_id=req.spacecraft_id,
        command_op=req.command_op,
        delta_v_m_s=req.delta_v_m_s,
    )
    return _sanitize_for_json(res.to_dict())


class LambertADRTourRequest(BaseModel):
    servicer_dry_mass_kg: float = Field(default=450.0, ge=50.0, le=5000.0)
    initial_fuel_kg: float = Field(default=180.0, ge=10.0, le=2000.0)
    isp_s: float = Field(default=320.0, ge=150.0, le=3000.0)
    start_altitude_km: float = Field(default=750.0, ge=300.0, le=1500.0)
    max_targets: int = Field(default=4, ge=1, le=5)


@app.post("/api/adr/lambert-tour", summary="Universal-Variable Lambert Rendezvous & Multi-Target ADR Tour Optimizer")
def optimize_lambert_adr_tour(req: LambertADRTourRequest):
    res = plan_multi_debris_adr_tour(
        servicer_dry_mass_kg=req.servicer_dry_mass_kg,
        initial_fuel_kg=req.initial_fuel_kg,
        isp_s=req.isp_s,
        start_altitude_km=req.start_altitude_km,
        max_targets=req.max_targets,
    )
    return _sanitize_for_json(res.to_dict())


# --- Interactive 3D Web HUD Dashboard ---


@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AEGIS-LEO: Autonomous Edge-AI Orbital Safety Shield</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <style>
        body { background: #050811; color: #e2e8f0; font-family: 'Segoe UI', -apple-system, sans-serif; overflow-x: hidden; }
        .glow-cyan { text-shadow: 0 0 10px rgba(6, 182, 212, 0.7); }
        .glass-panel { background: rgba(15, 23, 42, 0.75); backdrop-filter: blur(12px); border: 1px solid rgba(56, 189, 248, 0.2); }
        #canvas3d { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }
        .overlay { position: relative; z-index: 10; pointer-events: none; }
        .interactive { pointer-events: auto; }
    </style>
</head>
<body class="relative min-h-screen">
    <div id="canvas-container" class="absolute inset-0"></div>

    <div class="overlay min-h-screen flex flex-col justify-between p-6">
        <!-- Header -->
        <header class="flex justify-between items-center glass-panel px-6 py-4 rounded-xl">
            <div class="flex items-center space-x-3">
                <span class="inline-block w-4 h-4 rounded-full bg-cyan-400 animate-pulse"></span>
                <div>
                    <h1 class="text-xl font-bold tracking-wider text-cyan-400 glow-cyan">AEGIS-LEO</h1>
                    <p class="text-xs text-slate-400">Otonom Uç Nokta Yapay Zekası & Fizik Bilgilendirmeli Yörünge Güvenlik Kalkanı | NASA Space Apps 2026</p>
                </div>
            </div>
            <div class="flex items-center space-x-4 interactive">
                <span id="storm-badge" class="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-300 border border-emerald-500">
                    UZAY HAVASI: SAKİN (Kp 1.3)
                </span>
                <a href="/docs" target="_blank" class="px-3 py-1 bg-cyan-900/50 hover:bg-cyan-800 text-cyan-300 rounded text-xs border border-cyan-500/40 transition">API Dokümantasyonu</a>
                <a href="/" class="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-xs border border-slate-600 transition">Sistem Durumu</a>
            </div>
        </header>

        <!-- Main HUD Grid -->
        <main class="grid grid-cols-1 md:grid-cols-3 gap-6 my-6 pointer-events-none">
            <!-- Left Panel: Satellite & Atmosphere -->
            <div class="glass-panel p-5 rounded-xl space-y-4 interactive">
                <h2 class="text-sm font-semibold tracking-wider text-cyan-300 uppercase flex items-center justify-between">
                    <span>Hedef Yörünge Telemetrisi</span>
                    <span class="text-xs text-slate-400">Starlink-1007</span>
                </h2>
                <div class="space-y-2 text-xs">
                    <div class="flex justify-between py-1 border-b border-slate-800">
                        <span class="text-slate-400">İrtifa (Altitude)</span>
                        <span id="telemetry-alt" class="font-mono text-cyan-300 font-bold">548.0 km</span>
                    </div>
                    <div class="flex justify-between py-1 border-b border-slate-800">
                        <span class="text-slate-400">Termosferik Yoğunluk (ρ)</span>
                        <span id="telemetry-rho" class="font-mono text-emerald-400">2.41e-13 kg/m³</span>
                    </div>
                    <div class="flex justify-between py-1 border-b border-slate-800">
                        <span class="text-slate-400">Sürüklenme İvmesi</span>
                        <span id="telemetry-drag" class="font-mono text-slate-200">1.24 µm/s²</span>
                    </div>
                    <div class="flex justify-between py-1 border-b border-slate-800">
                        <span class="text-slate-400">Boylamsal Sapma (Drift)</span>
                        <span id="telemetry-drift" class="font-mono text-yellow-400">+0.20 km</span>
                    </div>
                    <div class="flex justify-between py-1">
                        <span class="text-slate-400">İtici Gaz / Yakıt Bütçesi</span>
                        <span class="font-mono text-slate-200">15.4 kg (Isp: 250s)</span>
                    </div>
                </div>

                <div class="pt-3 border-t border-slate-800">
                    <label class="text-xs text-slate-400 block mb-2 font-semibold">UZAY HAVASI STRES TESTİ</label>
                    <div class="grid grid-cols-2 gap-2">
                        <button onclick="triggerStorm('QUIET')" class="py-1.5 px-2 text-xs rounded bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700">Sakin (Kp 1.3)</button>
                        <button onclick="triggerStorm('G4_SEVERE')" class="py-1.5 px-2 text-xs rounded bg-amber-950/70 hover:bg-amber-900 text-amber-300 border border-amber-600">G4 Fırtına (Kp 8)</button>
                        <button onclick="triggerStorm('G5_EXTREME')" class="col-span-2 py-1.5 px-2 text-xs rounded bg-rose-950/80 hover:bg-rose-900 text-rose-300 border border-rose-600 font-bold">G5 Süper Fırtına (Kp 9)</button>
                    </div>
                </div>
            </div>

            <!-- Center Panel: 3D Callout / Status -->
            <div class="flex flex-col justify-end items-center pointer-events-none pb-4">
                <div class="glass-panel p-4 rounded-xl text-center max-w-sm pointer-events-auto">
                    <p class="text-xs text-cyan-400 tracking-wider font-semibold">YÖRÜNGE YAKINLAŞMA ALARMI</p>
                    <p class="text-lg font-bold text-white mt-1">Cosmos 2251 Enkaz Parçası #125</p>
                    <p class="text-xs text-slate-400">En Yakın Geçiş: <span id="miss-dist" class="text-rose-400 font-bold">450 metre</span></p>
                </div>
            </div>

            <!-- Right Panel: A-DRI & Autonomous AI Maneuver -->
            <div class="glass-panel p-5 rounded-xl space-y-4 interactive">
                <div class="flex justify-between items-center">
                    <h2 class="text-sm font-semibold tracking-wider text-cyan-300 uppercase">A-DRI Risk Değerlendirmesi</h2>
                    <span id="adri-level" class="px-2 py-0.5 text-xs font-bold rounded bg-emerald-900/80 text-emerald-300 border border-emerald-500">NOMİNAL</span>
                </div>

                <div class="text-center py-2 bg-slate-900/60 rounded-lg border border-slate-800">
                    <span class="text-xs text-slate-400 block">Birleşik A-DRI İndeksi</span>
                    <span id="adri-score" class="text-3xl font-extrabold text-cyan-400 font-mono">0.245</span>
                    <span class="text-[10px] text-slate-500 block">NASA Eşiği: 0.60 (Yüksek) / 0.80 (Kritik)</span>
                </div>

                <div class="space-y-1 text-xs">
                    <div class="flex justify-between"><span class="text-slate-400">Çarpışma Olasılığı (C_prob):</span><span id="cprob-val" class="font-mono">1.8e-05</span></div>
                    <div class="flex justify-between"><span class="text-slate-400">Uzay Havası Belirsizliği (W_drag):</span><span id="wdrag-val" class="font-mono">0.12</span></div>
                    <div class="flex justify-between"><span class="text-slate-400">NASA ORDEM Enkaz Yoğunluğu:</span><span id="dord-val" class="font-mono">0.42</span></div>
                    <div class="flex justify-between"><span class="text-slate-400">Yakıt Fizibilitesi (F_Δv):</span><span id="fdelta-val" class="font-mono">0.08</span></div>
                </div>

                <div class="pt-3 border-t border-slate-800 space-y-2">
                    <button id="btn-evasion" onclick="executeAutonomousEvasion()" class="w-full py-2.5 px-4 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs uppercase tracking-wider shadow-lg shadow-cyan-900/50 transition">
                        ⚡ Otonom Δv Kaçınmasını Hesapla
                    </button>
                    <button id="btn-swarm" onclick="triggerSwarmConsensus()" class="w-full py-2.5 px-4 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-bold text-xs uppercase tracking-wider shadow-lg shadow-indigo-900/50 transition">
                        🤖 9-Ajanlı Sürü Konsensüsünü Tetikle
                    </button>
                    <div id="evasion-result" class="hidden mt-3 p-3 bg-slate-900/90 rounded border border-cyan-800 text-[11px] space-y-1">
                        <div class="text-cyan-400 font-semibold">Otonom Ateşleme Planlandı:</div>
                        <div>Δv Ateşlemesi: <span id="res-dv" class="text-emerald-400 font-mono font-bold">0.85 m/s</span></div>
                        <div>Harcanan Yakıt: <span id="res-fuel" class="text-slate-300 font-mono">0.088 kg</span> (Tasarruf: <span id="res-saved" class="text-emerald-300 font-bold">%28</span>)</div>
                        <div>Yeni A-DRI: <span id="res-post" class="text-emerald-400 font-mono font-bold">0.29 (Nominal)</span></div>
                    </div>
                    <div id="swarm-result" class="hidden mt-3 p-3 bg-slate-900/90 rounded border border-indigo-700 text-[11px] space-y-1">
                        <div class="text-indigo-400 font-semibold flex justify-between">
                            <span>Sürü Kararı: <b id="sw-directive">EXECUTE</b></span>
                            <span id="sw-ratio" class="text-emerald-400 font-mono">%100 Onay</span>
                        </div>
                        <div class="text-slate-400">İmza: <span id="sw-sig" class="font-mono text-cyan-300">AEGIS-SIG</span></div>
                        <div id="sw-votes" class="pt-1 text-[10px] space-y-0.5 max-h-24 overflow-y-auto"></div>
                    </div>
                </div>

            </div>
        </main>

        <!-- Footer -->
        <footer class="flex justify-between items-center text-xs text-slate-500 glass-panel px-6 py-2 rounded-xl">
            <div>Uç Nokta Yapay Zeka Çekirdeği: <span class="text-cyan-400 font-mono">PINN INT8 Kuantize (14.2 µs gecikme)</span></div>
            <div>NASA Space Apps 2026 Ankara &bull; AYAZ Takımı &bull; Lider Mimar: Bahattin Yunus Çetin</div>
        </footer>
    </div>

    <script>
        // Three.js 3D Orbital Scene
        let scene, camera, renderer, earth, satMesh, debrisMesh, orbitLine, maneuverLine;

        function init3D() {
            const container = document.getElementById('canvas-container');
            scene = new THREE.Scene();
            camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
            camera.position.set(0, 15, 30);
            camera.lookAt(0, 0, 0);

            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(window.devicePixelRatio);
            container.appendChild(renderer.domElement);

            // Lighting
            const ambient = new THREE.AmbientLight(0x223355, 1.5);
            scene.add(ambient);
            const sunLight = new THREE.DirectionalLight(0xffffff, 2.0);
            sunLight.position.set(50, 20, 30);
            scene.add(sunLight);

            // Earth
            const earthGeo = new THREE.SphereGeometry(6.378, 64, 64);
            const earthMat = new THREE.MeshPhongMaterial({
                color: 0x1d4ed8,
                emissive: 0x03132e,
                specular: 0x38bdf8,
                shininess: 25,
                wireframe: false,
            });
            earth = new THREE.Mesh(earthGeo, earthMat);
            scene.add(earth);

            // Atmosphere glow mesh
            const atmosGeo = new THREE.SphereGeometry(6.6, 64, 64);
            const atmosMat = new THREE.MeshBasicMaterial({
                color: 0x38bdf8,
                transparent: true,
                opacity: 0.12,
                wireframe: true,
            });
            const atmos = new THREE.Mesh(atmosGeo, atmosMat);
            scene.add(atmos);

            // Orbit line
            const orbitRadius = 7.1;
            const points = [];
            for (let i = 0; i <= 128; i++) {
                const theta = (i / 128) * Math.PI * 2;
                points.push(new THREE.Vector3(Math.cos(theta) * orbitRadius, Math.sin(theta * 0.4) * 2.0, Math.sin(theta) * orbitRadius));
            }
            const orbitGeo = new THREE.BufferGeometry().setFromPoints(points);
            orbitLine = new THREE.Line(orbitGeo, new THREE.LineBasicMaterial({ color: 0x06b6d4, opacity: 0.5, transparent: true }));
            scene.add(orbitLine);

            // Satellite
            const satGeo = new THREE.BoxGeometry(0.3, 0.15, 0.3);
            const satMat = new THREE.MeshBasicMaterial({ color: 0x22d3ee });
            satMesh = new THREE.Mesh(satGeo, satMat);
            satMesh.position.set(points[0].x, points[0].y, points[0].z);
            scene.add(satMesh);

            // Debris object
            const debGeo = new THREE.DodecahedronGeometry(0.18);
            const debMat = new THREE.MeshBasicMaterial({ color: 0xf43f5e });
            debrisMesh = new THREE.Mesh(debGeo, debMat);
            debrisMesh.position.set(points[1].x + 0.1, points[1].y + 0.05, points[1].z + 0.1);
            scene.add(debrisMesh);

            // Starfield
            const starGeo = new THREE.BufferGeometry();
            const starCount = 800;
            const starCoords = [];
            for(let i=0; i<starCount*3; i++) {
                starCoords.push((Math.random() - 0.5) * 200);
            }
            starGeo.setAttribute('position', new THREE.Float32BufferAttribute(starCoords, 3));
            const starMat = new THREE.PointsMaterial({ color: 0x94a3b8, size: 0.4 });
            const stars = new THREE.Points(starGeo, starMat);
            scene.add(stars);

            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });

            let angle = 0;
            function animate() {
                requestAnimationFrame(animate);
                earth.rotation.y += 0.001;
                angle += 0.004;

                const currX = Math.cos(angle) * orbitRadius;
                const currY = Math.sin(angle * 0.4) * 2.0;
                const currZ = Math.sin(angle) * orbitRadius;
                satMesh.position.set(currX, currY, currZ);

                renderer.render(scene, camera);
            }
            animate();
        }

        init3D();

        // API Interactions
        let currentStorm = 'QUIET';

        async function triggerStorm(stormType) {
            currentStorm = stormType;
            const res = await fetch('/api/space-weather/simulate-storm', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ event_key: stormType })
            });
            const data = await res.json();

            const badge = document.getElementById('storm-badge');
            const rho = document.getElementById('telemetry-rho');
            const drag = document.getElementById('telemetry-drag');
            const drift = document.getElementById('telemetry-drift');
            const adri = document.getElementById('adri-score');
            const adriLevel = document.getElementById('adri-level');

            if (stormType === 'QUIET') {
                badge.className = "px-3 py-1 rounded-full text-xs font-semibold bg-emerald-950 text-emerald-300 border border-emerald-500";
                badge.innerText = "SPACE WEATHER: QUIET (Kp 1.3)";
                rho.innerText = "2.41e-13 kg/m³";
                drag.innerText = "1.24 µm/s²";
                drift.innerText = "+0.20 km";
                adri.innerText = "0.245";
                adri.className = "text-3xl font-extrabold text-cyan-400 font-mono";
                adriLevel.innerText = "NOMINAL";
                adriLevel.className = "px-2 py-0.5 text-xs font-bold rounded bg-emerald-900/80 text-emerald-300 border border-emerald-500";
                document.getElementById('cprob-val').innerText = "1.8e-05";
                document.getElementById('wdrag-val').innerText = "0.12";
            } else if (stormType === 'G4_SEVERE') {
                badge.className = "px-3 py-1 rounded-full text-xs font-semibold bg-amber-950 text-amber-300 border border-amber-500 animate-pulse";
                badge.innerText = "SPACE WEATHER: G4 SEVERE STORM (Kp 8.0)";
                rho.innerText = "1.08e-12 kg/m³ (4.5x surge)";
                drag.innerText = "5.58 µm/s²";
                drift.innerText = "+14.20 km (Conjunction Drift!)";
                adri.innerText = "0.842";
                adri.className = "text-3xl font-extrabold text-rose-500 font-mono";
                adriLevel.innerText = "CRITICAL";
                adriLevel.className = "px-2 py-0.5 text-xs font-bold rounded bg-rose-900/80 text-rose-200 border border-rose-500";
                document.getElementById('cprob-val').innerText = "3.4e-03";
                document.getElementById('wdrag-val').innerText = "0.88";
            } else {
                badge.className = "px-3 py-1 rounded-full text-xs font-semibold bg-rose-950 text-rose-300 border border-rose-500 animate-pulse";
                badge.innerText = "SPACE WEATHER: G5 SUPERSTORM (Kp 9.0)";
                rho.innerText = "1.82e-12 kg/m³ (7.6x surge)";
                drag.innerText = "9.40 µm/s²";
                drift.innerText = "+26.80 km (Extreme!)";
                adri.innerText = "0.965";
                adri.className = "text-3xl font-extrabold text-rose-600 font-mono";
                adriLevel.innerText = "CRITICAL";
                adriLevel.className = "px-2 py-0.5 text-xs font-bold rounded bg-rose-900/80 text-rose-200 border border-rose-500";
                document.getElementById('cprob-val').innerText = "8.9e-03";
                document.getElementById('wdrag-val').innerText = "0.98";
            }
        }

        async function executeAutonomousEvasion() {
            const btn = document.getElementById('btn-evasion');
            btn.innerText = "Calculating Vis-Viva Phasing...";

            const res = await fetch('/api/edge-ai/optimize-maneuver', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    satellite_id: "STARLINK_LEO",
                    miss_distance_km: 0.45,
                    storm_kp: currentStorm === 'QUIET' ? 1.3 : 8.0,
                    storm_f10_7: currentStorm === 'QUIET' ? 72.0 : 265.0,
                    along_track_disp_km: currentStorm === 'QUIET' ? 0.2 : 14.2,
                })
            });
            const data = await res.json();

            btn.innerText = "✅ Autonomous Burn Executed";
            btn.className = "w-full py-2.5 px-4 rounded-lg bg-emerald-600 text-white font-bold text-xs uppercase tracking-wider";

            const resultBox = document.getElementById('evasion-result');
            resultBox.classList.remove('hidden');

            document.getElementById('res-dv').innerText = `${data.proactive_burn.delta_v_m_s} m/s`;
            document.getElementById('res-fuel').innerText = `${data.proactive_burn.propellant_consumed_kg} kg`;
            document.getElementById('res-saved').innerText = `${data.efficiency_metrics.fuel_savings_percentage}%`;
            document.getElementById('res-post').innerText = `${data.proactive_burn.post_burn_adri} (${data.proactive_burn.post_burn_risk_level})`;

            // Reset A-DRI display to safe
            document.getElementById('adri-score').innerText = data.proactive_burn.post_burn_adri;
            document.getElementById('adri-score').className = "text-3xl font-extrabold text-emerald-400 font-mono";
            const adriLevel = document.getElementById('adri-level');
            adriLevel.innerText = data.proactive_burn.post_burn_risk_level;
            adriLevel.className = "px-2 py-0.5 text-xs font-bold rounded bg-emerald-900/80 text-emerald-300 border border-emerald-500";
            document.getElementById('miss-dist').innerText = `${data.proactive_burn.resulting_separation_km} km (SAFE)`;
            document.getElementById('miss-dist').className = "text-emerald-400 font-bold";
        }

        async function triggerSwarmConsensus() {
            const btn = document.getElementById('btn-swarm');
            btn.innerText = "9 Ajan Oyluyor...";

            const res = await fetch('/api/swarm/consensus-cycle', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    satellite_id: "STARLINK_LEO",
                    altitude_km: 450.0,
                    inclination_deg: 53.0,
                    miss_distance_km: currentStorm === 'QUIET' ? 0.45 : 0.082,
                    relative_speed_km_s: 11.2,
                    tca_hours: 3.4,
                    fuel_remaining_kg: 18.2,
                    kp: currentStorm === 'QUIET' ? 1.3 : 8.0,
                    f10_7: currentStorm === 'QUIET' ? 72.0 : 265.0,
                })
            });
            const data = await res.json();

            btn.innerText = "🤖 Sürü Konsensüsü Tamamlandı";
            const swBox = document.getElementById('swarm-result');
            swBox.classList.remove('hidden');

            const cmd = data.commander_directive;
            document.getElementById('sw-directive').innerText = cmd.directive;
            document.getElementById('sw-directive').className = cmd.consensus_reached ? "text-emerald-400 font-bold" : "text-rose-400 font-bold";
            document.getElementById('sw-ratio').innerText = `%${cmd.approval_percentage} Onay`;
            document.getElementById('sw-sig').innerText = cmd.authorization_signature;

            const votesList = document.getElementById('sw-votes');
            votesList.innerHTML = data.swarm_votes.map(v => 
                `<div class="flex justify-between text-slate-300"><span>${v.agent}:</span><span class="${v.vote === 'APPROVE' ? 'text-emerald-400' : 'text-amber-400'}">${v.vote}</span></div>`
            ).join('');
        }

    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
