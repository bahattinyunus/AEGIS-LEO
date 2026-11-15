"""
End-to-End Autonomous LEO Mission Simulation Pipeline
======================================================

Unifies all core AEGIS-LEO subsystems into a single deterministic mission
verification run:
  1. Satellite Catalog & Space Weather Storm Injection
  2. Multi-Model Thermospheric Density (NRLMSISE-00 / Jacchia-77 / Exponential)
  3. Orbital Eclipse, Beta Angle & Solar/Battery Power Budget
  4. Ground Station AOS/LOS Pass Scheduling & RF/Optical Link Budget
  5. Foster (1992) & Chan (1997) B-Plane + A-DRI Conjunction Risk Evaluation
  6. Edge-AI PINN Autonomous Avoidance Maneuver Synthesis
  7. 9-Agent Autonomous Swarm Consensus Cycle
  8. NASA-STD-8719.14B & FCC 5-Year Orbital Lifetime Compliance Audit
  9. 6-DOF ADCS Slew, 5-Node Thermal Balance & PQC Telecommand Verification
  10. Trapped Radiation / SAA Dose & SEU Exposure
  11. TCA Conjunction Screening & Post-Mission Disposal Transfer Budget
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np

from aegis_leo.agents.swarm_orchestrator import OrbitalSwarmOrchestrator
from aegis_leo.ai.edge_optimizer import EdgeManeuverOptimizer
from aegis_leo.compliance.nasa_std_8719 import NASAComplianceAuditor
from aegis_leo.constants import MU_EARTH_KM3_S2, EARTH_RADIUS_KM
from aegis_leo.data.catalog import get_satellite_by_id
from aegis_leo.data.space_weather import SpaceWeatherService
from aegis_leo.physics.atmosphere_models import compare_atmospheric_models
from aegis_leo.physics.eclipse_power import simulate_eclipse_and_power
from aegis_leo.physics.orbit_lifetime import simulate_orbit_decay
from aegis_leo.physics.pass_scheduler import LinkBand, schedule_ground_passes
from aegis_leo.physics.thermal_analyzer import simulate_spacecraft_thermal_balance
from aegis_leo.astrodynamics.attitude_adcs import simulate_attitude_slew
from aegis_leo.security.pqc_telecommand import run_pqc_telecommand_audit
from aegis_leo.risk.adri import ADRICalculator
from aegis_leo.risk.foster_bplane import compute_bplane_conjunction_risk
from aegis_leo.risk.tca_finder import OrbitalElements, build_intercepting_secondary, screen_conjunctions
from aegis_leo.physics.radiation_environment import analyze_radiation_environment
from aegis_leo.astrodynamics.orbit_transfer import plan_orbit_transfer


@dataclass
class MissionPipelineConfig:
    satellite_id: str = "SAT-01"
    storm_preset: str = "g4_severe"
    miss_distance_km: float = 0.85
    link_band: str = "X_BAND"
    pass_duration_hours: float = 6.0
    day_of_year: int = 172
    solar_array_area_m2: float = 2.4
    bus_load_w: float = 160.0
    battery_capacity_wh: float = 420.0


@dataclass
class MissionSimulationReport:
    timestamp_utc: str
    satellite_id: str
    satellite_name: str
    mission_readiness_score: float
    overall_status: str
    space_weather: Dict[str, Any]
    atmosphere: Dict[str, Any]
    eclipse_power: Dict[str, Any]
    ground_comms: Dict[str, Any]
    conjunction_risk: Dict[str, Any]
    avoidance_maneuver: Dict[str, Any]
    swarm_consensus: Dict[str, Any]
    compliance: Dict[str, Any]
    attitude_adcs: Dict[str, Any] = field(default_factory=dict)
    thermal_control: Dict[str, Any] = field(default_factory=dict)
    pqc_security: Dict[str, Any] = field(default_factory=dict)
    radiation_environment: Dict[str, Any] = field(default_factory=dict)
    tca_screening: Dict[str, Any] = field(default_factory=dict)
    disposal_transfer: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_end_to_end_mission(config: MissionPipelineConfig) -> MissionSimulationReport:
    """
    Execute a full-stack autonomous LEO safety & operations simulation pipeline.
    """
    sat = get_satellite_by_id(config.satellite_id)
    if not sat:
        raise ValueError(f"Satellite '{config.satellite_id}' not found in catalog.")

    # 1. Space Weather
    sw_service = SpaceWeatherService()
    weather = sw_service.set_storm_event(config.storm_preset.upper())
    kp = float(weather["kp"])
    f107 = float(weather["f10_7"])

    # 2. Multi-Model Atmosphere
    alt_km = float(sat["altitude_km"])
    atmos = compare_atmospheric_models(altitudes_km=[alt_km], kp=kp, f10_7=f107)

    # 3. Eclipse & Power Budget
    eclipse_report = simulate_eclipse_and_power(
        altitude_km=alt_km,
        inclination_deg=float(sat["inclination_deg"]),
        day_of_year=config.day_of_year,
        solar_array_area_m2=config.solar_array_area_m2,
        bus_load_w=config.bus_load_w,
        battery_capacity_wh=config.battery_capacity_wh,
    )

    # 4. Ground Station Pass & Link Budget
    band_enum = LinkBand(config.link_band.upper())
    pass_sched = schedule_ground_passes(
        altitude_km=alt_km,
        inclination_deg=float(sat["inclination_deg"]),
        duration_hours=config.pass_duration_hours,
        band=band_enum,
    )

    # 5. Foster B-Plane + A-DRI Conjunction Risk
    r_km = EARTH_RADIUS_KM + alt_km
    v_circ = math.sqrt(MU_EARTH_KM3_S2 / r_km)
    r1 = np.array([r_km, 0.0, 0.0], dtype=np.float64)
    v1 = np.array([0.0, v_circ, 0.0], dtype=np.float64)
    r2 = np.array([r_km + config.miss_distance_km * 0.6, config.miss_distance_km * 0.8, 0.0], dtype=np.float64)
    v2 = np.array([0.0, -v_circ * 0.95, 0.25 * v_circ], dtype=np.float64)
    cov1 = np.diag([0.04, 0.25, 0.04])
    cov2 = np.diag([0.09, 0.36, 0.09])

    bplane = compute_bplane_conjunction_risk(
        r1_eci=r1,
        v1_eci=v1,
        cov1_eci_km2=cov1,
        r2_eci=r2,
        v2_eci=v2,
        cov2_eci_km2=cov2,
        hard_body_radius_m=15.0,
    )

    adri_calc = ADRICalculator()
    adri = adri_calc.evaluate_risk(
        miss_distance_km=config.miss_distance_km,
        altitude_km=alt_km,
        inclination_deg=float(sat["inclination_deg"]),
        area_m2=float(sat["area_m2"]),
        mass_kg=float(sat["mass_kg"]),
        kp=kp,
        f10_7=f107,
        along_track_disp_km=12.0 if kp >= 5.0 else 0.5,
        remaining_propellant_kg=float(sat["remaining_propellant_kg"]),
    )

    # 6. Edge-AI Autonomous Avoidance Maneuver
    opt = EdgeManeuverOptimizer(
        satellite_mass_kg=float(sat["mass_kg"]),
        fuel_remaining_kg=float(sat["remaining_propellant_kg"]),
        isp_seconds=float(sat["isp_seconds"]),
        thrust_newtons=float(sat["thrust_n"]),
    )
    maneuver = opt.plan_autonomous_avoidance(
        current_altitude_km=alt_km,
        inclination_deg=float(sat["inclination_deg"]),
        initial_miss_distance_km=config.miss_distance_km,
        storm_kp=kp,
    )

    # 7. 9-Agent Swarm Consensus Cycle
    orchestrator = OrbitalSwarmOrchestrator()
    swarm_res = orchestrator.run_consensus_cycle({
        "satellite_id": config.satellite_id,
        "altitude_km": alt_km,
        "miss_distance_km": config.miss_distance_km,
        "kp": kp,
        "f10_7": f107,
        "fuel_remaining_kg": float(sat["remaining_propellant_kg"]),
    })

    # 8. NASA-STD-8719.14B & FCC 5-Year Lifetime Compliance
    auditor = NASAComplianceAuditor()
    nasa_audit = auditor.audit_mission(
        mission_name=sat["name"],
        norad_id=sat["norad_id"],
        altitude_km=alt_km,
        inclination_deg=float(sat["inclination_deg"]),
        mass_kg=float(sat["mass_kg"]),
        area_m2=float(sat["area_m2"]),
        operational_collision_prob=float(bplane.pc_foster),
        has_passivation_valves=True,
    )
    lifetime = simulate_orbit_decay(
        altitude_km=alt_km,
        mass_kg=float(sat["mass_kg"]),
        drag_area_m2=float(sat["area_m2"]),
        max_duration_years=15.0,
    )

    # 9. 6-DOF Attitude ADCS Slew-to-Burn Readiness
    adcs_report = simulate_attitude_slew(
        target_roll_deg=0.0,
        target_pitch_deg=30.0,
        target_yaw_deg=-15.0,
        altitude_km=alt_km,
        kp_storm=kp,
        f10_7=f107,
        duration_s=80.0,
    )

    # 10. Multi-Node Spacecraft Thermal Control Equilibrium
    thermal_report = simulate_spacecraft_thermal_balance(
        altitude_km=alt_km,
        beta_angle_deg=eclipse_report.beta_angle_deg,
        kp_storm=kp,
        f10_7=f107,
        avionics_load_w=min(120.0, config.bus_load_w * 0.5),
        burn_active=True,
    )

    # 11. Post-Quantum Cryptographic (PQC) Telecommand Verification
    pqc_report = run_pqc_telecommand_audit(
        spacecraft_id=config.satellite_id,
        command_op="EXECUTE_PROACTIVE_BURN",
        delta_v_m_s=float(maneuver["proactive_burn"].get("delta_v_m_s", 0.325)),
    )

    # 12. Trapped Radiation, SAA Exposure, TID & SEU Rate
    radiation = analyze_radiation_environment(
        altitude_km=alt_km,
        inclination_deg=float(sat["inclination_deg"]),
        kp=kp,
    )

    # 13. TCA Screening of a crossing debris object at the configured miss distance
    tca_target_s = 1500.0
    primary = OrbitalElements(altitude_km=alt_km, inclination_deg=float(sat["inclination_deg"]), name=sat["name"])
    primary.mean_anomaly_deg = math.degrees(-primary.mean_motion_rad_s * tca_target_s) % 360.0
    debris = build_intercepting_secondary(
        primary,
        tca_s=tca_target_s,
        inclination_deg=(float(sat["inclination_deg"]) + 45.0) % 180.0,
        radial_offset_km=config.miss_distance_km,
        include_j2=True,
    )
    tca_report = screen_conjunctions(primary, debris, duration_hours=3.0)

    # 14. Post-Mission Disposal Transfer Budget (lower to a fast-decay orbit)
    disposal_alt_km = max(300.0, min(alt_km - 50.0, 400.0))
    disposal = plan_orbit_transfer(
        initial_altitude_km=alt_km,
        target_altitude_km=disposal_alt_km,
        wet_mass_kg=float(sat["mass_kg"]),
        propellant_kg=float(sat["remaining_propellant_kg"]),
        isp_s=float(sat["isp_seconds"]),
        thrust_n=float(sat["thrust_n"]),
    )

    # Compute overall Mission Readiness Score (0 - 100)
    score = 100.0
    if not eclipse_report.power_margin_positive:
        score -= 15.0
    if pass_sched.total_passes == 0:
        score -= 10.0
    if adri["a_dri"] > 0.7:
        score -= 15.0
    nasa_is_compliant = nasa_audit.get("overall_status") == "COMPLIANT"
    if not nasa_is_compliant:
        score -= 20.0
    if not lifetime.fcc_5year_compliant:
        score -= 10.0
    if radiation.radiation_status == "TID_LIMIT_EXCEEDED":
        score -= 5.0
    score = max(0.0, min(100.0, round(score, 1)))

    overall_status = "MISSION_READY" if score >= 80.0 else ("CAUTION" if score >= 60.0 else "CRITICAL_REVIEW")

    return MissionSimulationReport(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        satellite_id=config.satellite_id,
        satellite_name=sat["name"],
        mission_readiness_score=score,
        overall_status=overall_status,
        space_weather={
            "event_name": weather["name"],
            "kp": kp,
            "f10_7": f107,
        },
        atmosphere=atmos,
        eclipse_power={
            "beta_angle_deg": eclipse_report.beta_angle_deg,
            "eclipse_fraction": eclipse_report.eclipse_fraction,
            "sunlit_duration_s": eclipse_report.sunlit_duration_s,
            "eclipse_duration_s": eclipse_report.eclipse_duration_s,
            "sso_required_inclination_deg": eclipse_report.sso_required_inclination_deg,
            "net_orbit_energy_wh": eclipse_report.net_orbit_energy_wh,
            "max_dod_pct": eclipse_report.max_dod_pct,
            "power_margin_positive": eclipse_report.power_margin_positive,
            "temp_min_c": eclipse_report.temp_min_c,
            "temp_max_c": eclipse_report.temp_max_c,
        },
        ground_comms={
            "band": pass_sched.band,
            "total_passes": pass_sched.total_passes,
            "total_contact_time_s": pass_sched.total_contact_time_s,
            "total_data_downlinked_mb": pass_sched.total_data_downlinked_mb,
            "best_station": pass_sched.best_station,
        },
        conjunction_risk={
            "a_dri": adri["a_dri"],
            "risk_level": adri["risk_level"],
            "foster_pc": float(bplane.pc_foster),
            "chan_pc": float(bplane.pc_chan),
            "max_pc": float(bplane.pc_max),
            "miss_distance_km": config.miss_distance_km,
        },
        avoidance_maneuver=maneuver["proactive_burn"],
        swarm_consensus={
            "status": "CONSENSUS_REACHED" if swarm_res["commander_directive"].get("consensus_reached") else "VETOED_OR_DEFERRED",
            "weighted_approval": float(swarm_res["commander_directive"].get("approval_ratio", 1.0 if swarm_res["commander_directive"].get("consensus_reached") else 0.0)),
            "directive_hash": swarm_res["commander_directive"].get("authorization_signature", "N/A"),
        },
        compliance={
            "nasa_std_8719_compliant": nasa_is_compliant,
            "fcc_5yr_compliant": lifetime.fcc_5year_compliant,
            "estimated_lifetime_years": lifetime.estimated_lifetime_years,
        },
        attitude_adcs={
            "burn_attitude_ready": adcs_report.burn_attitude_ready,
            "settling_time_s": adcs_report.settling_time_s,
            "final_pointing_error_arcsec": adcs_report.final_pointing_error_arcsec,
            "peak_wheel_speed_rpm": adcs_report.peak_wheel_speed_rpm,
        },
        thermal_control={
            "thermal_status": thermal_report.thermal_status,
            "all_nodes_within_limits": thermal_report.all_nodes_within_limits,
            "heater_energy_wh": thermal_report.heater_energy_wh,
            "peak_aero_heating_w": thermal_report.peak_aero_heating_w,
        },
        pqc_security={
            "algorithm": pqc_report.algorithm,
            "authentic_command_verified": pqc_report.authentic_command_verified,
            "replay_attack_blocked": pqc_report.replay_attack_blocked,
            "tamper_attack_blocked": pqc_report.tamper_attack_blocked,
            "merkle_root_hash": pqc_report.merkle_root_hash,
        },
        radiation_environment={
            "radiation_status": radiation.radiation_status,
            "saa_exposure_fraction": radiation.saa_exposure_fraction,
            "saa_passes_per_day": radiation.saa_passes_per_day,
            "peak_proton_flux_cm2_s": radiation.peak_proton_flux_cm2_s,
            "tid_krad_per_year": radiation.tid_krad_per_year,
            "seu_per_day": radiation.seu_per_day,
            "recommended_scrub_interval_s": radiation.recommended_scrub_interval_s,
        },
        tca_screening={
            "screening_status": tca_report.screening_status,
            "min_miss_distance_km": tca_report.min_miss_distance_km,
            "tca_s": tca_report.tca_of_min_miss_s,
            "relative_speed_km_s": tca_report.relative_speed_at_min_km_s,
            "events_within_threshold": tca_report.events_within_threshold,
        },
        disposal_transfer={
            "target_altitude_km": disposal_alt_km,
            "strategy": disposal.recommended_strategy,
            "total_delta_v_m_s": disposal.total_delta_v_m_s,
            "propellant_required_kg": disposal.propellant_required_kg,
            "feasible": disposal.feasible,
        },
    )
