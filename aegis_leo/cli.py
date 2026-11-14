"""
AEGIS-LEO Command Line Interface (CLI)
Provides terminal-based command execution for mission operators and automated pipelines.
"""

import sys
import os
import torch

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
from datetime import datetime, timezone

from aegis_leo.constants import QUIET_KP, QUIET_F10_7
from aegis_leo.data.catalog import SATELLITE_CATALOG, get_satellite_by_id
from aegis_leo.data.space_weather import SpaceWeatherService
from aegis_leo.risk.adri import ADRICalculator
from aegis_leo.ai.edge_optimizer import EdgeManeuverOptimizer
from aegis_leo.ai.quantizer import benchmark_edge_inference
from aegis_leo.ai.pinn import get_pretrained_edge_pinn
from aegis_leo.risk.monte_carlo import MonteCarloConjunctionSimulator
from aegis_leo.risk.kessler_cascade import nasa_breakup_model
from aegis_leo.physics.deorbit_simulation import compute_deorbit_burn, simulate_reentry_corridor, POINT_NEMO
from aegis_leo.ai.radiation_fault_simulator import RadiationFaultSimulator
from aegis_leo.data.celestrak_live import CelesTrakClient
from aegis_leo.compliance.nasa_std_8719 import NASAComplianceAuditor
from aegis_leo.agents.swarm_orchestrator import OrbitalSwarmOrchestrator
from aegis_leo.physics.orbit_lifetime import simulate_orbit_decay
from aegis_leo.adr.electrodynamic_tether import compare_all_deorbit_methods
from aegis_leo.adr.laser_ablation import simulate_laser_debris_engagement, LaserStationSpecs, DebrisTargetSpecs, LaserPlatformType
from aegis_leo.agents.swarm_consensus import SwarmConsensusCoordinator


def cmd_status(args):
    weather = SpaceWeatherService().get_current_conditions()
    pinn = get_pretrained_edge_pinn()
    bench = benchmark_edge_inference(pinn, num_iterations=100)

    print("\n" + "=" * 60)
    print(">> AEGIS-LEO SSA SHIELD - SYSTEM STATUS <<")
    print("=" * 60)
    print(f"Time (UTC):          {datetime.now(timezone.utc).isoformat()}")
    print(f"Space Weather Event: {weather['name']}")
    print(f"Current Kp Index:    {weather['kp']} | F10.7 Flux: {weather['f10_7']} SFU")
    print(f"Edge-AI Core Status: ONLINE ({bench['latency_microseconds']} µs latency, {bench['int8_quantized_memory_kb']} KB)")
    print(f"Tracked Satellites:  {len(SATELLITE_CATALOG)} cataloged targets")
    print("=" * 60 + "\n")


def cmd_eval(args):
    sat = get_satellite_by_id(args.sat)
    if not sat:
        print(f"Error: Satellite '{args.sat}' not found in catalog.")
        sys.exit(1)

    calc = ADRICalculator()
    risk = calc.evaluate_risk(
        miss_distance_km=args.miss,
        altitude_km=sat["altitude_km"],
        inclination_deg=sat["inclination_deg"],
        area_m2=sat["area_m2"],
        mass_kg=sat["mass_kg"],
        kp=args.kp,
        f10_7=args.f107,
        along_track_disp_km=14.2 if args.kp >= 5.0 else 0.2,
        remaining_propellant_kg=sat["remaining_propellant_kg"],
    )

    print(f"\n--- Conjunction Risk Evaluation: {sat['name']} ---")
    print(f"Miss Distance: {args.miss} km | Kp: {args.kp}")
    print(f"A-DRI Score:   {risk['a_dri']:.4f} [{risk['risk_level']}]")
    print(f"Action:        {risk['action_command']}")
    print(f"Components:    C_prob={risk['components']['c_prob_normalized']:.2f}, W_drag={risk['components']['w_drag']:.2f}, D_ord={risk['components']['d_ord']:.2f}, F_Delta_v={risk['components']['f_delta_v']:.2f}\n")


def cmd_burn(args):
    sat = get_satellite_by_id(args.sat)
    if not sat:
        print(f"Error: Satellite '{args.sat}' not found.")
        sys.exit(1)

    opt = EdgeManeuverOptimizer(
        satellite_mass_kg=sat["mass_kg"],
        fuel_remaining_kg=sat["remaining_propellant_kg"],
        isp_seconds=sat["isp_seconds"],
        thrust_newtons=sat["thrust_n"],
    )
    plan = opt.plan_autonomous_avoidance(
        current_altitude_km=sat["altitude_km"],
        inclination_deg=sat["inclination_deg"],
        initial_miss_distance_km=args.miss,
        storm_kp=args.kp,
    )
    pb = plan["proactive_burn"]
    eff = plan["efficiency_metrics"]

    print(f"\n--- Autonomous Evasion Plan: {sat['name']} ---")
    print(f"Strategy:            {pb['strategy']}")
    print(f"Burn Delta-v:        {pb['delta_v_m_s']:.3f} m/s")
    print(f"Propellant Used:     {pb['propellant_consumed_kg']:.4f} kg")
    print(f"Resulting Clearance: {pb['resulting_separation_km']:.2f} km")
    print(f"Post-Burn A-DRI:     {pb['post_burn_adri']} [{pb['post_burn_risk_level']}]")
    print(f"Propellant Saved:    {eff['fuel_savings_percentage']}% vs late ground maneuver\n")


def cmd_monte_carlo(args):
    mc = MonteCarloConjunctionSimulator()
    res = mc.run_simulation(nominal_miss_km=args.miss, num_trials=args.trials)
    print(f"\n--- Monte Carlo Conjunction Dispersion ({res['num_trials']} runs) ---")
    print(f"Nominal Miss:  {res['nominal_miss_km']} km")
    print(f"Empirical Pc:  {res['empirical_pc']:.2e} ({res['collisions_detected']} collisions)")
    print(f"Analytical Pc: {res['analytical_pc']:.2e}")
    print(f"Simulated Min: {res['min_simulated_miss_km']} km | Mean: {res['mean_simulated_miss_km']} km\n")


def cmd_breakup(args):
    res = nasa_breakup_model(target_mass_kg=args.target_mass, projectile_mass_kg=args.proj_mass)
    print(f"\n--- NASA Standard Breakup Model ---")
    print(f"Total Mass:           {res['total_mass_kg']} kg ({res['collision_type']})")
    print(f"Fragments > 10 cm:    {res['fragments_gt_10cm']:,} trackable fragments")
    print(f"Fragments > 1 cm:     {res['fragments_gt_1cm']:,} lethal fragments\n")


def cmd_deorbit(args):
    burn = compute_deorbit_burn(initial_altitude_km=args.alt, satellite_mass_kg=args.mass)
    descent = simulate_reentry_corridor()

    print(f"\n--- Controlled Deorbit & Atmospheric Entry Analysis ---")
    print(f"Initial Altitude:    {args.alt} km -> Target Perigee: {burn['target_perigee_km']} km")
    print(f"Retrograde Delta-v:  {burn['delta_v_retrograde_m_s']} m/s")
    print(f"Propellant Required: {burn['propellant_required_kg']} kg")
    print(f"Target Drop Zone:    {descent['target_drop_zone']}")
    print(f"Peak Stagnation Heat:{descent['peak_stagnation_heat_flux_w_cm2']} W/cm^2")
    print(f"NASA Req 4.7 Status: {'COMPLIANT' if descent['nasa_req_4_7_compliant'] else 'NON_COMPLIANT'} (Ec = {descent['casualty_expectation_ec']:.1e})\n")


def cmd_rad_test(args):
    sim = RadiationFaultSimulator(get_pretrained_edge_pinn())
    flux = sim.calculate_space_radiation_flux(altitude_km=args.alt, latitude_deg=-25.0, is_solar_particle_event=args.spe)
    inp = torch.tensor([[args.alt, 8.0, 240.0, 0.02, 7.6]], dtype=torch.float32)
    tmr = sim.evaluate_tmr_voting(inp, bit_flip_count=args.flips)

    print(f"\n--- Space Radiation & TMR Fault Injection Test ---")
    print(f"Environment:         Alt={args.alt} km, SAA Zone={'YES' if flux['in_saa_zone'] else 'NO'}, SPE={'YES' if flux['spe_active'] else 'NO'}")
    print(f"Proton Flux:         {flux['total_flux_p_cm2_s']} p/cm^2/s (Est. SEU Rate: {flux['estimated_seu_rate_per_mb_day']} /Mb/day)")
    print(f"Bit Flips Injected:  {tmr['injected_bit_flips']} weights")
    print(f"Single-Core Error:   {tmr['single_core_seu_error']}")
    print(f"TMR Voted Error:     {tmr['tmr_voted_error']} (Resilience: {tmr['resilience_percentage']}%)")
    print(f"Fault Masked:        {'SUCCESS - 100% MASKED' if tmr['fault_masked_successfully'] else 'FAILED'}\n")


def cmd_celestrak(args):
    client = CelesTrakClient()
    res = client.fetch_tle_by_catnr(args.norad)
    print(f"\n--- CelesTrak Live TLE: {res['name']} (NORAD #{args.norad}) ---")
    print(f"Status: {res['status']} | Source: {res['source']}")
    print(f"Line 1: {res['line1']}")
    print(f"Line 2: {res['line2']}\n")


def cmd_swarm(args):
    orchestrator = OrbitalSwarmOrchestrator()
    telemetry = {
        "satellite_id": args.sat,
        "altitude_km": args.alt,
        "inclination_deg": 53.0,
        "miss_distance_km": args.miss,
        "kp": args.kp,
        "f10_7": args.f107,
        "fuel_remaining_kg": args.fuel,
    }
    result = orchestrator.run_consensus_cycle(telemetry)

    print("\n" + "=" * 75)
    print(">> AEGIS-LEO 9-AGENT AUTONOMOUS ORBITAL SWARM CONSENSUS CYCLE <<")
    print("=" * 75)
    print(f"Cycle ID:    #{result['cycle_id']} | UTC: {result['timestamp']}")
    print(f"Target Sat:  {telemetry['satellite_id']} (Alt: {telemetry['altitude_km']} km, Fuel: {telemetry['fuel_remaining_kg']} kg)")
    print(f"Telemetry:   Miss={telemetry['miss_distance_km']*1000:.1f} m, Kp={telemetry['kp']:.1f}, F10.7={telemetry['f10_7']:.0f} SFU")
    print("-" * 75)
    print("SWARM ROSTER & ONBOARD AGENT DECISIONS (9 AGENTS):")
    for code, agent_res in result["agent_results"].items():
        name = agent_res.get("agent", code)
        status = agent_res.get("status", "ACTIVE")
        decision = agent_res.get("decision", "Nominal")
        print(f"  [{status:<6}] {name:<28}: {decision}")

    print("-" * 75)
    print("CONSENSUS VOTING LEDGER:")
    for vote in result["swarm_votes"]:
        agent_name = vote.get("agent", "")
        v_choice = vote.get("vote", "")
        conf = vote.get("confidence", 1.0) * 100
        reason = vote.get("reason", "Approved")
        print(f"  * {agent_name:<28} -> [{v_choice:<8}] (conf: {conf:.0f}%): {reason}")

    cmd = result["commander_directive"]
    print("=" * 75)
    print(f">> MISSION COMMANDER DIRECTIVE: {cmd['directive']}")
    print(f"   Status:            {cmd['status']}")
    print(f"   Consensus Ratio:   {cmd['approval_percentage']}% ({cmd['approval_ratio']*100:.1f}% approved)")
    print(f"   Crypto Signature:  {cmd['authorization_signature']}")
    print("=" * 75 + "\n")


def cmd_lifetime(args):
    res = simulate_orbit_decay(
        altitude_km=args.alt,
        mass_kg=args.mass,
        drag_area_m2=args.area,
        cd=args.cd,
    )
    print("\n" + "=" * 65)
    print(">> ORBIT LIFETIME & REGULATORY DEORBIT AUDIT <<")
    print("=" * 65)
    print(f"Initial Altitude:      {res.initial_altitude_km:.1f} km")
    print(f"Ballistic Coefficient: {res.ballistic_coefficient_kg_m2:.2f} kg/m²")
    print(f"Estimated Lifetime:    {res.estimated_lifetime_years:.2f} years ({res.reentry_date_estimate_days:.0f} days)")
    print(f"FCC 5-Year Mandate:    {'[PASS] COMPLIANT' if res.fcc_5year_compliant else '[FAIL] NON-COMPLIANT'}")
    print(f"NASA 25-Year Rule:     {'[PASS] COMPLIANT' if res.nasa_25year_compliant else '[FAIL] NON-COMPLIANT'}")
    print(f"Mean Solar Activity:   {res.solar_flux_avg:.1f} SFU (11-Year Cycle)")
    print(f"Status:                {res.status}")
    print("=" * 65 + "\n")


def cmd_tether(args):
    comp = compare_all_deorbit_methods(
        satellite_mass_kg=args.mass,
        initial_altitude_km=args.alt,
    )
    print("\n" + "=" * 70)
    print(">> ELECTRODYNAMIC TETHER & DEORBIT PROPULSION TRADE-OFF <<")
    print("=" * 70)
    print(f"Satellite Mass: {args.mass:.1f} kg | Initial Altitude: {args.alt:.1f} km")
    print("-" * 70)
    for method, metrics in comp["methods"].items():
        print(f"  * {method:<28}: {metrics['duration_days']:>7.1f} days | Propellant: {metrics['propellant_mass_kg']:>5.1f} kg | Mass: {metrics['subsystem_mass_kg']:>5.1f} kg")
    print("-" * 70)
    print(f">> RECOMMENDED STRATEGY: {comp['recommended_method']}")
    print("=" * 70 + "\n")


def cmd_laser(args):
    laser = LaserStationSpecs(
        platform=LaserPlatformType.SPACE_BASED_ORBITAL if args.space else LaserPlatformType.GROUND_BASED,
        pulse_energy_joules=100.0,
        repetition_rate_hz=args.power * 10.0,
    )
    debris = DebrisTargetSpecs(
        mass_kg=args.mass,
        effective_diameter_cm=args.dia,
        perigee_alt_km=args.perigee,
        apogee_alt_km=args.apogee,
    )
    res = simulate_laser_debris_engagement(
        laser=laser,
        debris=debris,
        engagement_duration_s=args.duration,
    )
    print("\n" + "=" * 68)
    print(">> HIGH-POWER PULSED LASER ORBITAL DEBRIS REMEDIATION (LODR) <<")
    print("=" * 68)
    print(f"Target ID:          {res.target_id} ({debris.mass_kg} kg, {debris.effective_diameter_cm} cm)")
    print(f"Platform:           {laser.platform.value} ({laser.average_power_kw:.1f} kW average beam)")
    print(f"Pulses on Target:   {res.pulses_on_target} / {res.total_pulses_fired} ({res.engagement_duration_s:.0f}s pass)")
    print(f"Target Fluence:     {res.average_fluence_j_cm2:.2f} J/cm² (Coupling Cm: {res.recoil_momentum_coupling_u_ns_j:.1f} µN·s/J)")
    print(f"Recoil Impulse:     {res.total_impulse_n_s*1000:.1f} mN·s -> Delta-V: {res.delta_v_m_s*1000:.2f} mm/s")
    print(f"Perigee Drop:       {res.perigee_drop_km*1000:.1f} m (New Perigee: {res.new_perigee_alt_km:.2f} km)")
    print(f"Decay Accelerated:  {res.deorbit_accelerated_years:.1f} years saved")
    print(f"Fragmentation Risk: {res.fragmentation_risk_score*100:.2f}% (Safe < 5%)")
    print("=" * 68 + "\n")


def cmd_consensus(args):
    coord = SwarmConsensusCoordinator()
    decision = coord.arbitrate_conjunction(
        sat_a_id=args.sat_a,
        sat_b_id=args.sat_b,
        initial_miss_m=args.miss,
        tca_s=args.tca,
        nominal_dv_req_m_s=args.dv,
    )
    print("\n" + "=" * 70)
    print(">> BYZANTINE FAULT-TOLERANT SWARM MANEUVER QUORUM ARBITRATION <<")
    print("=" * 70)
    print(f"Proposal ID:        {decision.proposal_id}")
    print(f"Elected Maneuver:   {decision.elected_satellite_to_maneuver} (Hold: {decision.holding_satellite})")
    print(f"Assigned Delta-V:   {decision.required_delta_v_m_s:.3f} m/s (Vector: {decision.delta_v_vector})")
    print(f"Consensus Quorum:   {decision.quorum_percentage:.1f}% ({decision.total_peers} peers, BFT F={decision.byzantine_tolerance_f})")
    print(f"Secondary Risk:     {'CLEARED - NO SECONDARY CONJUNCTION' if decision.secondary_risk_cleared else 'WARNING - SECONDARY HAZARD'}")
    print(f"Decision State:     {decision.agreement_state.value}")
    print("-" * 70)
    print("AUDIT TRAIL:")
    for event in decision.audit_trail:
        print(f"  * {event}")
    print("=" * 70 + "\n")



def main():
    parser = argparse.ArgumentParser(prog="aegis", description="AEGIS-LEO SSA Mission Control CLI")
    subparsers = parser.add_subparsers(dest="command")

    # status
    p_status = subparsers.add_parser("status", help="Show system status and space weather")
    p_status.set_defaults(func=cmd_status)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate conjunction risk (A-DRI)")
    p_eval.add_argument("--sat", default="STARLINK_LEO", help="Satellite ID")
    p_eval.add_argument("--miss", type=float, default=0.45, help="Miss distance (km)")
    p_eval.add_argument("--kp", type=float, default=8.0, help="Geomagnetic Kp index")
    p_eval.add_argument("--f107", type=float, default=240.0, help="F10.7 solar flux")
    p_eval.set_defaults(func=cmd_eval)

    # burn
    p_burn = subparsers.add_parser("burn", help="Plan autonomous avoidance maneuver")
    p_burn.add_argument("--sat", default="STARLINK_LEO", help="Satellite ID")
    p_burn.add_argument("--miss", type=float, default=0.45, help="Miss distance (km)")
    p_burn.add_argument("--kp", type=float, default=8.0, help="Geomagnetic Kp index")
    p_burn.set_defaults(func=cmd_burn)

    # monte-carlo
    p_mc = subparsers.add_parser("monte-carlo", help="Run Monte Carlo covariance dispersion")
    p_mc.add_argument("--miss", type=float, default=0.45, help="Nominal miss distance (km)")
    p_mc.add_argument("--trials", type=int, default=1000, help="Number of trials")
    p_mc.set_defaults(func=cmd_monte_carlo)

    # breakup
    p_brk = subparsers.add_parser("breakup", help="NASA standard breakup fragmentation model")
    p_brk.add_argument("--target-mass", type=float, default=260.0, help="Target satellite mass (kg)")
    p_brk.add_argument("--proj-mass", type=float, default=5.0, help="Projectile debris mass (kg)")
    p_brk.set_defaults(func=cmd_breakup)

    # deorbit
    p_deo = subparsers.add_parser("deorbit", help="Controlled deorbit & Point Nemo entry corridor")
    p_deo.add_argument("--alt", type=float, default=550.0, help="Current altitude (km)")
    p_deo.add_argument("--mass", type=float, default=260.0, help="Satellite mass (kg)")
    p_deo.set_defaults(func=cmd_deorbit)

    # rad-test
    p_rad = subparsers.add_parser("rad-test", help="Simulate space radiation & TMR fault masking")
    p_rad.add_argument("--alt", type=float, default=550.0, help="Altitude (km)")
    p_rad.add_argument("--spe", action="store_true", help="Simulate Solar Particle Event")
    p_rad.add_argument("--flips", type=int, default=4, help="Number of SEU bit flips to inject")
    p_rad.set_defaults(func=cmd_rad_test)

    # celestrak
    p_cel = subparsers.add_parser("celestrak", help="Fetch live TLE by NORAD Catalog Number")
    p_cel.add_argument("--norad", type=int, default=25544, help="NORAD catalog number (e.g. 25544 for ISS)")
    p_cel.set_defaults(func=cmd_celestrak)

    # swarm
    p_swm = subparsers.add_parser("swarm", help="Run 9-agent autonomous orbital swarm consensus cycle")
    p_swm.add_argument("--sat", default="STARLINK_LEO", help="Target satellite ID")
    p_swm.add_argument("--alt", type=float, default=450.0, help="Orbital altitude (km)")
    p_swm.add_argument("--miss", type=float, default=0.082, help="Conjunction miss distance (km)")
    p_swm.add_argument("--kp", type=float, default=7.5, help="Geomagnetic Kp index")
    p_swm.add_argument("--f107", type=float, default=245.0, help="F10.7 solar flux (SFU)")
    p_swm.add_argument("--fuel", type=float, default=18.5, help="Propellant remaining (kg)")
    p_swm.set_defaults(func=cmd_swarm)

    # lifetime
    p_life = subparsers.add_parser("lifetime", help="Predict orbit lifetime & check FCC 5-year compliance")
    p_life.add_argument("--alt", type=float, default=450.0, help="Orbital altitude (km)")
    p_life.add_argument("--mass", type=float, default=260.0, help="Spacecraft mass (kg)")
    p_life.add_argument("--area", type=float, default=2.2, help="Drag cross-section area (m²)")
    p_life.add_argument("--cd", type=float, default=2.2, help="Drag coefficient Cd")
    p_life.set_defaults(func=cmd_lifetime)

    # tether
    p_teth = subparsers.add_parser("tether", help="Electrodynamic tether propellantless deorbit comparison")
    p_teth.add_argument("--mass", type=float, default=250.0, help="Satellite mass (kg)")
    p_teth.add_argument("--alt", type=float, default=750.0, help="Initial altitude (km)")
    p_teth.set_defaults(func=cmd_tether)

    # laser
    p_las = subparsers.add_parser("laser", help="Laser orbital debris remediation momentum ablation")
    p_las.add_argument("--mass", type=float, default=1.2, help="Debris target mass (kg)")
    p_las.add_argument("--dia", type=float, default=12.0, help="Debris diameter (cm)")
    p_las.add_argument("--perigee", type=float, default=450.0, help="Target perigee altitude (km)")
    p_las.add_argument("--apogee", type=float, default=750.0, help="Target apogee altitude (km)")
    p_las.add_argument("--power", type=float, default=10.0, help="Laser average power (kW)")
    p_las.add_argument("--duration", type=float, default=45.0, help="Engagement duration (s)")
    p_las.add_argument("--space", action="store_true", help="Space-based platform (default ground-based)")
    p_las.set_defaults(func=cmd_laser)

    # consensus
    p_con = subparsers.add_parser("consensus", help="Swarm Byzantine fault-tolerant maneuver consensus")
    p_con.add_argument("--sat-a", default="SAT-WALKER-A1", help="First satellite ID")
    p_con.add_argument("--sat-b", default="SAT-WALKER-A2", help="Second satellite ID")
    p_con.add_argument("--miss", type=float, default=45.0, help="Miss distance (m)")
    p_con.add_argument("--tca", type=float, default=1800.0, help="Time to closest approach (s)")
    p_con.add_argument("--dv", type=float, default=0.45, help="Nominal Delta-V required (m/s)")
    p_con.set_defaults(func=cmd_consensus)


    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
