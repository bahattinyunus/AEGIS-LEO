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

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
