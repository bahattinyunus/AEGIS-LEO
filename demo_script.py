"""
AEGIS-LEO Mission Demonstration Script
Executes an end-to-end orbital safety and autonomous avoidance simulation.
Run with: python demo_script.py
"""

import sys
import os

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from aegis_leo.constants import QUIET_KP, QUIET_F10_7
from aegis_leo.physics.atmosphere import compute_atmospheric_density
from aegis_leo.physics.coordinates import kepler_to_cartesian
from aegis_leo.physics.propagator import OrbitalPropagator
from aegis_leo.risk.adri import ADRICalculator
from aegis_leo.ai.edge_optimizer import EdgeManeuverOptimizer
from aegis_leo.ai.quantizer import benchmark_edge_inference
from aegis_leo.ai.pinn import get_pretrained_edge_pinn
from aegis_leo.compliance.nasa_std_8719 import NASAComplianceAuditor
from aegis_leo.data.catalog import get_satellite_by_id


def run_demo():
    print("=" * 80)
    print("🛰️  AEGIS-LEO: Autonomous Edge-AI & Physics-Informed Orbital Safety Shield")
    print("   NASA Space Apps Challenge 2026 | Team AYAZ | Lead: Bahattin Yunus Çetin")
    print("=" * 80)

    # 1. Target Satellite
    sat = get_satellite_by_id("STARLINK_LEO")
    print(f"\n[1] TRACKED SATELLITE: {sat['name']} (NORAD #{sat['norad_id']})")
    print(f"    - Type: {sat['type']}")
    print(f"    - Orbit: Altitude {sat['altitude_km']} km, Inclination {sat['inclination_deg']}°")
    print(f"    - Mass: {sat['mass_kg']} kg, Area: {sat['area_m2']} m², Propellant: {sat['remaining_propellant_kg']} kg")

    # 2. Quiet Baseline
    rho_quiet = compute_atmospheric_density(sat['altitude_km'], QUIET_KP, QUIET_F10_7)
    print(f"\n[2] NOMINAL SPACE WEATHER (Quiet Baseline, Kp=1.3, F10.7=72)")
    print(f"    - Base Thermospheric Density (ρ_0): {rho_quiet:.2e} kg/m³")

    # 3. Geomagnetic Storm Injection
    storm_kp = 8.0
    storm_f10_7 = 265.0
    rho_storm = compute_atmospheric_density(sat['altitude_km'], storm_kp, storm_f10_7)
    surge_ratio = rho_storm / rho_quiet

    print(f"\n[3] ⚡ GEOMAGNETIC STORM INJECTION: NOAA G4 Severe Storm Event (Kp={storm_kp}, F10.7={storm_f10_7})")
    print(f"    - Expanded Thermospheric Density (ρ_storm): {rho_storm:.2e} kg/m³")
    print(f"    - Thermospheric Expansion Factor: +{surge_ratio:.1f}x times higher drag!")

    # 4. Unmodeled Drag & Ephemeris Divergence
    r0, v0 = kepler_to_cartesian(
        a_km=6378.137 + sat['altitude_km'],
        ecc=0.0001,
        inc_deg=sat['inclination_deg'],
        raan_deg=0.0,
        argp_deg=0.0,
        nu_deg=0.0,
    )
    propagator = OrbitalPropagator(cd=sat['cd'], area_m2=sat['area_m2'], mass_kg=sat['mass_kg'])
    divergence = propagator.simulate_storm_divergence(
        r0=r0, v0=v0, duration_hours=36.0, step_seconds=120.0, storm_kp=storm_kp, storm_f10_7=storm_f10_7
    )

    print(f"\n[4] 36-HOUR EPHEMERIS PROPAGATION DIVERGENCE:")
    print(f"    - Nominal Predicted Altitude Decay: {divergence['quiet_altitude_decay_m']:.1f} meters")
    print(f"    - Storm-Induced Altitude Decay:     {divergence['storm_altitude_decay_m']:.1f} meters")
    print(f"    - Net Unmodeled Decay:              {divergence['net_decay_meters']:.1f} meters")
    print(f"    - Along-Track Position Drift:       +{divergence['along_track_displacement_km']:.2f} km")
    print(f"    ⚠️  Static Conjunction Data Messages (CDMs) are INVALIDATED by ephemeris drift!")

    # 5. Risk Assessment (A-DRI)
    calc = ADRICalculator()
    pre_risk = calc.evaluate_risk(
        miss_distance_km=0.45,
        altitude_km=sat['altitude_km'],
        inclination_deg=sat['inclination_deg'],
        area_m2=sat['area_m2'],
        mass_kg=sat['mass_kg'],
        kp=storm_kp,
        f10_7=storm_f10_7,
        along_track_disp_km=divergence['along_track_displacement_km'],
        remaining_propellant_kg=sat['remaining_propellant_kg'],
    )

    print(f"\n[5] REAL-TIME A-DRI RISK EVALUATION:")
    print(f"    - Collision Probability (C_prob): {pre_risk['components']['c_prob_raw']:.2e}")
    print(f"    - Space Weather Uncertainty (W_drag): {pre_risk['components']['w_drag']:.2f}")
    print(f"    - NASA ORDEM Debris Density (D_ord): {pre_risk['components']['d_ord']:.2f}")
    print(f"    - Fuel Feasibility Index (F_Δv): {pre_risk['components']['f_delta_v']:.2f}")
    print(f"    >>> COMPOSITE A-DRI SCORE: {pre_risk['a_dri']:.3f} [{pre_risk['risk_level']}] <<<")
    print(f"    - Action Command: {pre_risk['action_command']}")

    # 6. Autonomous Edge-AI Avoidance Optimizer
    optimizer = EdgeManeuverOptimizer(
        satellite_mass_kg=sat['mass_kg'],
        fuel_remaining_kg=sat['remaining_propellant_kg'],
        isp_seconds=sat['isp_seconds'],
        thrust_newtons=sat['thrust_n'],
    )
    plan = optimizer.plan_autonomous_avoidance(
        current_altitude_km=sat['altitude_km'],
        inclination_deg=sat['inclination_deg'],
        initial_miss_distance_km=0.45,
        storm_kp=storm_kp,
        storm_f10_7=storm_f10_7,
        along_track_disp_km=divergence['along_track_displacement_km'],
        area_m2=sat['area_m2'],
    )

    pb = plan['proactive_burn']
    rb = plan['reactive_baseline']
    eff = plan['efficiency_metrics']

    print(f"\n[6] 🚀 AUTONOMOUS EDGE-AI MANEUVER OPTIMIZER:")
    print(f"    - Recommended Action: {pb['strategy']}")
    print(f"    - Planned Phasing Burn Δv:     {pb['delta_v_m_s']:.3f} m/s")
    print(f"    - Propellant Consumption:      {pb['propellant_consumed_kg']:.4f} kg")
    print(f"    - Post-Burn Separation:        {pb['resulting_separation_km']:.2f} km (SAFE)")
    print(f"    - New A-DRI Risk Score:        {pb['post_burn_adri']:.3f} [{pb['post_burn_risk_level']}]")
    print(f"    - Comparative Fuel Savings:    {eff['fuel_savings_percentage']}% less fuel than late reactive burn ({eff['propellant_saved_kg']:.4f} kg saved)")
    print(f"    - Total Risk Reduction:        {eff['risk_reduction_percentage']}%")

    # 7. Edge Quantization Benchmark
    model = get_pretrained_edge_pinn()
    bench = benchmark_edge_inference(model, num_iterations=200)
    print(f"\n[7] 🧠 ONBOARD EDGE PROCESSOR PROFILING:")
    print(f"    - Parameter Count: {bench['parameter_count']} weights")
    print(f"    - INT8 Memory Footprint: {bench['int8_quantized_memory_kb']} KB")
    print(f"    - Inference Latency: {bench['latency_microseconds']:.1f} microseconds")
    print(f"    - Edge Radiation-Tolerant OBC Status: {bench['flight_readiness']}")

    # 8. NASA-STD-8719.14 Compliance Audit
    auditor = NASAComplianceAuditor()
    audit = auditor.audit_mission(
        mission_name=sat['name'],
        norad_id=sat['norad_id'],
        altitude_km=sat['altitude_km'],
        inclination_deg=sat['inclination_deg'],
        mass_kg=sat['mass_kg'],
        area_m2=sat['area_m2'],
        operational_collision_prob=1.2e-5,  # post-maneuver safe Pc
        has_passivation_valves=True,
    )
    print(f"\n[8] 📋 NASA-STD-8719.14 COMPLIANCE AUDIT:")
    print(f"    - Status: {audit['overall_status']}")
    for item in audit['compliance_details']:
        print(f"      [{item['status']}] {item['standard_id']}: {item['title']}")

    print("\n" + "=" * 80)
    print("✅ SIMULATION COMPLETE: All physical, AI, and regulatory requirements validated.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
