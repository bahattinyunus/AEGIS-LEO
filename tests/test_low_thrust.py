"""
Unit Tests for Continuous Low-Thrust Orbital Avoidance Trajectory Planner
Validates Edelbaum analytical approximations, Gauss phasing equations,
and RK4 numerical micro-thrust integration for Hall thrusters.
"""

import math
import pytest
import numpy as np

from aegis_leo.constants import EARTH_RADIUS_KM, EARTH_MU
from aegis_leo.ai.low_thrust import (
    ElectricThruster,
    PropulsionType,
    SteeringLaw,
    EdelbaumApproximator,
    MicroThrustIntegrator,
    LowThrustAvoidancePlanner,
    LowThrustPlan,
)
from aegis_leo.physics.coordinates import kepler_to_cartesian


def test_electric_thruster_catalog_and_properties():
    """Verify built-in thruster models and physical parameter properties."""
    spt100 = ElectricThruster.spt_100()
    assert spt100.name == "SPT-100 Hall Thruster"
    assert spt100.propulsion_type == PropulsionType.HALL_EFFECT
    assert spt100.thrust_n == 0.083
    assert spt100.isp_s == 1600.0
    assert spt100.power_w == 1350.0
    assert spt100.propellant == "Xenon"
    assert spt100.mass_flow_rate_kg_s > 0.0
    assert 50.0 < spt100.thrust_to_power_mn_per_kw < 70.0

    starlink = ElectricThruster.starlink_hall()
    assert starlink.propellant == "Krypton"
    assert starlink.thrust_n == 0.080
    assert starlink.isp_s == 1800.0

    bht200 = ElectricThruster.bht_200()
    assert bht200.thrust_n == 0.013
    assert bht200.power_w == 200.0

    next_ion = ElectricThruster.next_ion()
    assert next_ion.propulsion_type == PropulsionType.GRIDDED_ION
    assert next_ion.isp_s == 4190.0


def test_edelbaum_coplanar_and_combined_delta_v():
    """Verify Edelbaum analytical Delta-v approximations."""
    r1 = 6878.137  # ~500 km LEO
    r2 = 6978.137  # ~600 km LEO

    # 1. Coplanar altitude transfer: Delta-v = |v1 - v2|
    dv_coplanar = EdelbaumApproximator.coplanar_transfer_delta_v(r1, r2)
    v1 = math.sqrt(EARTH_MU / r1) * 1000.0
    v2 = math.sqrt(EARTH_MU / r2) * 1000.0
    expected_dv = abs(v1 - v2)
    assert pytest.approx(dv_coplanar, rel=1e-5) == expected_dv
    assert 50.0 < dv_coplanar < 60.0  # Approx 55 m/s for 100 km LEO shift

    # 2. Combined transfer with small inclination change (0.1 deg)
    delta_i_rad = math.radians(0.1)
    dv_comb = EdelbaumApproximator.combined_transfer_delta_v(r1, r2, delta_i_rad)
    assert dv_comb > dv_coplanar

    # 3. Pure plane change
    dv_plane = EdelbaumApproximator.pure_inclination_delta_v(r1, delta_i_rad)
    assert dv_plane > 0.0
    assert pytest.approx(dv_comb, rel=0.01) == math.sqrt(dv_coplanar**2 + dv_plane**2)


def test_edelbaum_phasing_and_burn_duration_solver():
    """Verify Gauss Variational along-track displacement and burn solver."""
    mass = 250.0   # kg
    thrust = 0.080 # 80 mN
    acc = thrust / mass

    # 1. Quadratic along-track growth: Delta s = 1.5 * a * t^2
    t_burn = 3600.0  # 1 hour
    disp_m = EdelbaumApproximator.phasing_along_track_displacement(t_burn, thrust, mass)
    expected_disp = 1.5 * acc * (t_burn**2)
    assert pytest.approx(disp_m, rel=1e-5) == expected_disp
    assert disp_m > 5000.0  # Over 5 km clearance in 1 hour

    # 2. Radial displacement
    r_km = 6878.137
    rad_m = EdelbaumApproximator.radial_displacement(t_burn, thrust, mass, r_km)
    assert rad_m > 0.0
    assert rad_m < disp_m  # Along-track displacement is much larger than radial in LEO

    # 3. Solver test: find tau to achieve 5 km clearance with 24 hours lead time
    target_clearance_m = 5000.0
    lead_time_s = 24.0 * 3600.0
    tau_s, achieved = EdelbaumApproximator.solve_required_burn_duration(
        target_clearance_m, lead_time_s, thrust, mass
    )
    assert achieved is True
    assert 0.0 < tau_s < 1200.0  # Needs less than 20 minutes of micro-thrust 24h early!

    # Verify that executing tau_s followed by coast yields exact target clearance
    total_disp = EdelbaumApproximator.total_phasing_displacement(
        burn_duration_s=tau_s,
        coast_duration_s=lead_time_s - tau_s,
        thrust_n=thrust,
        mass_kg=mass,
    )
    assert pytest.approx(total_disp, rel=1e-4) == target_clearance_m


def test_micro_thrust_integrator_eclipse_and_steering():
    """Test integrator eclipse detection and unit steering vector computation."""
    thruster = ElectricThruster.starlink_hall()
    integrator = MicroThrustIntegrator(thruster=thruster, allow_eclipse_thrust=False)

    # Position on sunlit side (+X when sun is at +X)
    r_sunlit = np.array([6878.0, 0.0, 0.0])
    assert integrator.is_in_eclipse(r_sunlit) is False

    # Position in Earth shadow (-X behind Earth)
    r_shadow = np.array([-6878.0, 100.0, 50.0])
    assert integrator.is_in_eclipse(r_shadow) is True

    # Position behind Earth but outside cylinder
    r_outside = np.array([-6878.0, 7000.0, 0.0])
    assert integrator.is_in_eclipse(r_outside) is False

    # Steering vector directions
    r_vec = np.array([6878.0, 0.0, 0.0])
    v_vec = np.array([0.0, 7.6, 0.0])

    t_pro = integrator.compute_steering_unit_vector(r_vec, v_vec, SteeringLaw.TANGENTIAL_PROGRADE)
    assert np.allclose(t_pro, np.array([0.0, 1.0, 0.0]))

    t_retro = integrator.compute_steering_unit_vector(r_vec, v_vec, SteeringLaw.TANGENTIAL_RETROGRADE)
    assert np.allclose(t_retro, np.array([0.0, -1.0, 0.0]))

    t_norm = integrator.compute_steering_unit_vector(r_vec, v_vec, SteeringLaw.CROSS_TRACK_NORMAL)
    assert np.allclose(t_norm, np.array([0.0, 0.0, 1.0]))


def test_micro_thrust_rk4_propagation_and_fuel_consumption():
    """Verify continuous micro-thrust RK4 integration, orbit raising, and propellant depletion."""
    thruster = ElectricThruster.starlink_hall()
    dry_mass = 240.0
    fuel_mass = 20.0
    integrator = MicroThrustIntegrator(
        thruster=thruster,
        dry_mass_kg=dry_mass,
        fuel_mass_kg=fuel_mass,
        allow_eclipse_thrust=True,
    )

    r0, v0 = kepler_to_cartesian(a_km=6878.137, ecc=0.001, inc_deg=53.0, raan_deg=0.0, argp_deg=0.0, nu_deg=0.0)

    # Propagate for 1800 seconds (30 minutes) with prograde micro-thrust
    duration = 1800.0
    trajectory = integrator.propagate(
        r0=r0,
        v0=v0,
        total_duration_s=duration,
        step_s=30.0,
        steering_law=SteeringLaw.TANGENTIAL_PROGRADE,
        burn_start_s=0.0,
        burn_duration_s=duration,
    )

    traj_ballistic = integrator.propagate(
        r0=r0,
        v0=v0,
        total_duration_s=duration,
        step_s=30.0,
        steering_law=SteeringLaw.COAST,
    )

    assert len(trajectory) == 61
    assert trajectory[0].time_s == 0.0
    assert trajectory[-1].time_s == duration

    # Check that propellant was consumed
    initial_mass = dry_mass + fuel_mass
    final_mass = trajectory[-1].mass_kg
    assert final_mass < initial_mass
    expected_m_dot = thruster.thrust_n / (thruster.isp_s * 9.80665)
    expected_fuel = expected_m_dot * duration
    assert pytest.approx(initial_mass - final_mass, rel=0.01) == expected_fuel

    # Mechanical energy of powered trajectory should be higher than ballistic trajectory at same epoch
    final_r = trajectory[-1].r_eci_km
    final_v = trajectory[-1].v_eci_km
    r_bal = traj_ballistic[-1].r_eci_km
    v_bal = traj_ballistic[-1].v_eci_km

    e_powered = 0.5 * np.linalg.norm(final_v)**2 - EARTH_MU / np.linalg.norm(final_r)
    e_ballistic = 0.5 * np.linalg.norm(v_bal)**2 - EARTH_MU / np.linalg.norm(r_bal)
    assert e_powered > e_ballistic
    work_done_km2_s2 = e_powered - e_ballistic
    assert 0.003 < work_done_km2_s2 < 0.006


def test_low_thrust_avoidance_planner_analytical():
    """Verify analytical avoidance planning and comparison with chemical thrusters."""
    planner = LowThrustAvoidancePlanner(
        thruster=ElectricThruster.starlink_hall(),
        satellite_dry_mass_kg=240.0,
        fuel_mass_kg=20.0,
    )

    plan = planner.plan_analytical_avoidance(
        current_altitude_km=550.0,
        lead_time_hours=24.0,
        target_clearance_km=5.0,
        initial_miss_km=0.3,
    )

    assert plan.status == "OPTIMAL_PLAN_FOUND"
    assert plan.thruster_name == "Starlink Krypton Hall Thruster"
    assert plan.miss_distance_tca_km >= 5.0
    assert plan.burn_duration_seconds > 0.0
    assert plan.propellant_used_kg > 0.0
    assert plan.propellant_remaining_kg < 20.0
    assert plan.electrical_energy_kwh > 0.0
    # Electric propulsion should save >75% fuel compared to chemical 220s Isp
    assert plan.savings_vs_chemical_percent > 75.0


def test_low_thrust_avoidance_planner_numerical():
    """Verify high-precision numerical micro-thrust avoidance planning against debris."""
    planner = LowThrustAvoidancePlanner(
        thruster=ElectricThruster.starlink_hall(),
        satellite_dry_mass_kg=240.0,
        fuel_mass_kg=20.0,
    )

    r0, v0 = kepler_to_cartesian(a_km=6878.137, ecc=0.0005, inc_deg=53.0, raan_deg=20.0, argp_deg=0.0, nu_deg=0.0)

    # Predict unperturbed position at 12 hours
    lead_hours = 12.0
    t_seconds = lead_hours * 3600.0
    unperturbed = planner.integrator.propagate(
        r0=r0, v0=v0, total_duration_s=t_seconds, step_s=60.0, steering_law=SteeringLaw.COAST
    )
    r_sat_nominal_tca = unperturbed[-1].r_eci_km

    # Simulate a dangerous conjunction where debris passes within 250 meters of nominal position
    r_debris_tca = r_sat_nominal_tca + np.array([0.15, 0.15, 0.1])

    plan = planner.plan_numerical_avoidance(
        r0_sat=r0,
        v0_sat=v0,
        r_debris_tca=r_debris_tca,
        lead_time_hours=lead_hours,
        target_clearance_km=5.0,
        strategy=SteeringLaw.TANGENTIAL_PROGRADE,
        step_s=60.0,
    )

    assert plan.status == "NUMERICAL_AVOIDANCE_CONVERGED"
    assert plan.miss_distance_initial_km < 0.3
    assert plan.miss_distance_tca_km >= 4.0
    assert plan.clearance_gain_km > 3.5
    assert plan.trajectory is not None
    assert len(plan.trajectory) > 10


def test_avoidance_and_restoration_mission_cycle():
    """Verify 3-phase mission: Avoidance Micro-Burn -> TCA Coast -> Recovery Counter-Burn."""
    planner = LowThrustAvoidancePlanner(
        thruster=ElectricThruster.starlink_hall(),
        satellite_dry_mass_kg=240.0,
        fuel_mass_kg=20.0,
    )

    r0, v0 = kepler_to_cartesian(a_km=6878.137, ecc=0.0002, inc_deg=45.0, raan_deg=10.0, argp_deg=0.0, nu_deg=0.0)

    mission = planner.plan_avoidance_and_restoration(
        r0_sat=r0,
        v0_sat=v0,
        lead_time_hours=12.0,
        target_clearance_km=5.0,
        restoration_delay_hours=4.0,
        step_s=60.0,
    )

    assert mission["status"] == "AVOIDANCE_AND_RESTORATION_COMPLETED"
    assert mission["phase1_avoidance_burn_duration_s"] > 0.0
    assert mission["phase3_restoration_burn_duration_s"] > 0.0
    assert mission["total_propellant_used_kg"] > 0.0
    # Altitude and semi-major axis should return very close to baseline nominal at same epoch
    assert mission["semi_major_axis_residual_m"] < 10.0   # Residual within 10 meters
    assert mission["altitude_residual_error_m"] < 100.0   # Radial residual within 100 meters
    assert mission["velocity_residual_m_s"] < 0.1         # Velocity within 0.1 m/s


def test_tradeoff_frontier_generation():
    """Verify Pareto trade-off curve across multiple lead times."""
    planner = LowThrustAvoidancePlanner()
    frontier = planner.generate_tradeoff_frontier(
        current_altitude_km=550.0,
        target_clearance_km=5.0,
        lead_times_hours=[6.0, 12.0, 24.0, 48.0],
    )

    assert len(frontier) == 4
    # Earlier lead time must require strictly less burn time and fuel
    for i in range(len(frontier) - 1):
        earlier = frontier[i]
        later = frontier[i + 1]
        assert earlier["lead_time_hours"] < later["lead_time_hours"]
        assert earlier["burn_duration_seconds"] > later["burn_duration_seconds"]
        assert earlier["propellant_used_grams"] > later["propellant_used_grams"]
