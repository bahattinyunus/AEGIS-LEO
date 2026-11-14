"""
Unit Tests for Model Predictive Control (MPC) & Sequential Convex Programming (SCP)
Non-Convex Orbital Collision Avoidance Maneuver Planner

Validates:
1. Spacecraft thruster saturation, slew rate kinematics, and Tsiolkovsky propellant depletion.
2. 3D non-convex avoidance ellipsoids, Mahalanobis distances, and separating hyperplanes.
3. Hill-Clohessy-Wiltshire (HCW) relative orbital dynamics and exact discrete transition matrices.
4. Pure NumPy Infeasible-Start Primal-Dual Interior-Point (PDIP) QP solver convergence and precision.
5. Sequential Convex Programming (SCP) convergence on single- and multi-obstacle collision avoidance.
6. Thruster saturation and slew rate constraint compliance.
7. Receding-Horizon MPC closed-loop simulation with warm starting.
8. Pareto trade-off curve generation across safety margin factors.
"""

import math
import pytest
import numpy as np

from aegis_leo.constants import EARTH_RADIUS_KM, EARTH_MU, STANDARD_GRAVITY
from aegis_leo.ai.mpc_avoidance import (
    SpacecraftThrusterLimits,
    AvoidanceEllipsoid,
    ClohessyWiltshireDynamics,
    PrimalDualQPSolver,
    QPSolution,
    SCPAvoidancePlanner,
    SCPAvoidanceResult,
    MPCAvoidanceController,
    MPCSimulationResult,
    generate_avoidance_tradeoff,
)


def test_spacecraft_thruster_limits():
    """Verify physical thruster limits, kinematic slew limits, and propellant depletion."""
    limits = SpacecraftThrusterLimits(
        mass_kg=250.0,
        max_thrust_n=5.0,
        max_slew_rate_deg_s=3.0,
        delta_v_budget_m_s=30.0,
        isp_s=300.0,
    )
    # Derived acceleration bound: 5.0 N / 250 kg = 0.02 m/s^2
    assert pytest.approx(limits.max_accel_m_s2, rel=1e-5) == 0.02
    assert limits.effective_exhaust_velocity_m_s == pytest.approx(300.0 * STANDARD_GRAVITY, rel=1e-5)

    # Tsiolkovsky propellant consumption
    # Delta_m = m_wet * (1 - exp(-Delta_v / c))
    dv = 10.0
    c = 300.0 * STANDARD_GRAVITY
    expected_prop = 250.0 * (1.0 - math.exp(-dv / c))
    assert pytest.approx(limits.compute_propellant_used(dv), rel=1e-5) == expected_prop
    assert limits.compute_propellant_used(0.0) == 0.0

    # Slew limit per 10s step
    dt = 10.0
    slew_step = limits.slew_limit_per_step(dt)
    assert slew_step > 0.0
    assert slew_step == pytest.approx(limits.max_jerk_m_s3 * dt, rel=1e-5)


def test_avoidance_ellipsoid_geometry_and_linearization():
    """Verify 3D avoidance ellipsoid distance metrics and separating hyperplane."""
    center = np.array([100.0, 200.0, -50.0])
    semi_axes = np.array([40.0, 100.0, 30.0])
    ellipsoid = AvoidanceEllipsoid(name="Target Debris", center=center, semi_axes=semi_axes)

    # 1. Center of ellipsoid: distance is 0, contains is True
    assert ellipsoid.normalized_distance(center) == 0.0
    assert ellipsoid.contains(center) is True
    assert ellipsoid.clearance_margin(center) == -1.0

    # 2. Point on boundary along radial axis: [100 + 40, 200, -50]
    boundary_pt = center + np.array([40.0, 0.0, 0.0])
    assert pytest.approx(ellipsoid.normalized_distance(boundary_pt), rel=1e-5) == 1.0
    assert ellipsoid.contains(boundary_pt, margin_factor=1.0) is False
    assert pytest.approx(ellipsoid.clearance_margin(boundary_pt), abs=1e-5) == 0.0

    # 3. Safe exterior point: [100 + 80, 200, -50] (d_E = 2.0)
    ext_pt = center + np.array([80.0, 0.0, 0.0])
    assert pytest.approx(ellipsoid.normalized_distance(ext_pt), rel=1e-5) == 2.0
    assert ellipsoid.contains(ext_pt) is False
    assert pytest.approx(ellipsoid.clearance_margin(ext_pt), rel=1e-5) == 1.0

    # 4. Separating hyperplane linearization
    n_vec, offset = ellipsoid.linearize_separating_hyperplane(ext_pt, margin_factor=1.0)
    # The reference point must satisfy the halfspace constraint: n^T ext_pt >= offset
    val_ref = float(n_vec @ ext_pt)
    assert val_ref >= offset - 1e-6

    # The interior center must strictly violate the halfspace: n^T center < offset
    val_center = float(n_vec @ center)
    assert val_center < offset

    # Test rotated ellipsoid
    # 90 deg rotation around z-axis
    R = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    rotated_ellipsoid = AvoidanceEllipsoid(center=center, semi_axes=semi_axes, rotation=R)
    assert rotated_ellipsoid.normalized_distance(center) == 0.0
    # Along y now has semi_axis 40
    rotated_bound = center + np.array([0.0, 40.0, 0.0])
    assert pytest.approx(rotated_ellipsoid.normalized_distance(rotated_bound), rel=1e-5) == 1.0


def test_clohessy_wiltshire_dynamics():
    """Verify HCW relative orbit mechanics, mean motion, and discrete matrices."""
    alt_km = 500.0
    cw = ClohessyWiltshireDynamics(orbit_altitude_km=alt_km)

    # 1. Mean motion for 500 km LEO
    r_orbit_m = (EARTH_RADIUS_KM + alt_km) * 1000.0
    expected_n = math.sqrt(EARTH_MU * 1e9 / (r_orbit_m ** 3))
    assert pytest.approx(cw.mean_motion, rel=1e-6) == expected_n
    assert 5400.0 < cw.orbital_period_s < 6000.0
    assert 7500.0 < cw.orbital_velocity_m_s < 7800.0

    # 2. Continuous matrices
    A, B = cw.continuous_matrices()
    assert A.shape == (6, 6)
    assert B.shape == (6, 3)
    assert A[0, 3] == 1.0
    assert A[3, 0] == pytest.approx(3.0 * (expected_n ** 2), rel=1e-6)

    # 3. Discrete matrices: Phi(dt) and Bd(dt)
    dt = 10.0
    Phi, Bd = cw.discrete_matrices(dt)
    assert Phi.shape == (6, 6)
    assert Bd.shape == (6, 3)

    # Identity limit: Phi(0) = I_6, Bd(0) = 0
    Phi_0, Bd_0 = cw.discrete_matrices(0.0)
    assert np.allclose(Phi_0, np.eye(6), atol=1e-9)
    assert np.allclose(Bd_0, np.zeros((6, 3)), atol=1e-9)

    # Group property: Phi(dt1) @ Phi(dt2) == Phi(dt1 + dt2)
    dt1, dt2 = 7.0, 13.0
    Phi1, _ = cw.discrete_matrices(dt1)
    Phi2, _ = cw.discrete_matrices(dt2)
    Phi_sum, _ = cw.discrete_matrices(dt1 + dt2)
    assert np.allclose(Phi1 @ Phi2, Phi_sum, atol=1e-8)

    # Step propagation and rollout
    x0 = np.array([10.0, -50.0, 5.0, 0.1, -0.2, 0.0])
    u0 = np.array([0.01, -0.01, 0.0])
    x1 = cw.propagate_step(x0, u0, dt)
    assert x1.shape == (6,)

    controls = np.tile(u0, (5, 1))
    roll = cw.rollout(x0, controls, dt)
    assert roll.shape == (6, 6)
    assert np.allclose(roll[0], x0)
    assert np.allclose(roll[1], x1)


def test_primal_dual_qp_solver():
    """Verify pure NumPy Infeasible-Start Primal-Dual Interior-Point QP solver."""
    # Problem:
    # min  0.5 * (z1^2 + z2^2) - 2 z1 - 3 z2
    # s.t. z1 + z2 <= 1
    #      -z1 <= 0
    #      -z2 <= 0
    # Unconstrained minimum is at (2, 3), sum = 5 > 1, so constraint is active.
    # Analytical constrained minimum via KKT:
    # z1* = 0.5, z2* = 0.5 (or by Lagrange multipliers: z1 = 0, z2 = 1? Let's check:
    # grad f = [z1 - 2, z2 - 3] + lam * [1, 1] = 0 -> z1 = 2 - lam, z2 = 3 - lam
    # z1 + z2 = 5 - 2 lam = 1 -> lam = 2 -> z1 = 0, z2 = 1.
    H = np.eye(2)
    q = np.array([-2.0, -3.0])
    A = np.array([
        [1.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0]
    ])
    b = np.array([1.0, 0.0, 0.0])

    sol = PrimalDualQPSolver.solve(H, q, A, b, max_iter=30, tol=1e-8)
    assert sol.converged is True
    assert sol.iterations < 25
    assert sol.primal_residual < 1e-5
    assert sol.dual_residual < 1e-5

    # Check analytical solution: z1 = 0.0, z2 = 1.0
    assert pytest.approx(sol.z[0], abs=1e-3) == 0.0
    assert pytest.approx(sol.z[1], abs=1e-3) == 1.0
    # Check all constraints satisfied
    assert np.all(A @ sol.z <= b + 1e-5)


def test_scp_avoidance_planner_single_obstacle():
    """
    Verify Sequential Convex Programming for a non-convex collision encounter.
    The unforced trajectory passes straight through the keep-out zone (collision).
    SCP must steer the spacecraft outside the avoidance ellipsoid while respecting
    thruster saturation and slew rate constraints.
    """
    cw = ClohessyWiltshireDynamics(orbit_altitude_km=500.0)
    limits = SpacecraftThrusterLimits(
        mass_kg=200.0,
        max_thrust_n=4.0,           # 20 mm/s^2 max accel
        max_slew_rate_deg_s=3.0,
        max_jerk_m_s3=0.005,        # 5 mm/s^3 jerk limit
    )

    N_horizon = 20
    dt = 10.0
    planner = SCPAvoidancePlanner(
        dynamics=cw,
        thruster_limits=limits,
        horizon_steps=N_horizon,
        dt_s=dt,
        fuel_weight=1.0,
        slew_weight=5.0,
        terminal_tracking_weight=0.2,
        max_scp_iterations=12,
    )

    # Chief relative state at origin
    x0 = np.zeros(6)

    # Debris approaches along in-track axis (y: -200 m to +200 m)
    # Closest encounter at t = 100s (step 10), passing at [10, 0, 5] m
    debris_traj = np.zeros((N_horizon + 1, 3))
    for k in range(N_horizon + 1):
        t = k * dt
        debris_traj[k] = np.array([10.0, -200.0 + 2.0 * t, 5.0])

    # Avoidance ellipsoid: rx = 40 m, ry = 80 m, rz = 40 m
    semi_axes = np.array([40.0, 80.0, 40.0])
    ellipsoid = AvoidanceEllipsoid(semi_axes=semi_axes)

    # 1. Verify unforced trajectory enters keep-out zone
    unforced_traj = cw.rollout(x0, np.zeros((N_horizon, 3)), dt)
    unforced_dE = [
        AvoidanceEllipsoid(center=debris_traj[k], semi_axes=semi_axes).normalized_distance(unforced_traj[k, :3])
        for k in range(N_horizon + 1)
    ]
    min_unforced_dE = min(unforced_dE)
    assert min_unforced_dE < 0.3, "Unforced trajectory should be a critical collision"

    # 2. Run SCP avoidance optimization
    result: SCPAvoidanceResult = planner.plan_avoidance(
        x0=x0,
        obstacle=ellipsoid,
        debris_trajectory=debris_traj,
        margin_factor=1.0,
    )

    # 3. Assertions on SCP results
    assert result.converged is True
    assert result.iterations <= 10
    # Minimum normalized distance must be strictly >= 1.0 (exterior to keep-out ellipsoid)
    assert result.min_normalized_distance >= 0.999
    # Delta-v must be positive, reasonable (< 15 m/s)
    assert 0.1 < result.delta_v_total_m_s < 15.0
    assert result.propellant_consumed_kg > 0.0

    # 4. Check thruster saturation: max component <= u_max
    u_max = limits.max_accel_m_s2
    assert result.max_control_magnitude <= u_max * math.sqrt(3) + 1e-4
    assert np.all(np.abs(result.controls) <= u_max + 1e-4)

    # 5. Check slew rate limits between consecutive controls
    slew_max = limits.slew_limit_per_step(dt)
    diffs = np.abs(np.diff(result.controls, axis=0))
    assert np.all(diffs <= slew_max + 1e-4)

    # 6. Check virtual control slacks were completely eliminated (zero violation)
    assert np.max(result.slack_violations) < 1e-3


def test_scp_avoidance_planner_multi_obstacle():
    """
    Verify SCP planner avoiding multiple simultaneous debris obstacles.
    """
    cw = ClohessyWiltshireDynamics(orbit_altitude_km=550.0)
    limits = SpacecraftThrusterLimits(mass_kg=220.0, max_thrust_n=6.0)
    N_horizon = 20
    dt = 10.0

    planner = SCPAvoidancePlanner(
        dynamics=cw,
        thruster_limits=limits,
        horizon_steps=N_horizon,
        dt_s=dt,
        max_scp_iterations=12,
    )

    x0 = np.zeros(6)

    # Two debris objects:
    # Debris 1 encounters at t = 70s, x = 12 m, y = 0
    # Debris 2 encounters at t = 140s, x = -15 m, y = 0, z = 10 m
    debris_traj1 = np.zeros((N_horizon + 1, 3))
    debris_traj2 = np.zeros((N_horizon + 1, 3))

    for k in range(N_horizon + 1):
        t = k * dt
        debris_traj1[k] = np.array([12.0, -140.0 + 2.0 * t, 0.0])
        debris_traj2[k] = np.array([-15.0, -280.0 + 2.0 * t, 10.0])

    obs1 = AvoidanceEllipsoid(name="Fragment Alpha", semi_axes=np.array([30.0, 60.0, 30.0]))
    obs2 = AvoidanceEllipsoid(name="Fragment Beta", semi_axes=np.array([30.0, 60.0, 30.0]))

    res = planner.plan_avoidance(
        x0=x0,
        obstacle=[obs1, obs2],
        debris_trajectory=[debris_traj1, debris_traj2],
        margin_factor=1.0,
    )

    assert res.converged is True
    assert res.min_normalized_distance >= 0.999
    assert res.delta_v_total_m_s > 0.0


def test_mpc_receding_horizon_controller():
    """
    Verify closed-loop receding-horizon MPC simulation over multi-step mission.
    """
    cw = ClohessyWiltshireDynamics(orbit_altitude_km=500.0)
    limits = SpacecraftThrusterLimits(mass_kg=200.0, max_thrust_n=5.0)

    controller = MPCAvoidanceController(
        dynamics=cw,
        thruster_limits=limits,
        horizon_steps=12,
        dt_s=10.0,
    )

    x0 = np.zeros(6)
    semi_axes = np.array([35.0, 70.0, 35.0])
    obstacle = AvoidanceEllipsoid(semi_axes=semi_axes)

    # Debris moves along in-track: y(t) = -180 + 2.0 * t (TCA at t = 90 s)
    def debris_pos(t: float) -> np.ndarray:
        return np.array([8.0, -180.0 + 2.0 * t, 4.0])

    # Run closed loop simulation for 180 seconds (18 steps)
    sim_res: MPCSimulationResult = controller.simulate_closed_loop(
        x0=x0,
        simulation_duration_s=180.0,
        obstacle=obstacle,
        debris_trajectory_generator=debris_pos,
    )

    assert sim_res.all_constraints_satisfied is True
    assert sim_res.min_normalized_distance >= 0.999
    assert sim_res.total_delta_v_m_s > 0.0
    assert len(sim_res.control_history) == 18
    assert len(sim_res.state_history) == 19


def test_avoidance_tradeoff_pareto_curve():
    """
    Verify Pareto trade-off generator varying safety margin factor vs Delta-v.
    """
    cw = ClohessyWiltshireDynamics(orbit_altitude_km=500.0)
    limits = SpacecraftThrusterLimits(mass_kg=250.0, max_thrust_n=5.0)
    N_horizon = 15
    dt = 10.0

    planner = SCPAvoidancePlanner(
        dynamics=cw,
        thruster_limits=limits,
        horizon_steps=N_horizon,
        dt_s=dt,
        max_scp_iterations=10,
    )

    x0 = np.zeros(6)
    debris_traj = np.zeros((N_horizon + 1, 3))
    for k in range(N_horizon + 1):
        t = k * dt
        debris_traj[k] = np.array([10.0, -150.0 + 2.0 * t, 0.0])

    obstacle = AvoidanceEllipsoid(semi_axes=np.array([30.0, 60.0, 30.0]))

    scales = [0.8, 1.0, 1.2]
    tradeoff = generate_avoidance_tradeoff(
        planner=planner,
        x0=x0,
        obstacle=obstacle,
        debris_trajectory=debris_traj,
        scale_factors=scales,
    )

    assert len(tradeoff) == 3
    # Check that larger safety margin generally requires larger or equal Delta-v
    dv_08 = tradeoff[0]["delta_v_m_s"]
    dv_10 = tradeoff[1]["delta_v_m_s"]
    dv_12 = tradeoff[2]["delta_v_m_s"]
    assert dv_08 <= dv_10 + 1e-3
    assert dv_10 <= dv_12 + 1e-3


def test_terminal_orbit_restoration():
    """
    Verify post-avoidance terminal tracking / orbit slot restoration.
    """
    cw = ClohessyWiltshireDynamics(orbit_altitude_km=500.0)
    limits = SpacecraftThrusterLimits(mass_kg=200.0, max_thrust_n=5.0)
    N_horizon = 25
    dt = 10.0

    planner = SCPAvoidancePlanner(
        dynamics=cw,
        thruster_limits=limits,
        horizon_steps=N_horizon,
        dt_s=dt,
        terminal_tracking_weight=1.0,
        max_scp_iterations=12,
    )

    x0 = np.zeros(6)
    # Debris passes early at t = 60s (step 6)
    debris_traj = np.zeros((N_horizon + 1, 3))
    for k in range(N_horizon + 1):
        t = k * dt
        debris_traj[k] = np.array([10.0, -120.0 + 2.0 * t, 5.0])

    obstacle = AvoidanceEllipsoid(semi_axes=np.array([30.0, 60.0, 30.0]))
    res = planner.plan_avoidance(
        x0=x0,
        obstacle=obstacle,
        debris_trajectory=debris_traj,
        x_target=np.zeros(6),
    )

    assert res.converged is True
    # At end of horizon (t = 250s, 190s after encounter), position should be restored near origin
    p_final = res.trajectory[-1, :3]
    final_pos_err = np.linalg.norm(p_final)
    # Restored to within modest stationkeeping corridor
    assert final_pos_err < 15.0
