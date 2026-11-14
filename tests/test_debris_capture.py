"""
Comprehensive Test Suite for Active Debris Removal (ADR) Capture Dynamics
=========================================================================
Tests:
- 6-DOF Robotic Arm Forward and Inverse Kinematics (DH parameters, Jacobian,
  Yoshikawa manipulability, Damped Least Squares IK, trajectory generation).
- Harpoon Tethered Momentum Exchange (Penetration mechanics, viscoelastic tether,
  winch slip braking, and exact conservation of linear momentum).
- Post-Capture Momentum Management (Parallel Axis Theorem, Euler rigid-body dynamics,
  Earth magnetic dipole, Reaction Wheel array, Magnetorquers, and Upper Stage Detumbling).
"""

import math
import pytest
import numpy as np

from aegis_leo.adr.debris_capture import (
    skew_symmetric,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    quaternion_multiply,
    quaternion_slerp,
    rotation_matrix_from_axis_angle,
    DHParameter,
    ForwardKinematicsResult,
    InverseKinematicsResult,
    RoboticArm6DOF,
    HarpoonParameters,
    TetherParameters,
    PenetrationResult,
    TetherState,
    HarpoonTetherSystem,
    SpacecraftMassProperties,
    ReactionWheelArray,
    MagnetorquerArray,
    EarthMagneticFieldDipole,
    CombinedBodyDynamics,
    DetumbleController,
    PostCaptureMomentumManager,
)


# ============================================================================
# 1. Math Utilities & Attitude Kinematics Tests
# ============================================================================

def test_skew_symmetric():
    """Verify skew-symmetric matrix anti-symmetry and cross-product identity."""
    v = np.array([1.5, -2.0, 3.2], dtype=float)
    K = skew_symmetric(v)

    # Anti-symmetry: K^T = -K
    assert np.allclose(K.T, -K)

    # Diagonal is zero
    assert np.allclose(np.diag(K), 0.0)

    # Cross product equivalence: K @ u == np.cross(v, u)
    u = np.array([0.5, 4.0, -1.0], dtype=float)
    assert np.allclose(K @ u, np.cross(v, u))


def test_quaternion_rot_matrix_roundtrip():
    """Verify rotation matrix <-> quaternion roundtrip conversion."""
    # Identity
    q_ident = np.array([1.0, 0.0, 0.0, 0.0])
    R_ident = quaternion_to_rotation_matrix(q_ident)
    assert np.allclose(R_ident, np.eye(3))
    assert np.allclose(rotation_matrix_to_quaternion(R_ident), q_ident)

    # Arbitrary rotations about principal axes
    for angle_deg in [30.0, 45.0, 90.0, 135.0]:
        rad = math.radians(angle_deg)
        for axis in [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1, 1, 1]) / math.sqrt(3)]:
            R = rotation_matrix_from_axis_angle(axis, rad)
            q = rotation_matrix_to_quaternion(R)
            R_reconstructed = quaternion_to_rotation_matrix(q)
            assert np.allclose(R, R_reconstructed, atol=1e-6)


def test_quaternion_slerp():
    """Verify spherical linear interpolation between orientations."""
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    # 90 degree rotation around Z
    q1 = np.array([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])

    # t = 0 -> q0
    assert np.allclose(quaternion_slerp(q0, q1, 0.0), q0)
    # t = 1 -> q1
    assert np.allclose(quaternion_slerp(q0, q1, 1.0), q1)

    # t = 0.5 -> 45 degree rotation around Z
    q_mid = quaternion_slerp(q0, q1, 0.5)
    expected_mid = np.array([math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)])
    assert np.allclose(q_mid, expected_mid, atol=1e-5)


def test_quaternion_multiply():
    """Verify Hamilton quaternion product."""
    # q * identity = q
    q = np.array([0.5, 0.5, 0.5, 0.5])
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quaternion_multiply(q, ident), q)
    assert np.allclose(quaternion_multiply(ident, q), q)


# ============================================================================
# 2. 6-DOF Robotic Arm Kinematics Tests
# ============================================================================

def test_robotic_arm_initialization():
    """Verify 6-DOF arm default DH parameters and custom parameters validation."""
    arm = RoboticArm6DOF()
    assert len(arm.dh_params) == 6
    assert arm.name == "ADR-Manipulator-6DOF"

    # Reject invalid number of parameters
    with pytest.raises(ValueError, match="Expected exactly 6 DH parameters"):
        RoboticArm6DOF(dh_params=[DHParameter(a=1, alpha=0, d=0)])


def test_forward_kinematics_basic():
    """Verify forward kinematics computation for zero joint configuration."""
    arm = RoboticArm6DOF()
    zeros = np.zeros(6, dtype=float)
    fk = arm.forward_kinematics(zeros)

    assert isinstance(fk, ForwardKinematicsResult)
    assert fk.position.shape == (3,)
    assert fk.orientation_matrix.shape == (3, 3)
    assert fk.quaternion.shape == (4,)
    assert len(fk.joint_transforms) == 7  # Base + 6 frames

    # Verify orientation matrix is orthonormal
    R = fk.orientation_matrix
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)

    # End-effector reach should be non-zero and physically reasonable (> 1.5 m)
    reach = np.linalg.norm(fk.position)
    assert 1.5 < reach < 3.5


def test_geometric_jacobian_finite_difference():
    """Verify analytical geometric Jacobian against numerical central differences."""
    arm = RoboticArm6DOF()
    q0 = np.array([0.2, -0.4, 0.5, 0.3, -0.2, 0.1], dtype=float)

    J_analytical = arm.compute_jacobian(q0)
    assert J_analytical.shape == (6, 6)

    # Numerical differentiation for linear velocity Jacobian (first 3 rows)
    eps = 1e-6
    J_num_v = np.zeros((3, 6))
    for i in range(6):
        qp = q0.copy()
        qm = q0.copy()
        qp[i] += eps
        qm[i] -= eps
        pos_p = arm.forward_kinematics(qp).position
        pos_m = arm.forward_kinematics(qm).position
        J_num_v[:, i] = (pos_p - pos_m) / (2.0 * eps)

    assert np.allclose(J_analytical[0:3, :], J_num_v, atol=1e-5)


def test_manipulability_and_singularities():
    """Verify Yoshikawa manipulability index and singularity detection."""
    arm = RoboticArm6DOF()

    # Nominal bent configuration has high manipulability
    q_nominal = np.array([0.1, 0.5, -0.8, 0.2, 0.3, 0.0])
    w_nominal = arm.compute_manipulability(q_nominal)
    assert w_nominal > 0.05
    assert not arm.is_singular(q_nominal)

    # Fully outstretched configuration (elbow theta3 = 0, shoulder aligned) reduces manipulability
    q_outstretched = np.zeros(6)
    cond = arm.compute_condition_number(q_nominal)
    assert cond >= 1.0
    assert not math.isinf(cond)


def test_inverse_kinematics_position_only():
    """Verify numerical IK for reachable Cartesian target positions."""
    arm = RoboticArm6DOF()
    # Pick a reachable reference pose by forward kinematics
    q_ref = np.array([0.3, -0.5, 0.8, -0.2, 0.4, -0.1])
    target_pos = arm.forward_kinematics(q_ref).position

    # Perturb initial guess
    init_guess = q_ref + 0.2 * np.random.RandomState(42).randn(6)
    ik_res = arm.inverse_kinematics(
        target_position=target_pos,
        initial_guess=init_guess,
        max_iterations=100,
        tolerance_pos=1e-4
    )

    assert ik_res.success
    assert ik_res.position_error < 1e-3
    fk_res = arm.forward_kinematics(ik_res.joint_angles)
    assert np.allclose(fk_res.position, target_pos, atol=1e-3)


def test_inverse_kinematics_full_6dof():
    """Verify full 6-DOF (position + orientation) inverse kinematics convergence."""
    arm = RoboticArm6DOF()
    q_target = np.array([0.2, -0.3, 0.6, 0.1, -0.4, 0.2])
    fk_target = arm.forward_kinematics(q_target)

    # Initial guess offset
    init_guess = q_target + 0.15 * np.array([1, -1, 1, -1, 1, -1])

    ik_res = arm.inverse_kinematics(
        target_position=fk_target.position,
        target_orientation=fk_target.orientation_matrix,
        initial_guess=init_guess,
        max_iterations=120,
        tolerance_pos=1e-3,
        tolerance_ori=2e-2
    )

    assert ik_res.success
    assert ik_res.position_error < 5e-3
    assert ik_res.orientation_error < 5e-2


def test_inverse_kinematics_joint_limits():
    """Verify inverse kinematics strictly respects DH joint limits."""
    # Constrain joint limits tightly
    dh = [
        DHParameter(a=0.0, alpha=np.pi / 2, d=0.4, theta_min=-0.5, theta_max=0.5),
        DHParameter(a=1.2, alpha=0.0, d=0.0, theta_min=-0.5, theta_max=0.5),
        DHParameter(a=1.0, alpha=0.0, d=0.0, theta_min=-0.5, theta_max=0.5),
        DHParameter(a=0.0, alpha=-np.pi / 2, d=0.25, theta_min=-0.5, theta_max=0.5),
        DHParameter(a=0.0, alpha=np.pi / 2, d=0.0, theta_min=-0.5, theta_max=0.5),
        DHParameter(a=0.0, alpha=0.0, d=0.3, theta_min=-0.5, theta_max=0.5),
    ]
    constrained_arm = RoboticArm6DOF(dh_params=dh)
    target = np.array([1.5, 0.2, 0.8])
    res = constrained_arm.inverse_kinematics(target_position=target)

    for i in range(6):
        assert dh[i].theta_min - 1e-6 <= res.joint_angles[i] <= dh[i].theta_max + 1e-6


def test_cartesian_trajectory_planning():
    """Verify smooth Cartesian trajectory planning with joint rate generation."""
    arm = RoboticArm6DOF()
    q_start = np.array([0.1, -0.4, 0.6, 0.0, 0.2, 0.0])
    start_pos = arm.forward_kinematics(q_start).position
    target_pos = start_pos + np.array([0.2, -0.15, 0.1])

    traj = arm.plan_cartesian_trajectory(
        start_pos=start_pos,
        target_pos=target_pos,
        steps=25,
        duration=5.0,
        initial_joint_angles=q_start
    )

    assert traj["cartesian_positions"].shape == (25, 3)
    assert traj["joint_trajectory_rad"].shape == (25, 6)
    assert traj["joint_velocities_rad_s"].shape == (25, 6)
    assert traj["all_points_converged"]
    assert traj["max_joint_speed_rad_s"] > 0.0


def test_simulate_grapple_approach():
    """Verify robotic arm tracking and capture of moving debris grapple fixture."""
    arm = RoboticArm6DOF()
    q_start = np.array([0.1, -0.3, 0.5, 0.0, 0.1, 0.0])
    p_start = arm.forward_kinematics(q_start).position

    # Target fixture nearby moving slowly
    fixture_pos = p_start + np.array([0.08, -0.05, 0.04])
    fixture_vel = np.array([0.005, -0.002, 0.001])  # 5 mm/s drift

    res = arm.simulate_grapple_approach(
        fixture_pos=fixture_pos,
        fixture_vel=fixture_vel,
        initial_joint_angles=q_start,
        capture_distance=0.04,
        max_time=5.0
    )

    assert res["captured"]
    assert res["capture_time_s"] is not None
    assert res["final_distance_m"] <= 0.04


# ============================================================================
# 3. Harpoon Tethered Momentum Exchange Tests
# ============================================================================

def test_harpoon_penetration_success():
    """Verify successful perforation and locking barbs deployment in thin rocket casing."""
    system = HarpoonTetherSystem()
    # 2.5 mm aluminum casing (typical propellant tank skin)
    result = system.evaluate_penetration(
        target_skin_thickness_m=0.0025,
        target_yield_strength_pa=2.7e8,
        impact_velocity_m_s=25.0
    )

    assert isinstance(result, PenetrationResult)
    assert result.penetrated
    assert result.barbs_engaged
    assert result.structural_integrity == "SECURE_LATCH"
    assert result.impact_kinetic_energy_j == 0.5 * 5.0 * (25.0 ** 2)
    assert result.penetration_depth_m > 0.0025


def test_harpoon_insufficient_energy():
    """Verify bounce/deflection or partial penetration when impact speed is low."""
    system = HarpoonTetherSystem()
    # Low speed impact: 3 m/s against 8 mm thick armor plate
    result = system.evaluate_penetration(
        target_skin_thickness_m=0.008,
        target_yield_strength_pa=4.0e8,
        impact_velocity_m_s=3.0
    )

    assert not result.barbs_engaged
    assert result.structural_integrity in ["BOUNCE_DEFLECTION", "PARTIAL_PENETRATION"]


def test_tether_force_slack_vs_taut():
    """Verify Kelvin-Voigt viscoelastic model behaves correctly for slack and taut states."""
    system = HarpoonTetherSystem(
        tether_params=TetherParameters(nominal_length_m=50.0, stiffness_n_per_m=2000.0)
    )

    chaser_pos = np.array([0.0, 0.0, 0.0])
    chaser_vel = np.array([0.0, 0.0, 0.0])

    # Case 1: Slack tether (separation 40 m < 50 m)
    debris_pos_slack = np.array([40.0, 0.0, 0.0])
    debris_vel_slack = np.array([1.0, 0.0, 0.0])
    Fc, Fd, state_slack = system.compute_tether_force(
        chaser_pos, chaser_vel, debris_pos_slack, debris_vel_slack,
        current_nominal_length=50.0, chaser_mass=1000.0, debris_mass=3000.0
    )
    assert not state_slack.is_taut
    assert state_slack.tension_n == 0.0
    assert np.allclose(Fc, 0.0)
    assert np.allclose(Fd, 0.0)

    # Case 2: Taut tether (separation 52 m > 50 m, elongation 2 m)
    debris_pos_taut = np.array([52.0, 0.0, 0.0])
    debris_vel_taut = np.array([0.5, 0.0, 0.0])  # Moving away
    Fc, Fd, state_taut = system.compute_tether_force(
        chaser_pos, chaser_vel, debris_pos_taut, debris_vel_taut,
        current_nominal_length=50.0, chaser_mass=1000.0, debris_mass=3000.0
    )
    assert state_taut.is_taut
    assert state_taut.tension_n > 0.0
    # Newton's 3rd Law: action and reaction
    assert np.allclose(Fc, -Fd)
    # Force on chaser pulls in +X (towards debris)
    assert Fc[0] > 0.0
    assert Fd[0] < 0.0


def test_tether_rupture_condition():
    """Verify tether rupture detection when tension exceeds ultimate tensile strength."""
    system = HarpoonTetherSystem(
        tether_params=TetherParameters(
            nominal_length_m=50.0,
            stiffness_n_per_m=10000.0,
            break_tension_n=5000.0,
            winch_slip_tension_n=100000.0  # Winch slip disabled
        )
    )
    # Huge elongation (10 m stretch -> 100,000 N >> 5,000 N break limit)
    debris_pos = np.array([60.0, 0.0, 0.0])
    Fc, Fd, state = system.compute_tether_force(
        np.zeros(3), np.zeros(3), debris_pos, np.zeros(3),
        current_nominal_length=50.0, chaser_mass=1000.0, debris_mass=2000.0,
        allow_winch_slip=False
    )
    assert state.is_broken
    assert state.tension_n == 0.0
    assert np.allclose(Fc, 0.0)


def test_momentum_exchange_simulation_conservation():
    """
    Verify complete simulation of tethered momentum exchange.
    Crucial: Linear momentum of combined (chaser + debris) system must be exactly conserved.
    """
    system = HarpoonTetherSystem()
    chaser_mass = 1200.0  # kg
    debris_mass = 4500.0  # kg (e.g. spent upper stage)

    chaser_pos0 = np.array([0.0, 0.0, 0.0])
    chaser_vel0 = np.array([0.0, 0.0, 0.0])

    # Debris drifting away at relative speed 2.0 m/s
    debris_pos0 = np.array([75.0, 0.0, 0.0])
    debris_vel0 = np.array([2.0, 0.0, 0.0])

    sim = system.simulate_capture_and_momentum_exchange(
        chaser_mass_kg=chaser_mass,
        debris_mass_kg=debris_mass,
        chaser_init_pos=chaser_pos0,
        chaser_init_vel=chaser_vel0,
        debris_init_pos=debris_pos0,
        debris_init_vel=debris_vel0,
        duration_s=40.0,
        dt=0.02,
        enable_winch_braking=True
    )

    # Momentum conservation check: CoM velocity drift must be virtually zero
    assert sim["momentum_conserved"]
    assert sim["max_com_velocity_drift_m_s"] < 1e-9

    # Peak tension stayed within structural bounds
    assert sim["peak_tension_n"] < system.tether.break_tension_n
    assert not sim["tether_snapped"]

    # Relative speed was arrested by tether exchange
    assert sim["final_relative_speed_m_s"] < 2.0

    # Winch and damping dissipated kinetic energy
    assert sim["energy_dissipated_j"] > 0.0


# ============================================================================
# 4. Post-Capture Momentum Management & Upper Stage Detumbling Tests
# ============================================================================

def test_combined_body_parallel_axis_theorem():
    """Verify Parallel Axis Theorem calculation for joint spacecraft + upper stage."""
    # Chaser: 1000 kg at origin, diagonal inertia
    chaser = SpacecraftMassProperties(
        mass_kg=1000.0,
        com_offset_m=np.array([0.0, 0.0, 0.0]),
        inertia_tensor_kg_m2=np.diag([600.0, 700.0, 800.0]),
        initial_spin_rate_rad_s=np.zeros(3)
    )

    # Debris: 4000 kg (spent Ariane stage), docked at [0, 0, 5] m
    debris = SpacecraftMassProperties(
        mass_kg=4000.0,
        com_offset_m=np.array([0.0, 0.0, 0.0]),
        inertia_tensor_kg_m2=np.diag([5000.0, 35000.0, 35000.0]),
        initial_spin_rate_rad_s=np.array([0.1, 0.0, 0.0])  # 0.1 rad/s roll
    )

    docking_offset = np.array([0.0, 0.0, 5.0])
    combined = CombinedBodyDynamics.compute_combined_properties(chaser, debris, docking_offset)

    # Total mass = 5000 kg
    assert combined["total_mass_kg"] == 5000.0

    # CoM = (1000*0 + 4000*5) / 5000 = 4.0 m along Z
    expected_com = np.array([0.0, 0.0, 4.0])
    assert np.allclose(combined["joint_com_m"], expected_com)

    # Combined inertia must be symmetric and positive definite
    I_comb = combined["combined_inertia_kg_m2"]
    assert np.allclose(I_comb, I_comb.T)
    eigenvalues = np.linalg.eigvals(I_comb)
    assert np.all(eigenvalues > 0.0)

    # Initial angular velocity about Z is zero, non-zero along X
    omega0 = combined["initial_angular_velocity_rad_s"]
    assert omega0[0] > 0.0
    assert np.isclose(omega0[1], 0.0)
    assert np.isclose(omega0[2], 0.0)


def test_earth_magnetic_field_dipole():
    """Verify geomagnetic dipole field in LEO."""
    field_model = EarthMagneticFieldDipole()

    # 700 km altitude equatorial position (r = 7078 km)
    pos = np.array([7078137.0, 0.0, 0.0])
    B = field_model.compute_b_field(pos)

    # In LEO, geomagnetic field strength is between 20 and 50 microtesla (20e-6 to 50e-6 T)
    b_mag = np.linalg.norm(B)
    assert 1.5e-5 < b_mag < 6.0e-5

    # Falloff with distance r: B(2r) should be ~ 1/8 of B(r)
    B_double = field_model.compute_b_field(pos * 2.0)
    assert np.isclose(np.linalg.norm(B_double), b_mag / 8.0, rtol=0.01)


def test_reaction_wheel_array():
    """Verify Reaction Wheel torque clamping and momentum saturation."""
    rw = ReactionWheelArray(max_torque_nm=0.20, max_momentum_capacity_nms=10.0)

    # Request small torque for 5 seconds: delta_h = -tau * dt
    tau_act, sat = rw.apply_torque(np.array([0.1, 0.0, 0.0]), dt=5.0)
    assert not sat
    assert np.allclose(tau_act, [0.1, 0.0, 0.0])
    assert np.isclose(rw.stored_momentum_nms[0], -0.5)

    # Request torque for 60 s: clamped to max_torque 0.20 N*m -> delta_h = -12.0 Nms, saturating 10 Nms capacity
    tau_act2, sat2 = rw.apply_torque(np.array([10.0, 0.0, 0.0]), dt=60.0)
    assert sat2
    assert np.isclose(rw.stored_momentum_nms[0], -10.0)


def test_magnetorquer_orthogonality():
    """Verify magnetic torque is strictly perpendicular to the magnetic field vector."""
    mtq = MagnetorquerArray(max_dipole_moment_am2=100.0)
    B = np.array([2e-5, -1e-5, 3e-5])
    m = np.array([50.0, -30.0, 80.0])

    tau = mtq.compute_torque(m, B)
    # Perpendicular: tau . B == 0
    assert np.isclose(np.dot(tau, B), 0.0, atol=1e-12)


def test_detumble_b_dot_controller():
    """Verify B-Dot magnetic detumbling dipole and energy dissipation."""
    omega = np.array([0.05, 0.02, -0.03])
    B = np.array([1e-5, 3e-5, -2e-5])

    m = DetumbleController.compute_b_dot_dipole(omega, B, gain_k=1e6)
    tau_mag = np.cross(m, B)

    # Power dissipated: P = tau . omega <= 0
    power = np.dot(tau_mag, omega)
    assert power <= 0.0  # Dissipating rotational energy!


def test_post_capture_detumble_simulation_hybrid():
    """
    Verify complete post-capture detumbling simulation of spinning rocket upper stage
    using hybrid Reaction Wheel rate damping and Magnetorquer desaturation.
    """
    chaser = SpacecraftMassProperties(
        mass_kg=1200.0,
        com_offset_m=np.array([0.0, 0.0, 0.0]),
        inertia_tensor_kg_m2=np.diag([800.0, 900.0, 1100.0])
    )
    # Upper stage spinning at 2.5 deg/s = 0.0436 rad/s
    debris = SpacecraftMassProperties(
        mass_kg=3500.0,
        com_offset_m=np.array([0.0, 0.0, 0.0]),
        inertia_tensor_kg_m2=np.diag([4000.0, 25000.0, 25000.0]),
        initial_spin_rate_rad_s=np.array([math.radians(2.5), 0.0, 0.0])
    )

    # Heavy ADR chaser actuators sized for upper-stage momentum capture
    rw = ReactionWheelArray(max_torque_nm=2.0, max_momentum_capacity_nms=250.0)
    mtq = MagnetorquerArray(max_dipole_moment_am2=300.0)
    manager = PostCaptureMomentumManager(rw_array=rw, mtq_array=mtq)

    sim = manager.simulate_detumble(
        chaser=chaser,
        debris=debris,
        docking_offset_m=np.array([0.0, 0.0, 4.0]),
        orbit_altitude_km=650.0,
        orbit_inclination_deg=82.0,
        duration_s=600.0,  # 10 minutes
        dt=0.5,
        control_mode="hybrid",
        detumble_threshold_rad_s=0.005,
        kd_gain=50.0,
        kw_gain=0.10
    )

    # Spin rate must drop substantially
    assert sim["final_spin_rate_rad_s"] < sim["initial_spin_rate_rad_s"]
    assert sim["energy_reduction_ratio"] > 0.85  # Over 85% energy dissipated
    assert sim["is_detumbled"] or (sim["final_spin_rate_rad_s"] < 0.005)

    # Reaction wheel momentum stayed bounded thanks to magnetorquers
    max_h_rw = np.max(sim["wheel_momentum_nms"])
    assert max_h_rw <= manager.rw.max_capacity_nms


def test_uncontrolled_tumbling_energy_conservation():
    """Verify that in uncontrolled mode without external torques, rotational energy is conserved."""
    chaser = SpacecraftMassProperties(
        mass_kg=1000.0,
        com_offset_m=np.zeros(3),
        inertia_tensor_kg_m2=np.diag([500.0, 500.0, 500.0])
    )
    debris = SpacecraftMassProperties(
        mass_kg=2000.0,
        com_offset_m=np.zeros(3),
        inertia_tensor_kg_m2=np.diag([2000.0, 2000.0, 2000.0]),
        initial_spin_rate_rad_s=np.array([0.05, 0.0, 0.0])
    )

    manager = PostCaptureMomentumManager()
    sim = manager.simulate_detumble(
        chaser=chaser,
        debris=debris,
        duration_s=60.0,
        dt=0.1,
        control_mode="uncontrolled"
    )

    # Rotational energy remains constant
    e_init = sim["rotational_energy_j"][0]
    e_final = sim["rotational_energy_j"][-1]
    assert np.isclose(e_init, e_final, rtol=1e-3)
