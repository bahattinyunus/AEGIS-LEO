"""
Unit & Integration Tests for Clohessy-Wiltshire (HCW) & Yamanaka-Ankersen (YA) Relative Motion Engine
Verifies HCW analytical solutions, YA elliptical relative motion, corridor safety, and rendezvous targeting.
"""

import math
import numpy as np
import pytest

from aegis_leo.constants import EARTH_MU_SI, STANDARD_GRAVITY
from aegis_leo.astrodynamics.cw_proximity import (
    ClohessyWiltshirePropagator,
    YamanakaAnkersenPropagator,
    ApproachSafetyValidator,
    CorridorType,
    CorridorViolation,
    ApproachSafetyReport,
    RendezvousPlan,
    RelativeOrbitGeometry,
    mean_to_eccentric_anomaly,
    eccentric_to_true_anomaly,
    true_to_eccentric_anomaly,
    true_to_mean_anomaly,
    propagate_kepler_orbit,
    compute_hcw_continuous_matrix,
    compute_hcw_stm,
    propagate_hcw,
    compute_closed_orbit_velocity,
    compute_relative_orbit_geometry,
    compute_yamanaka_ankersen_psi,
    compute_pulsating_frame_transform,
    compute_yamanaka_ankersen_stm,
    propagate_yamanaka_ankersen,
    check_approach_corridor,
    validate_approach_trajectory,
    check_passive_collision_safety,
    plan_2_impulse_rendezvous,
    plan_4_impulse_rendezvous,
)


# ==============================================================================
# 1. KEPLERIAN ANOMALY SOLVERS TESTS
# ==============================================================================

def test_mean_to_eccentric_anomaly_circular_and_elliptic():
    """Verify Kepler equation solver across circular and eccentric orbits."""
    # Circular: M == E
    assert mean_to_eccentric_anomaly(1.234, 0.0) == pytest.approx(1.234, abs=1e-12)

    # Moderate eccentricity round-trip
    e = 0.25
    for m_deg in [0.0, 30.0, 90.0, 180.0, 270.0, 350.0]:
        m = math.radians(m_deg)
        e_anom = mean_to_eccentric_anomaly(m, e)
        # Check Kepler equation M = E - e*sin(E)
        m_reconstructed = (e_anom - e * math.sin(e_anom)) % (2.0 * math.pi)
        assert m_reconstructed == pytest.approx(m % (2.0 * math.pi), abs=1e-11)

    # High eccentricity (e = 0.85)
    e_high = 0.85
    m_high = 1.0
    e_sol = mean_to_eccentric_anomaly(m_high, e_high)
    assert (e_sol - e_high * math.sin(e_sol)) == pytest.approx(m_high, abs=1e-11)


def test_anomaly_conversions_roundtrip():
    """Verify roundtrip conversions between eccentric and true anomaly."""
    e = 0.15
    for nu_deg in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 315.0]:
        nu = math.radians(nu_deg)
        e_anom = true_to_eccentric_anomaly(nu, e)
        nu_recovered = eccentric_to_true_anomaly(e_anom, e)
        assert nu_recovered == pytest.approx(nu, abs=1e-11)

    # Mean to true roundtrip
    m0 = 1.5
    e_val = 0.08
    e_anom = mean_to_eccentric_anomaly(m0, e_val)
    nu = eccentric_to_true_anomaly(e_anom, e_val)
    m_back = true_to_mean_anomaly(nu, e_val)
    assert m_back == pytest.approx(m0, abs=1e-11)


def test_propagate_kepler_orbit_period():
    """Verify that Keplerian orbit propagation over 1 period returns to starting anomaly."""
    a = 7000e3
    e = 0.05
    nu0 = 0.5
    n = math.sqrt(EARTH_MU_SI / (a**3))
    period = 2.0 * math.pi / n

    nu_f, _, _ = propagate_kepler_orbit(a, e, nu0, period)
    assert nu_f == pytest.approx(nu0, abs=1e-10)


# ==============================================================================
# 2. CLOHESSY-WILTSHIRE (HCW) TESTS
# ==============================================================================

def test_hcw_continuous_matrix():
    """Verify HCW A matrix structure and eigenvalues."""
    a = 7000e3
    n = math.sqrt(EARTH_MU_SI / (a**3))
    a_mat = compute_hcw_continuous_matrix(n)

    assert a_mat.shape == (6, 6)
    # Trace of Hamiltonian relative motion matrix must be 0
    assert np.trace(a_mat) == pytest.approx(0.0, abs=1e-14)

    # Eigenvalues: 0, 0, +i*n, -i*n, +i*n, -i*n
    eigvals = np.linalg.eigvals(a_mat)
    imag_parts = np.sort(np.abs(np.imag(eigvals)))
    # Two zero eigenvalues
    assert np.min(np.abs(eigvals)) == pytest.approx(0.0, abs=1e-10)
    # Four eigenvalues with magnitude n
    assert imag_parts[-1] == pytest.approx(n, rel=1e-5)


def test_hcw_stm_properties():
    """Verify HCW State Transition Matrix identity, determinant, and group properties."""
    a = 7000e3
    n = math.sqrt(EARTH_MU_SI / (a**3))

    # 1. Identity at t = 0
    phi_0 = compute_hcw_stm(0.0, n)
    assert np.allclose(phi_0, np.eye(6), atol=1e-14)

    # 2. Determinant == 1 (volume preservation / Liouville's theorem)
    t = 1200.0
    phi_t = compute_hcw_stm(t, n)
    assert np.linalg.det(phi_t) == pytest.approx(1.0, abs=1e-10)

    # 3. Transition semigroup property: Phi(t1 + t2) == Phi(t2) @ Phi(t1)
    t1 = 300.0
    t2 = 450.0
    phi_1 = compute_hcw_stm(t1, n)
    phi_2 = compute_hcw_stm(t2, n)
    phi_12 = compute_hcw_stm(t1 + t2, n)
    assert np.allclose(phi_12, phi_2 @ phi_1, atol=1e-11)


def test_hcw_closed_orbit_condition():
    """Verify that ydot0 = -2*n*x0 yields a closed periodic 2:1 relative ellipse."""
    a = 7000e3
    n = math.sqrt(EARTH_MU_SI / (a**3))
    period = 2.0 * math.pi / n

    x0 = 100.0   # 100 m radial offset
    y0 = 50.0    # 50 m along-track offset
    z0 = 30.0    # 30 m cross-track
    xdot0 = 0.0
    ydot0 = compute_closed_orbit_velocity(x0, n)  # -2*n*x0
    zdot0 = 0.0

    state_0 = np.array([x0, y0, z0, xdot0, ydot0, zdot0])

    # Propagate over 1, 2, and 5 full orbits
    for num_orbits in [1, 2, 5]:
        t_span = np.array([0.0, num_orbits * period])
        traj = propagate_hcw(state_0, t_span, n)
        # Final state must exactly match initial state (zero drift)
        assert np.allclose(traj[0], traj[1], atol=1e-8)

    # Analyze geometry
    geom = compute_relative_orbit_geometry(state_0, n)
    assert geom.is_closed is True
    assert geom.secular_drift_rate_m_s == pytest.approx(0.0, abs=1e-12)
    # Aspect ratio of closed relative motion is exactly 2:1
    assert geom.semi_major_axis_m == pytest.approx(2.0 * geom.semi_minor_axis_m, rel=1e-6)
    assert geom.out_of_plane_amplitude_m == pytest.approx(30.0, abs=1e-10)


def test_hcw_secular_drift():
    """Verify non-closed relative orbit drifts along track as predicted by theory."""
    a = 7000e3
    n = math.sqrt(EARTH_MU_SI / (a**3))
    period = 2.0 * math.pi / n

    x0 = 50.0
    ydot0 = 0.0  # violates -2*n*x0 condition -> drift rate = -6*n*x0 = -300*n
    state_0 = np.array([x0, 0.0, 0.0, 0.0, ydot0, 0.0])

    geom = compute_relative_orbit_geometry(state_0, n)
    assert geom.is_closed is False
    expected_drift_rate = -6.0 * n * x0
    assert geom.secular_drift_rate_m_s == pytest.approx(expected_drift_rate, rel=1e-9)

    # After 1 orbit, along-track displacement is drift_rate * period = -6*n*x0 * (2*pi/n) = -12*pi*x0
    traj = propagate_hcw(state_0, np.array([0.0, period]), n)
    expected_delta_y = -12.0 * math.pi * x0
    actual_delta_y = traj[1, 1] - traj[0, 1]
    assert actual_delta_y == pytest.approx(expected_delta_y, rel=1e-6)


# ==============================================================================
# 3. YAMANAKA-ANKERSEN (YA) ELLIPTICAL RELATIVE MOTION TESTS
# ==============================================================================

def test_yamanaka_ankersen_reduces_to_hcw_at_zero_eccentricity():
    """Verify that Yamanaka-Ankersen STM converges to HCW STM when eccentricity e -> 0."""
    a = 7000e3
    n = math.sqrt(EARTH_MU_SI / (a**3))
    dt = 800.0
    nu0 = 0.0
    nu_f = n * dt

    # Classical HCW STM
    phi_hcw = compute_hcw_stm(dt, n)

    # Yamanaka-Ankersen STM with e = 0.0
    phi_ya = compute_yamanaka_ankersen_stm(
        true_anomaly_0=nu0,
        true_anomaly=nu_f,
        eccentricity=0.0,
        semi_major_axis_m=a,
        delta_time_s=dt,
        normalized=False,
    )

    # Must match to machine precision
    max_abs_diff = np.max(np.abs(phi_ya - phi_hcw))
    assert max_abs_diff < 1e-12


def test_yamanaka_ankersen_properties_eccentric():
    """Verify YA STM identity, group composition, and inverse on eccentric orbit (e = 0.2)."""
    a = 8000e3
    e = 0.2
    nu0 = 0.4
    dt1 = 500.0
    dt2 = 600.0

    nu1, _, _ = propagate_kepler_orbit(a, e, nu0, dt1)
    nu2, _, _ = propagate_kepler_orbit(a, e, nu1, dt2)

    # 1. Identity: Phi(nu0, nu0) == I
    phi_identity = compute_yamanaka_ankersen_stm(nu0, nu0, e, a, delta_time_s=0.0)
    assert np.allclose(phi_identity, np.eye(6), atol=1e-12)

    # 2. Composition: Phi(nu2, nu0) == Phi(nu2, nu1) @ Phi(nu1, nu0)
    phi_01 = compute_yamanaka_ankersen_stm(nu0, nu1, e, a, delta_time_s=dt1)
    phi_12 = compute_yamanaka_ankersen_stm(nu1, nu2, e, a, delta_time_s=dt2)
    phi_02 = compute_yamanaka_ankersen_stm(nu0, nu2, e, a, delta_time_s=dt1 + dt2)

    composed = phi_12 @ phi_01
    assert np.allclose(phi_02, composed, atol=1e-9)

    # 3. Inverse: inv(Phi(nu1, nu0)) == Phi(nu0, nu1)
    phi_10 = compute_yamanaka_ankersen_stm(nu1, nu0, e, a, delta_time_s=-dt1)
    assert np.allclose(np.linalg.inv(phi_01), phi_10, atol=1e-9)


def test_yamanaka_ankersen_propagation_consistency():
    """Verify that propagate_yamanaka_ankersen produces physically consistent states."""
    a = 7500e3
    e = 0.1
    state_0 = np.array([200.0, -100.0, 50.0, 0.0, -0.2, 0.05])
    t_span = np.linspace(0.0, 1500.0, 10)

    traj = propagate_yamanaka_ankersen(state_0, t_span, a, e, true_anomaly_0=0.2)
    assert traj.shape == (10, 6)
    # Initial state must match
    assert np.allclose(traj[0], state_0, atol=1e-12)


# ==============================================================================
# 4. APPROACH CORRIDOR SAFETY LIMITS TESTS
# ==============================================================================

def test_check_approach_corridor_nominal_vbar():
    """Verify nominal V-bar approach inside corridor is marked safe."""
    # Approaching from +y: y=500m, ydot = -0.5 m/s (closing), small lateral offset (10m)
    state_safe = np.array([5.0, 500.0, 2.0, 0.0, -0.5, 0.0])
    is_safe, violations = check_approach_corridor(
        state_safe, corridor_type=CorridorType.V_BAR, cone_half_angle_deg=15.0,
        glideslope_slope=0.002, min_closing_vel_m_s=0.01, koz_radius_m=200.0
    )
    assert is_safe is True
    assert len(violations) == 0


def test_check_approach_corridor_cone_violation():
    """Verify lateral offset exceeding cone angle triggers violation."""
    # At y = 100 m, cone half angle 10 deg => allowable lateral offset = 100 * tan(10 deg) = 17.63 m
    # Provide lateral offset = sqrt(20^2 + 10^2) = 22.36 m > 17.63 m
    state_outside = np.array([20.0, 100.0, 10.0, 0.0, -0.1, 0.0])
    is_safe, violations = check_approach_corridor(
        state_outside, corridor_type=CorridorType.V_BAR, cone_half_angle_deg=10.0
    )
    assert is_safe is False
    assert any("Cone violation" in v for v in violations)


def test_check_approach_corridor_velocity_exceedance():
    """Verify excessive glideslope approach speed triggers violation."""
    # At y = 200 m, slope 0.002, min_vel 0.01 => allowable speed = 0.002*200 + 0.01 = 0.41 m/s
    # Provide closing speed = 1.2 m/s
    state_too_fast = np.array([0.0, 200.0, 0.0, 0.0, -1.2, 0.0])
    is_safe, violations = check_approach_corridor(
        state_too_fast, corridor_type=CorridorType.V_BAR, glideslope_slope=0.002,
        min_closing_vel_m_s=0.01
    )
    assert is_safe is False
    assert any("Glideslope velocity exceedance" in v for v in violations)


def test_check_approach_corridor_negative_closing_vel():
    """Verify moving away from target is flagged."""
    # Spacecraft at +y moving further positive (opening range)
    state_retreating = np.array([0.0, 300.0, 0.0, 0.0, +0.3, 0.0])
    is_safe, violations = check_approach_corridor(
        state_retreating, corridor_type=CorridorType.V_BAR
    )
    assert is_safe is False
    assert any("Negative closing velocity" in v for v in violations)


def test_check_approach_corridor_rbar():
    """Verify R-bar corridor checking (radial axis x is primary, lateral is sqrt(y^2+z^2))."""
    # Safe R-bar approach from below (+x = 300m, xdot = -0.3 m/s)
    state_rbar_safe = np.array([300.0, 5.0, 2.0, -0.3, 0.0, 0.0])
    is_safe, violations = check_approach_corridor(
        state_rbar_safe, corridor_type=CorridorType.R_BAR, cone_half_angle_deg=15.0
    )
    assert is_safe is True

    # Unsafe R-bar approach: lateral offset y = 150m > 300 * tan(15 deg) = 80.38m
    state_rbar_unsafe = np.array([300.0, 150.0, 0.0, -0.3, 0.0, 0.0])
    is_safe, violations = check_approach_corridor(
        state_rbar_unsafe, corridor_type=CorridorType.R_BAR, cone_half_angle_deg=15.0
    )
    assert is_safe is False
    assert any("Cone violation" in v for v in violations)


def test_validate_approach_trajectory_safe_glideslope():
    """Verify trajectory validator accepts a well-behaved continuous glideslope approach."""
    # Synthesize safe decelerating V-bar glideslope trajectory: y(t) = y0 * exp(-c * t)
    n_pts = 50
    t_span = np.linspace(0.0, 1000.0, n_pts)
    c_decay = 0.002
    y_vals = 1000.0 * np.exp(-c_decay * t_span)
    ydot_vals = -c_decay * y_vals  # perfectly tracks glideslope slope 0.002

    states = np.zeros((n_pts, 6))
    states[:, 1] = y_vals
    states[:, 4] = ydot_vals
    # Lateral offsets well inside 15 deg cone (cone limit is y * tan(15 deg) ~ 0.268 * y)
    states[:, 0] = 0.02 * y_vals
    states[:, 2] = 0.01 * y_vals

    report = validate_approach_trajectory(
        states, t_span, corridor_type="V-bar", cone_half_angle_deg=15.0,
        glideslope_slope=0.003, min_closing_vel_m_s=0.05, max_closing_vel_m_s=3.0,
        capture_radius_m=10.0
    )

    assert report.is_safe is True
    assert len(report.cone_violations) == 0
    assert len(report.velocity_violations) == 0
    assert report.min_distance_to_target_m < 150.0


def test_passive_collision_safety_check():
    """Verify passive collision avoidance (CAM) screening."""
    a = 7000e3
    n = math.sqrt(EARTH_MU_SI / (a**3))
    period = 2.0 * math.pi / n

    # Spacecraft on a drift trajectory offset by 500 m radially
    safe_abort_state = np.array([500.0, 0.0, 0.0, 0.0, -2.0 * n * 500.0, 0.0])
    t_span = np.linspace(0.0, 2.0 * period, 50)
    is_safe, min_dist = check_passive_collision_safety(
        safe_abort_state, t_span, n, koz_radius_m=200.0
    )
    assert is_safe is True
    assert min_dist >= 200.0


# ==============================================================================
# 5. MULTI-IMPULSE RENDEZVOUS PLANNING TESTS
# ==============================================================================

def test_plan_2_impulse_circular_rendezvous():
    """Verify 2-impulse rendezvous in circular orbit lands exactly on target."""
    a = 7000e3
    n = math.sqrt(EARTH_MU_SI / (a**3))
    t_transfer = 1800.0  # 30 minutes (~ 1/3 orbit)

    # Initial state: 1000m along-track, 50m radial
    state_0 = np.array([50.0, 1000.0, 0.0, 0.0, 0.0, 0.0])
    # Target state: Origin with zero velocity
    state_f = np.zeros(6)

    plan = plan_2_impulse_rendezvous(
        initial_relative_state=state_0,
        final_relative_state=state_f,
        transfer_time_s=t_transfer,
        orbit_type="circular",
        semi_major_axis_m=a,
        dry_mass_kg=400.0,
        isp_s=310.0,
        num_trajectory_steps=100,
    )

    assert plan.success is True
    assert plan.total_delta_v > 0.0
    assert plan.propellant_used_kg > 0.0
    # Final state in simulated trajectory must reach target within millimeter tolerance
    final_pos = plan.trajectory_states[-1, 0:3]
    final_vel = plan.trajectory_states[-1, 3:6]
    assert np.allclose(final_pos, state_f[0:3], atol=1e-3)
    assert np.allclose(final_vel, state_f[3:6], atol=1e-3)


def test_plan_2_impulse_elliptical_rendezvous():
    """Verify 2-impulse rendezvous targeting on an eccentric orbit (e = 0.15)."""
    a = 7500e3
    e = 0.15
    nu0 = 0.5
    t_transfer = 2000.0

    state_0 = np.array([80.0, -800.0, 20.0, 0.0, 0.1, 0.0])
    state_f = np.zeros(6)

    plan = plan_2_impulse_rendezvous(
        initial_relative_state=state_0,
        final_relative_state=state_f,
        transfer_time_s=t_transfer,
        orbit_type="elliptical",
        semi_major_axis_m=a,
        eccentricity=e,
        true_anomaly_0=nu0,
        dry_mass_kg=500.0,
        isp_s=300.0,
        num_trajectory_steps=100,
    )

    assert plan.success is True
    assert plan.total_delta_v > 0.0
    final_pos = plan.trajectory_states[-1, 0:3]
    assert np.allclose(final_pos, state_f[0:3], atol=1e-3)


def test_plan_2_impulse_singular_geometry_detection():
    """Verify that targeting near full orbital periods detects ill-conditioned Phi_rv."""
    a = 7000e3
    n = math.sqrt(EARTH_MU_SI / (a**3))
    period = 2.0 * math.pi / n  # Exactly 1 orbit -> sin(nt)=0 => singular

    state_0 = np.array([100.0, 500.0, 0.0, 0.0, 0.0, 0.0])
    state_f = np.zeros(6)

    plan = plan_2_impulse_rendezvous(
        initial_relative_state=state_0,
        final_relative_state=state_f,
        transfer_time_s=period,
        orbit_type="circular",
        semi_major_axis_m=a,
    )

    assert plan.success is False
    assert "Singular transfer geometry" in plan.message


def test_plan_4_impulse_rendezvous_with_waypoint_hold():
    """Verify 4-impulse rendezvous through intermediate waypoint hold point."""
    a = 7000e3
    t_leg1 = 1500.0
    t_hold = 600.0   # 10 min hold
    t_leg2 = 1200.0

    state_0 = np.array([40.0, 2000.0, 0.0, 0.0, 0.0, 0.0])
    # Waypoint S1 at y = 500 m
    waypoint = np.array([0.0, 500.0, 0.0, 0.0, 0.0, 0.0])
    # Final capture box at y = 10 m
    state_f = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])

    plan = plan_4_impulse_rendezvous(
        initial_relative_state=state_0,
        final_relative_state=state_f,
        leg1_transfer_time_s=t_leg1,
        leg2_transfer_time_s=t_leg2,
        waypoint_relative_state=waypoint,
        hold_duration_s=t_hold,
        orbit_type="circular",
        semi_major_axis_m=a,
        dry_mass_kg=450.0,
        isp_s=310.0,
    )

    assert plan.success is True
    assert plan.maneuver_type == "4-impulse"
    assert plan.delta_v_1 is not None
    assert plan.delta_v_2 is not None
    assert plan.delta_v_3 is not None
    assert plan.delta_v_4 is not None
    assert plan.total_delta_v > 0.0
    assert plan.propellant_used_kg > 0.0
    assert plan.transfer_time_s == pytest.approx(t_leg1 + t_hold + t_leg2, abs=1e-6)

    # Trajectory reaches final state
    final_pos = plan.trajectory_states[-1, 0:3]
    assert np.allclose(final_pos, state_f[0:3], atol=1e-2)


# ==============================================================================
# 6. OBJECT-ORIENTED PROPAGATORS & VALIDATORS INTEGRATION
# ==============================================================================

def test_oop_propagator_classes():
    """Verify ClohessyWiltshirePropagator and YamanakaAnkersenPropagator interfaces."""
    a = 7000e3
    cw_prop = ClohessyWiltshirePropagator(semi_major_axis_m=a)
    assert cw_prop.period_s > 0.0

    state_0 = np.array([100.0, 0.0, 0.0, 0.0, cw_prop.closed_orbit_velocity(100.0), 0.0])
    geom = cw_prop.geometry(state_0)
    assert geom.is_closed is True

    t_span = np.array([0.0, 600.0, 1200.0])
    traj_cw = cw_prop.propagate(state_0, t_span)
    assert traj_cw.shape == (3, 6)

    # YA propagator
    ya_prop = YamanakaAnkersenPropagator(semi_major_axis_m=a, eccentricity=0.08)
    assert ya_prop.eccentricity == 0.08
    traj_ya = ya_prop.propagate(state_0, t_span)
    assert traj_ya.shape == (3, 6)


def test_oop_safety_validator_integration():
    """Verify ApproachSafetyValidator wrapper works seamlessly."""
    validator = ApproachSafetyValidator(
        corridor_type="V-bar", cone_half_angle_deg=20.0, glideslope_slope=0.005
    )

    # State test
    safe_state = np.array([1.0, 300.0, 1.0, 0.0, -0.2, 0.0])
    is_safe, reasons = validator.check_state(safe_state)
    assert is_safe is True

    # Trajectory test
    t_span = np.array([0.0, 100.0])
    states = np.array([
        [1.0, 300.0, 1.0, 0.0, -0.2, 0.0],
        [1.0, 280.0, 1.0, 0.0, -0.2, 0.0],
    ])
    report = validator.validate_trajectory(states, t_span)
    assert isinstance(report, ApproachSafetyReport)
    assert report.is_safe is True
