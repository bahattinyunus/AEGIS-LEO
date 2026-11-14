"""
Unit Tests for AEGIS-LEO Autonomous Orbit Determination (AOD) Sensor Fusion
===========================================================================
Validates Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF),
Star Tracker optical LOS, GNSS pseudorange/Doppler, and 3-axis Magnetometer
under geomagnetic storm space-weather environments.
"""

import pytest
import math
import numpy as np

from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    EARTH_MU,
    QUIET_KP,
    STORM_G1_KP,
    STORM_G4_KP,
    STORM_G5_KP,
)
from aegis_leo.navigation.sensor_fusion import (
    NavigationState,
    GNSSObservation,
    GNSSBatchMeasurement,
    StarTrackerObservation,
    MagnetometerObservation,
    FilterUpdateDiagnostics,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    AODSensorFusionEngine,
    compute_geomagnetic_field,
    compute_geomagnetic_jacobian,
    compute_magnetometer_covariance,
    compute_los_vector,
    compute_los_jacobian,
    compute_gnss_pseudorange,
    compute_gnss_doppler,
    compute_gnss_jacobian,
    compute_gdop,
    compute_orbit_accelerations,
    propagate_orbit_rk4,
    compute_dynamics_jacobian,
    compute_state_transition_matrix,
    compute_process_noise_covariance,
    evaluate_navigation_performance,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nominal_leo_state():
    """Nominal circular LEO orbit at 500 km altitude (inclination ~51.6 deg)."""
    r_mag = EARTH_RADIUS_KM + 500.0  # ~6878.137 km
    v_circ = math.sqrt(EARTH_MU / r_mag)  # ~7.612 km/s

    inc = math.radians(51.6)
    pos = np.array([r_mag, 0.0, 0.0], dtype=float)
    vel = np.array([0.0, v_circ * math.cos(inc), v_circ * math.sin(inc)], dtype=float)

    cov = np.eye(8, dtype=float)
    cov[0:3, 0:3] *= (0.5 ** 2)       # (500 m)^2 = 0.25 km^2
    cov[3:6, 3:6] *= (0.002 ** 2)     # (2 m/s)^2 = 4e-6 km^2/s^2
    cov[6, 6] = (0.005 ** 2)          # (5 m clock bias)^2
    cov[7, 7] = (0.0001 ** 2)         # clock drift

    return NavigationState(
        epoch_seconds=0.0,
        position_km=pos,
        velocity_km_s=vel,
        clock_bias_km=0.003,       # 3 meters clock bias
        clock_drift_km_s=0.00001,  # 1 cm/s drift
        covariance=cov,
    )


@pytest.fixture
def gnss_constellation(nominal_leo_state):
    """4 GNSS satellites in MEO (~26,560 km radius) with good GDOP."""
    r_meo = 26560.0  # km
    v_meo = math.sqrt(EARTH_MU / r_meo)  # ~3.874 km/s

    sats = []
    # 4 tetrahedron-like geometries for optimal dilution of precision
    configs = [
        (r_meo, 0.0, 0.0, 0.0, v_meo, 0.0),
        (0.0, r_meo, 0.0, -v_meo, 0.0, 0.0),
        (0.0, 0.0, r_meo, 0.0, v_meo, 0.0),
        (-r_meo / math.sqrt(2), -r_meo / math.sqrt(2), 0.0, v_meo / math.sqrt(2), -v_meo / math.sqrt(2), 0.0),
    ]

    for i, cfg in enumerate(configs):
        sat_pos = np.array(cfg[0:3], dtype=float)
        sat_vel = np.array(cfg[3:6], dtype=float)
        pr_true, _, _ = compute_gnss_pseudorange(
            nominal_leo_state.position_km, sat_pos, nominal_leo_state.clock_bias_km
        )
        dop_true = compute_gnss_doppler(
            nominal_leo_state.position_km, nominal_leo_state.velocity_km_s,
            sat_pos, sat_vel, nominal_leo_state.clock_drift_km_s
        )
        sats.append(GNSSObservation(
            sat_id=f"GPS_{i+1:02d}",
            prn=i + 1,
            pseudorange_km=pr_true + 0.0005,  # 0.5 m synthetic noise
            sat_pos_km=sat_pos,
            pseudorange_rate_km_s=dop_true + 0.00001,
            sat_vel_km_s=sat_vel,
            pseudorange_variance_km2=(0.002 ** 2),  # 2m 1-sigma
            doppler_variance_km2_s2=(0.00005 ** 2),
        ))
    return GNSSBatchMeasurement(epoch_seconds=0.0, observations=sats)


# ---------------------------------------------------------------------------
# Test Cases: NavigationState & Representation
# ---------------------------------------------------------------------------

def test_navigation_state_properties(nominal_leo_state):
    state = nominal_leo_state
    assert math.isclose(state.altitude_km, 500.0, rel_tol=1e-5)
    assert math.isclose(state.radius_km, EARTH_RADIUS_KM + 500.0, rel_tol=1e-6)
    assert 7.5 < state.speed_km_s < 7.8

    # 6-dim vector conversion
    v6 = state.as_vector(dim=6)
    assert len(v6) == 6
    assert np.allclose(v6[0:3], state.position_km)
    assert np.allclose(v6[3:6], state.velocity_km_s)

    # 8-dim vector conversion and round-trip
    v8 = state.as_vector(dim=8)
    assert len(v8) == 8
    assert math.isclose(v8[6], state.clock_bias_km)
    assert math.isclose(v8[7], state.clock_drift_km_s)

    reconstructed = NavigationState.from_vector(v8, epoch_seconds=12.5, covariance=state.covariance)
    assert math.isclose(reconstructed.epoch_seconds, 12.5)
    assert np.allclose(reconstructed.position_km, state.position_km)
    assert math.isclose(reconstructed.clock_bias_km, state.clock_bias_km)


# ---------------------------------------------------------------------------
# Test Cases: Geomagnetic Field Physics & Storm Disturbance
# ---------------------------------------------------------------------------

def test_geomagnetic_field_quiet_vs_storm(nominal_leo_state):
    r_sc = nominal_leo_state.position_km

    # Quiet field
    b_quiet = compute_geomagnetic_field(r_sc, epoch_seconds=0.0, kp=QUIET_KP, dst_nt=0.0)
    b_mag_quiet = float(np.linalg.norm(b_quiet))
    # At 500 km LEO, magnetic field should be ~20,000 to 50,000 nT
    assert 20000.0 < b_mag_quiet < 55000.0

    # Geomagnetic storm G4 (Kp = 8.0) with strong Dst ring current depression (-150 nT)
    b_storm = compute_geomagnetic_field(r_sc, epoch_seconds=0.0, kp=STORM_G4_KP, dst_nt=-150.0)
    b_diff = float(np.linalg.norm(b_storm - b_quiet))
    assert b_diff > 100.0  # Noticeable external magnetic perturbation

    # Extreme G5 storm (Kp = 9.0)
    b_g5 = compute_geomagnetic_field(r_sc, epoch_seconds=0.0, kp=STORM_G5_KP, dst_nt=-350.0)
    assert float(np.linalg.norm(b_g5 - b_quiet)) > float(np.linalg.norm(b_storm - b_quiet))


def test_geomagnetic_jacobian_maxwell_and_finite_difference(nominal_leo_state):
    r_sc = nominal_leo_state.position_km
    h_analytical = compute_geomagnetic_jacobian(r_sc, epoch_seconds=100.0)

    # Maxwell's equation: div(B) = trace(H) == 0
    trace_h = float(np.trace(h_analytical))
    assert math.isclose(trace_h, 0.0, abs_tol=1e-7)

    # Central finite difference comparison
    eps = 0.01  # 10 meters
    h_numerical = np.zeros((3, 3), dtype=float)
    for j in range(3):
        perturb = np.zeros(3)
        perturb[j] = eps
        b_plus = compute_geomagnetic_field(r_sc + perturb, epoch_seconds=100.0, kp=QUIET_KP, dst_nt=0.0)
        b_minus = compute_geomagnetic_field(r_sc - perturb, epoch_seconds=100.0, kp=QUIET_KP, dst_nt=0.0)
        h_numerical[:, j] = (b_plus - b_minus) / (2.0 * eps)

    # Note: numerical B includes small dipole axis projection terms; compare main structure
    assert np.allclose(h_analytical, h_numerical, rtol=1e-3, atol=1e-2)


def test_storm_adaptive_magnetometer_covariance():
    base_var = 25.0 ** 2  # 625 nT^2
    r_quiet, scale_quiet = compute_magnetometer_covariance(base_var, kp=1.5, dst_nt=0.0)
    assert math.isclose(scale_quiet, 1.0)
    assert np.allclose(r_quiet, base_var * np.eye(3))

    # G1 Minor Storm
    _, scale_g1 = compute_magnetometer_covariance(base_var, kp=STORM_G1_KP, dst_nt=-50.0)
    assert scale_g1 > 1.0

    # G4 Severe Storm
    _, scale_g4 = compute_magnetometer_covariance(base_var, kp=STORM_G4_KP, dst_nt=-150.0)
    assert scale_g4 > scale_g1 * 5.0

    # G5 Extreme Storm
    r_g5, scale_g5 = compute_magnetometer_covariance(base_var, kp=STORM_G5_KP, dst_nt=-300.0)
    assert scale_g5 > scale_g4
    assert r_g5[0, 0] > 10.0 * base_var


# ---------------------------------------------------------------------------
# Test Cases: Star Tracker Optical LOS
# ---------------------------------------------------------------------------

def test_star_tracker_los_and_jacobian(nominal_leo_state):
    r_sc = nominal_leo_state.position_km
    # Ground optical tracking beacon at Madrid / Goldstone coordinates
    r_beacon = np.array([4849.0, -360.0, 4115.0], dtype=float)

    u_los, dist_km = compute_los_vector(r_sc, r_beacon)
    assert math.isclose(float(np.linalg.norm(u_los)), 1.0, rel_tol=1e-7)
    assert dist_km > 500.0

    h_analytical = compute_los_jacobian(r_sc, r_beacon, state_dim=8)
    assert h_analytical.shape == (3, 8)
    # Velocity and clock sensitivities should be identically zero
    assert np.allclose(h_analytical[:, 3:8], 0.0)

    # Numerical Jacobian check for position part
    eps = 0.001  # 1 meter
    h_numerical = np.zeros((3, 3), dtype=float)
    for j in range(3):
        perturb = np.zeros(3)
        perturb[j] = eps
        u_plus, _ = compute_los_vector(r_sc + perturb, r_beacon)
        u_minus, _ = compute_los_vector(r_sc - perturb, r_beacon)
        h_numerical[:, j] = (u_plus - u_minus) / (2.0 * eps)

    assert np.allclose(h_analytical[:, 0:3], h_numerical, rtol=1e-4, atol=1e-5)

    # Orthogonality property: d(u)/dr along u is zero
    assert np.allclose(u_los @ h_analytical[:, 0:3], np.zeros(3), atol=1e-6)


# ---------------------------------------------------------------------------
# Test Cases: GNSS Pseudorange & GDOP
# ---------------------------------------------------------------------------

def test_gnss_models_and_dilution_of_precision(nominal_leo_state, gnss_constellation):
    r_sc = nominal_leo_state.position_km
    v_sc = nominal_leo_state.velocity_km_s
    obs0 = gnss_constellation.observations[0]

    # Pseudorange calculation
    pr, rng, u_vec = compute_gnss_pseudorange(r_sc, obs0.sat_pos_km, nominal_leo_state.clock_bias_km)
    assert pr > rng
    assert math.isclose(pr - rng, nominal_leo_state.clock_bias_km, abs_tol=1e-8)
    assert math.isclose(float(np.linalg.norm(u_vec)), 1.0)

    # Doppler calculation
    dop = compute_gnss_doppler(
        r_sc, v_sc, obs0.sat_pos_km, obs0.sat_vel_km_s, nominal_leo_state.clock_drift_km_s
    )
    assert abs(dop) < 15.0  # Relative velocity in LEO-MEO is < 15 km/s

    # Jacobian check
    h_mat = compute_gnss_jacobian(
        r_sc, obs0.sat_pos_km, state_dim=8, include_doppler=True,
        v_spacecraft_km_s=v_sc, sat_vel_km_s=obs0.sat_vel_km_s
    )
    assert h_mat.shape == (2, 8)
    assert math.isclose(h_mat[0, 6], 1.0)  # Clock bias derivative
    assert math.isclose(h_mat[1, 7], 1.0)  # Clock drift derivative

    # GDOP
    sat_positions = [obs.sat_pos_km for obs in gnss_constellation.observations]
    dop_metrics = compute_gdop(r_sc, sat_positions)
    assert dop_metrics["GDOP"] < 5.0  # Good constellation geometry
    assert dop_metrics["PDOP"] < 4.0
    assert dop_metrics["TDOP"] < 3.0


# ---------------------------------------------------------------------------
# Test Cases: Orbital Propagation & Dynamics
# ---------------------------------------------------------------------------

def test_orbital_propagation_and_transition_matrix(nominal_leo_state):
    state = nominal_leo_state

    # 1. Propagation for 60 seconds via RK4
    state_60 = propagate_orbit_rk4(state, dt_seconds=60.0, include_j2=True)
    assert math.isclose(state_60.epoch_seconds, 60.0)
    # Energy should remain nearly conserved (radius roughly constant for circular orbit)
    assert math.isclose(state_60.radius_km, state.radius_km, rel_tol=1e-3)

    # Clock propagation
    expected_bias = state.clock_bias_km + state.clock_drift_km_s * 60.0
    assert math.isclose(state_60.clock_bias_km, expected_bias, abs_tol=1e-9)

    # 2. Dynamics Jacobian
    f_mat = compute_dynamics_jacobian(state, dim=8, include_j2=True)
    assert f_mat.shape == (8, 8)
    assert np.allclose(f_mat[0:3, 3:6], np.eye(3))  # dr/dt = v
    assert math.isclose(f_mat[6, 7], 1.0)          # db/dt = drift

    # 3. State transition matrix Phi
    phi = compute_state_transition_matrix(f_mat, dt=10.0)
    assert phi.shape == (8, 8)
    # For small dt, det(Phi) ~ 1 (symplectic/volume-preserving Hamiltonian dynamics)
    det_phi = float(np.linalg.det(phi))
    assert math.isclose(det_phi, 1.0, rel_tol=1e-3)

    # 4. Process noise covariance Q
    q_mat = compute_process_noise_covariance(dt=10.0, dim=8)
    assert q_mat.shape == (8, 8)
    assert np.allclose(q_mat, q_mat.T)
    # Must be positive-semidefinite (eigenvalues >= 0)
    eigenvals = np.linalg.eigvalsh(q_mat)
    assert np.all(eigenvals >= -1e-12)


# ---------------------------------------------------------------------------
# Test Cases: Extended Kalman Filter (EKF)
# ---------------------------------------------------------------------------

def test_ekf_predict_and_updates(nominal_leo_state, gnss_constellation):
    ekf = ExtendedKalmanFilter(initial_state=nominal_leo_state, state_dim=8)
    init_trace = float(np.trace(ekf.state.covariance))

    # 1. Predict step increases uncertainty
    ekf.predict(dt_seconds=10.0)
    pred_trace = float(np.trace(ekf.state.covariance))
    assert pred_trace > init_trace

    # 2. GNSS Batch measurement update at t=10.0
    sats_10 = []
    for obs in gnss_constellation.observations:
        pr_true, _, _ = compute_gnss_pseudorange(
            ekf.state.position_km, obs.sat_pos_km, ekf.state.clock_bias_km
        )
        dop_true = compute_gnss_doppler(
            ekf.state.position_km, ekf.state.velocity_km_s,
            obs.sat_pos_km, obs.sat_vel_km_s, ekf.state.clock_drift_km_s
        )
        sats_10.append(GNSSObservation(
            sat_id=obs.sat_id,
            prn=obs.prn,
            pseudorange_km=pr_true + 0.0005,
            sat_pos_km=obs.sat_pos_km,
            pseudorange_rate_km_s=dop_true + 0.00001,
            sat_vel_km_s=obs.sat_vel_km_s,
            pseudorange_variance_km2=obs.pseudorange_variance_km2,
            doppler_variance_km2_s2=obs.doppler_variance_km2_s2,
        ))
    batch_10 = GNSSBatchMeasurement(epoch_seconds=10.0, observations=sats_10)

    diag_gnss = ekf.update_gnss_batch(batch_10, include_doppler=True)
    assert not diag_gnss.is_gated
    assert diag_gnss.num_measurements == 8  # 4 pseudoranges + 4 Dopplers
    assert diag_gnss.trace_covariance_post < pred_trace

    # 3. Star Tracker LOS update
    r_target = np.array([4000.0, 3000.0, 5000.0], dtype=float)
    u_true, _ = compute_los_vector(ekf.state.position_km, r_target)
    st_obs = StarTrackerObservation(
        epoch_seconds=10.0,
        los_unit_vector=u_true + np.random.normal(0, 1e-5, 3),
        target_pos_km=r_target,
    )
    diag_st = ekf.update_star_tracker(st_obs)
    assert not diag_st.is_gated
    assert diag_st.trace_covariance_post <= diag_gnss.trace_covariance_post

    # 4. Magnetometer update under moderate storm
    b_true = compute_geomagnetic_field(ekf.state.position_km, epoch_seconds=10.0, kp=STORM_G1_KP)
    mag_obs = MagnetometerObservation(
        epoch_seconds=10.0,
        b_measured_nt=b_true + np.array([5.0, -4.0, 3.0]),
        kp_index=STORM_G1_KP,
    )
    diag_mag = ekf.update_magnetometer(mag_obs)
    assert not diag_mag.is_gated
    assert diag_mag.storm_noise_scale > 1.0


def test_ekf_outlier_rejection(nominal_leo_state):
    ekf = ExtendedKalmanFilter(initial_state=nominal_leo_state, state_dim=8, chi2_gate_threshold=9.0)

    # Inject massive corrupted Star Tracker measurement (orthogonal direction)
    r_target = np.array([6000.0, 0.0, 0.0], dtype=float)
    corrupted_los = np.array([0.0, 1.0, 0.0], dtype=float)  # 90-degree outlier vector
    st_obs = StarTrackerObservation(
        epoch_seconds=0.0,
        los_unit_vector=corrupted_los,
        target_pos_km=r_target,
    )
    diag = ekf.update_star_tracker(st_obs)
    assert diag.is_gated  # Outlier properly gated
    assert math.isclose(diag.state_correction_pos_km, 0.0)


# ---------------------------------------------------------------------------
# Test Cases: Unscented Kalman Filter (UKF)
# ---------------------------------------------------------------------------

def test_ukf_predict_and_updates(nominal_leo_state, gnss_constellation):
    ukf = UnscentedKalmanFilter(initial_state=nominal_leo_state, state_dim=8)
    init_trace = float(np.trace(ukf.state.covariance))

    # 1. Predict step
    ukf.predict(dt_seconds=10.0)
    pred_trace = float(np.trace(ukf.state.covariance))
    assert pred_trace > init_trace

    # 2. GNSS Batch update at t=10.0
    sats_10 = []
    for obs in gnss_constellation.observations:
        pr_true, _, _ = compute_gnss_pseudorange(
            ukf.state.position_km, obs.sat_pos_km, ukf.state.clock_bias_km
        )
        dop_true = compute_gnss_doppler(
            ukf.state.position_km, ukf.state.velocity_km_s,
            obs.sat_pos_km, obs.sat_vel_km_s, ukf.state.clock_drift_km_s
        )
        sats_10.append(GNSSObservation(
            sat_id=obs.sat_id,
            prn=obs.prn,
            pseudorange_km=pr_true + 0.0005,
            sat_pos_km=obs.sat_pos_km,
            pseudorange_rate_km_s=dop_true + 0.00001,
            sat_vel_km_s=obs.sat_vel_km_s,
            pseudorange_variance_km2=obs.pseudorange_variance_km2,
            doppler_variance_km2_s2=obs.doppler_variance_km2_s2,
        ))
    batch_10 = GNSSBatchMeasurement(epoch_seconds=10.0, observations=sats_10)

    diag_gnss = ukf.update_gnss_batch(batch_10, include_doppler=True)
    assert not diag_gnss.is_gated
    assert diag_gnss.trace_covariance_post < pred_trace

    # 3. Magnetometer update with storm index
    b_true = compute_geomagnetic_field(ukf.state.position_km, epoch_seconds=10.0, kp=STORM_G4_KP, dst_nt=-120.0)
    mag_obs = MagnetometerObservation(
        epoch_seconds=10.0,
        b_measured_nt=b_true,
        kp_index=STORM_G4_KP,
        dst_index_nt=-120.0,
    )
    diag_mag = ukf.update_magnetometer(mag_obs)
    assert not diag_mag.is_gated
    assert diag_mag.storm_noise_scale > 2.0


# ---------------------------------------------------------------------------
# Test Cases: High-Level AOD Engine & Multi-Sensor Fusion
# ---------------------------------------------------------------------------

def test_aod_engine_multi_sensor_orbit_determination(nominal_leo_state, gnss_constellation):
    # Setup initial estimated state with artificial offset
    perturbed_pos = nominal_leo_state.position_km + np.array([0.3, -0.2, 0.1])  # ~374 meters offset
    perturbed_vel = nominal_leo_state.velocity_km_s + np.array([0.001, 0.001, -0.0005])

    initial_est = NavigationState(
        epoch_seconds=0.0,
        position_km=perturbed_pos,
        velocity_km_s=perturbed_vel,
        clock_bias_km=nominal_leo_state.clock_bias_km + 0.005,  # 5m offset
        clock_drift_km_s=nominal_leo_state.clock_drift_km_s,
        covariance=nominal_leo_state.covariance.copy(),
    )

    engine_ekf = AODSensorFusionEngine(initial_state=initial_est, filter_type="EKF", state_dim=8)
    engine_ukf = AODSensorFusionEngine(initial_state=initial_est, filter_type="UKF", state_dim=8)

    # Step through 5 epochs of multi-sensor updates
    truth_trajectory = []
    current_truth = nominal_leo_state

    r_target_beacon = np.array([3000.0, 4000.0, 5000.0], dtype=float)

    for step in range(5):
        dt = 10.0
        t_epoch = (step + 1) * dt
        current_truth = propagate_orbit_rk4(current_truth, dt_seconds=dt)
        truth_trajectory.append(current_truth)

        # Generate observations at current truth
        # GNSS
        sats = []
        for i, obs in enumerate(gnss_constellation.observations):
            pr_val, _, _ = compute_gnss_pseudorange(
                current_truth.position_km, obs.sat_pos_km, current_truth.clock_bias_km
            )
            dop_val = compute_gnss_doppler(
                current_truth.position_km, current_truth.velocity_km_s,
                obs.sat_pos_km, obs.sat_vel_km_s, current_truth.clock_drift_km_s
            )
            sats.append(GNSSObservation(
                sat_id=obs.sat_id,
                prn=obs.prn,
                pseudorange_km=pr_val,
                sat_pos_km=obs.sat_pos_km,
                pseudorange_rate_km_s=dop_val,
                sat_vel_km_s=obs.sat_vel_km_s,
            ))
        batch = GNSSBatchMeasurement(epoch_seconds=t_epoch, observations=sats)

        # Star tracker
        u_los, _ = compute_los_vector(current_truth.position_km, r_target_beacon)
        st_obs = [StarTrackerObservation(
            epoch_seconds=t_epoch,
            los_unit_vector=u_los,
            target_pos_km=r_target_beacon,
        )]

        # Magnetometer under storm conditions (Kp ramping up to G3 storm)
        kp_val = 2.0 + step * 1.2
        b_vec = compute_geomagnetic_field(current_truth.position_km, epoch_seconds=t_epoch, kp=kp_val)
        mag_obs = MagnetometerObservation(
            epoch_seconds=t_epoch,
            b_measured_nt=b_vec,
            kp_index=kp_val,
        )

        # Process epoch on both engines
        engine_ekf.process_epoch(
            epoch_seconds=t_epoch,
            gnss_batch=batch,
            star_tracker_obs=st_obs,
            magnetometer_obs=mag_obs,
        )
        engine_ukf.process_epoch(
            epoch_seconds=t_epoch,
            gnss_batch=batch,
            star_tracker_obs=st_obs,
            magnetometer_obs=mag_obs,
        )

    # Verify convergence: position error should decrease substantially
    final_dr_ekf = np.linalg.norm(engine_ekf.current_state.position_km - current_truth.position_km) * 1000.0  # meters
    final_dr_ukf = np.linalg.norm(engine_ukf.current_state.position_km - current_truth.position_km) * 1000.0

    init_error_m = np.linalg.norm(perturbed_pos - nominal_leo_state.position_km) * 1000.0
    assert final_dr_ekf < init_error_m * 0.2  # >80% error reduction
    assert final_dr_ukf < init_error_m * 0.2

    # Performance metrics evaluation
    metrics_ekf = evaluate_navigation_performance(truth_trajectory, engine_ekf.history_states[1:])
    assert metrics_ekf["pos_rmse_meters"] < init_error_m
    assert metrics_ekf["covariance_3sigma_consistency_pct"] >= 80.0


def test_magnetometer_body_frame_rotation(nominal_leo_state):
    """Test magnetometer update when sensor is mounted in body frame with known attitude DCM."""
    # 90-degree yaw rotation about Z axis
    dcm_yaw_90 = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    r_sc = nominal_leo_state.position_km
    b_eci = compute_geomagnetic_field(r_sc, epoch_seconds=0.0, kp=QUIET_KP)
    b_body = dcm_yaw_90 @ b_eci

    # Verify body field rotation
    assert math.isclose(np.linalg.norm(b_body), np.linalg.norm(b_eci))
    assert math.isclose(b_body[0], b_eci[1])
    assert math.isclose(b_body[1], -b_eci[0])

    ekf = ExtendedKalmanFilter(initial_state=nominal_leo_state, state_dim=8)
    mag_obs = MagnetometerObservation(
        epoch_seconds=0.0,
        b_measured_nt=b_body,
        dcm_eci_to_body=dcm_yaw_90,
    )
    diag = ekf.update_magnetometer(mag_obs)
    assert not diag.is_gated
    # Innovation should be near zero because measurement perfectly matches model
    assert np.linalg.norm(diag.innovation) < 1e-6


def test_six_state_filter_mode(nominal_leo_state, gnss_constellation):
    """Verify 6-state navigation filter mode (pure orbit without clock states)."""
    ekf_6 = ExtendedKalmanFilter(initial_state=nominal_leo_state, state_dim=6)
    assert ekf_6.state_dim == 6
    assert ekf_6.state.covariance.shape == (6, 6)

    # 1. Update at t=0.0
    diag = ekf_6.update_gnss_batch(gnss_constellation)
    assert not diag.is_gated
    assert diag.trace_covariance_post < diag.trace_covariance_prior

    # 2. Predict
    ekf_6.predict(dt_seconds=5.0)
    assert ekf_6.state.covariance.shape == (6, 6)

    # UKF 6-state
    ukf_6 = UnscentedKalmanFilter(initial_state=nominal_leo_state, state_dim=6)
    assert ukf_6.state_dim == 6
    diag_ukf = ukf_6.update_gnss_batch(gnss_constellation)
    assert not diag_ukf.is_gated
    assert diag_ukf.trace_covariance_post < diag_ukf.trace_covariance_prior

    ukf_6.predict(dt_seconds=5.0)
    assert ukf_6.state.covariance.shape == (6, 6)


def test_gnss_outage_autonomous_navigation(nominal_leo_state):
    """
    Test navigation during a complete GNSS blackout.
    Filter must sustain orbit determination using only Star Tracker + 3-Axis Magnetometer.
    """
    engine = AODSensorFusionEngine(initial_state=nominal_leo_state, filter_type="EKF", state_dim=8)

    current_truth = nominal_leo_state
    r_target_beacon = np.array([5000.0, 2000.0, 4000.0], dtype=float)

    for step in range(6):
        dt = 10.0
        t_epoch = (step + 1) * dt
        current_truth = propagate_orbit_rk4(current_truth, dt_seconds=dt)

        u_los, _ = compute_los_vector(current_truth.position_km, r_target_beacon)
        st_obs = [StarTrackerObservation(
            epoch_seconds=t_epoch,
            los_unit_vector=u_los + np.random.normal(0, 1e-6, 3),
            target_pos_km=r_target_beacon,
        )]

        b_vec = compute_geomagnetic_field(current_truth.position_km, epoch_seconds=t_epoch, kp=QUIET_KP)
        mag_obs = MagnetometerObservation(
            epoch_seconds=t_epoch,
            b_measured_nt=b_vec + np.random.normal(0, 5.0, 3),
            kp_index=QUIET_KP,
        )

        # Zero GNSS observations (complete outage)
        diags = engine.process_epoch(
            epoch_seconds=t_epoch,
            gnss_batch=None,
            star_tracker_obs=st_obs,
            magnetometer_obs=mag_obs,
        )
        assert len(diags) == 2

    # Verify orbit tracking remained stable without diverging
    final_error_km = np.linalg.norm(engine.current_state.position_km - current_truth.position_km)
    assert final_error_km < 1.0  # Kept within 1 km despite complete GNSS outage


def test_star_tracker_invalid_flag(nominal_leo_state):
    """Test that blinded/eclipsed star tracker observations are ignored."""
    ekf = ExtendedKalmanFilter(initial_state=nominal_leo_state, state_dim=8)
    init_cov = ekf.state.covariance.copy()

    blinded_obs = StarTrackerObservation(
        epoch_seconds=0.0,
        los_unit_vector=np.array([1.0, 0.0, 0.0]),
        target_pos_km=np.array([0.0, 0.0, 0.0]),
        is_valid=False,  # Sun blinding
    )
    diag = ekf.update_star_tracker(blinded_obs)
    assert diag.is_gated
    assert np.allclose(ekf.state.covariance, init_cov)


def test_atmospheric_drag_acceleration():
    """Verify drag acceleration opposes velocity and diminishes at higher altitudes."""
    r_leo_400 = np.array([EARTH_RADIUS_KM + 400.0, 0.0, 0.0])
    r_leo_800 = np.array([EARTH_RADIUS_KM + 800.0, 0.0, 0.0])
    v_leo = np.array([0.0, 7.6, 0.0])

    a_tot_400 = compute_orbit_accelerations(r_leo_400, v_leo, include_j2=False)
    a_tot_800 = compute_orbit_accelerations(r_leo_800, v_leo, include_j2=False)

    # In y-direction, only drag acts (gravity acts along x)
    # Drag must decelerate the spacecraft (negative y)
    assert a_tot_400[1] < 0.0
    assert a_tot_800[1] < 0.0
    # Drag at 400 km must be orders of magnitude larger than at 800 km
    assert abs(a_tot_400[1]) > abs(a_tot_800[1]) * 100.0

