"""
Unit tests for 6-DOF Attitude Determination & Control System (ADCS),
Quaternion Kinematics, Reaction Wheel Pyramid, and Magnetorquer Desaturation.
"""

import math
import numpy as np

from aegis_leo.astrodynamics.attitude_adcs import (
    build_pyramid_rwa_matrix,
    compute_leo_disturbance_torques,
    euler_to_quaternion,
    normalize_quaternion,
    quaternion_conjugate,
    quaternion_error,
    quaternion_multiply,
    quaternion_to_dcm,
    quaternion_to_euler_deg,
    simulate_attitude_slew,
)


def test_quaternion_normalization_and_identity():
    q = normalize_quaternion(np.array([0.0, 0.0, 0.0, 5.0]))
    assert np.allclose(q, [0.0, 0.0, 0.0, 1.0])
    q_zero = normalize_quaternion(np.zeros(4))
    assert np.allclose(q_zero, [0.0, 0.0, 0.0, 1.0])


def test_euler_quaternion_roundtrip():
    roll, pitch, yaw = 15.0, -25.0, 40.0
    q = euler_to_quaternion(roll, pitch, yaw)
    r_out, p_out, y_out = quaternion_to_euler_deg(q)
    assert math.isclose(r_out, roll, abs_tol=1e-4)
    assert math.isclose(p_out, pitch, abs_tol=1e-4)
    assert math.isclose(y_out, yaw, abs_tol=1e-4)


def test_quaternion_multiply_and_conjugate():
    q = euler_to_quaternion(20.0, 10.0, -30.0)
    q_conj = quaternion_conjugate(q)
    prod = quaternion_multiply(q, q_conj)
    assert np.allclose(prod[:3], [0.0, 0.0, 0.0], atol=1e-7)
    assert math.isclose(abs(float(prod[3])), 1.0, abs_tol=1e-7)


def test_quaternion_to_dcm_orthogonality():
    q = euler_to_quaternion(33.0, -18.0, 72.0)
    dcm = quaternion_to_dcm(q)
    assert dcm.shape == (3, 3)
    assert np.allclose(dcm @ dcm.T, np.eye(3), atol=1e-7)
    assert math.isclose(float(np.linalg.det(dcm)), 1.0, abs_tol=1e-7)


def test_pyramid_rwa_pseudoinverse():
    W, W_pinv = build_pyramid_rwa_matrix()
    assert W.shape == (3, 4)
    assert W_pinv.shape == (4, 3)
    assert np.allclose(W @ W_pinv, np.eye(3), atol=1e-7)


def test_leo_disturbance_torques_increase_in_storm():
    q = euler_to_quaternion(10.0, 20.0, 5.0)
    I_mat = np.diag([42.0, 38.0, 26.0])
    cp = np.array([0.12, -0.05, 0.08])
    tau_quiet, b_body = compute_leo_disturbance_torques(q, I_mat, 450.0, kp=1.0, f10_7=75.0, cp_cg_offset_m=cp)
    tau_storm, _ = compute_leo_disturbance_torques(q, I_mat, 450.0, kp=8.5, f10_7=260.0, cp_cg_offset_m=cp)
    assert np.linalg.norm(b_body) > 1e-6
    assert np.linalg.norm(tau_storm) > np.linalg.norm(tau_quiet)


def test_simulate_attitude_slew_convergence():
    rep = simulate_attitude_slew(
        target_roll_deg=5.0,
        target_pitch_deg=35.0,
        target_yaw_deg=-20.0,
        altitude_km=550.0,
        kp_storm=6.0,
        duration_s=90.0,
    )
    assert rep.initial_error_deg > 20.0
    assert rep.slew_converged is True
    assert rep.burn_attitude_ready is True
    assert rep.final_pointing_error_arcsec < 200.0
    assert rep.peak_wheel_speed_rpm > 0.0
    assert rep.total_adcs_energy_wh > 0.0
    assert len(rep.timeline) > 5
    d = rep.to_dict()
    assert "settling_time_s" in d
