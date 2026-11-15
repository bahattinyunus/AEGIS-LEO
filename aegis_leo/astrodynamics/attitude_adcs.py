"""
6-DOF Attitude Determination & Control System (ADCS) Engine
===========================================================

Provides rigid-body spacecraft rotational dynamics, unit-quaternion kinematics,
4-wheel pyramid Reaction Wheel Assembly (RWA) momentum management, magnetic
torque rod (B-dot detumbling & cross-product momentum dumping) control, and
pre-maneuver slew-to-burn attitude verification under space weather disturbances.

Physical & Mathematical Formulation:
  1. Quaternion Kinematics (scalar-last convention q = [qx, qy, qz, qw]):
       dq/dt = 0.5 * Omega(omega) * q
  2. Euler Rotational Dynamics with Reaction Wheels:
       I * d(omega)/dt = -omega x (I * omega + h_rw) - dh_rw/dt + tau_mtq + tau_dist
  3. Environmental Disturbance Torques in LEO:
       - Gravity-Gradient: tau_gg = (3 * mu / r^3) * (r_hat_b x (I * r_hat_b))
       - Aerodynamic Weathercock: tau_aero = r_cp x (-0.5 * rho * Cd * A * v_rel^2 * v_hat_b)
       - Solar Radiation Pressure: tau_srp = r_sp x F_srp
  4. 4-Wheel Pyramid RWA Allocation:
       tau_rw_4 = W^+ * tau_cmd_3,  where W^+ = W^T * (W * W^T)^(-1)
  5. Magnetorquer (MTQ) Momentum Desaturation & B-dot Detumbling:
       m_dump = k_dump * (h_rw x B_b) / ||B_b||^2
       tau_mtq = m_dump x B_b
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aegis_leo.constants import EARTH_RADIUS_KM, MU_EARTH_KM3_S2
from aegis_leo.physics.atmosphere import compute_atmospheric_density


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize a 4-element quaternion [qx, qy, qz, qw] with positive scalar part convention."""
    q_arr = np.asarray(q, dtype=np.float64).flatten()
    norm = float(np.linalg.norm(q_arr))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q_arr = q_arr / norm
    return q_arr


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """Return conjugate [-qx, -qy, -qz, qw] of unit quaternion."""
    q_arr = normalize_quaternion(q)
    return np.array([-q_arr[0], -q_arr[1], -q_arr[2], q_arr[3]], dtype=np.float64)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [qx, qy, qz, qw] using Hamilton product."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quaternion_error(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """Compute attitude error quaternion q_err = q_target^(-1) (*) q_current."""
    q_c = normalize_quaternion(q_current)
    q_t = normalize_quaternion(q_target)
    q_err = quaternion_multiply(quaternion_conjugate(q_t), q_c)
    return normalize_quaternion(q_err)


def euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Convert 3-2-1 (yaw-pitch-roll) Euler angles in degrees to quaternion [qx, qy, qz, qw]."""
    phi = math.radians(roll_deg) * 0.5
    theta = math.radians(pitch_deg) * 0.5
    psi = math.radians(yaw_deg) * 0.5

    cr, sr = math.cos(phi), math.sin(phi)
    cp, sp = math.cos(theta), math.sin(theta)
    cy, sy = math.cos(psi), math.sin(psi)

    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return normalize_quaternion(np.array([qx, qy, qz, qw], dtype=np.float64))


def quaternion_to_euler_deg(q: np.ndarray) -> Tuple[float, float, float]:
    """Convert quaternion [qx, qy, qz, qw] to (roll_deg, pitch_deg, yaw_deg)."""
    qx, qy, qz, qw = normalize_quaternion(q)

    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [qx, qy, qz, qw] to 3x3 Direction Cosine Matrix (Inertial -> Body)."""
    qx, qy, qz, qw = normalize_quaternion(q)
    return np.array(
        [
            [1.0 - 2.0 * (qy**2 + qz**2), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy)],
            [2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx**2 + qz**2), 2.0 * (qy * qz + qw * qx)],
            [2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx**2 + qy**2)],
        ],
        dtype=np.float64,
    )


def build_pyramid_rwa_matrix(cant_angle_deg: float = 54.7356) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build 3x4 Reaction Wheel Assembly (RWA) spin-axis projection matrix W
    and its 4x3 Moore-Penrose pseudoinverse W_pinv for a 4-wheel symmetric pyramid.
    """
    beta = math.radians(cant_angle_deg)
    cb, sb = math.cos(beta), math.sin(beta)
    W = np.array(
        [
            [sb, -sb, -sb, sb],
            [sb, sb, -sb, -sb],
            [cb, cb, cb, cb],
        ],
        dtype=np.float64,
    )
    # Normalize columns to unit vectors
    W = W / np.linalg.norm(W, axis=0, keepdims=True)
    W_pinv = W.T @ np.linalg.inv(W @ W.T)
    return W, W_pinv


@dataclass
class SpacecraftADCSConfig:
    """Physical parameters for spacecraft 6-DOF attitude & RWA/MTQ control."""
    inertia_diag_kg_m2: Tuple[float, float, float] = (42.0, 38.0, 26.0)
    max_wheel_torque_nm: float = 0.12
    max_wheel_momentum_nms: float = 4.0
    wheel_inertia_kg_m2: float = 0.0065
    max_mtq_dipole_am2: float = 15.0
    cp_cg_offset_m: Tuple[float, float, float] = (0.12, -0.05, 0.08)
    drag_area_m2: float = 2.4
    cd: float = 2.2
    kp_gain: float = 2.8
    kd_gain: float = 9.5
    mtq_dump_gain: float = 0.04


@dataclass
class ADCSSlewStep:
    time_s: float
    pointing_error_deg: float
    omega_norm_deg_s: float
    max_wheel_rpm: float
    rwa_torque_norm_mnm: float
    disturbance_torque_urnm: float


@dataclass
class ADCSSlewReport:
    target_euler_deg: Tuple[float, float, float]
    initial_error_deg: float
    final_pointing_error_arcsec: float
    settling_time_s: float
    slew_converged: bool
    burn_attitude_ready: bool
    peak_wheel_speed_rpm: float
    peak_wheel_momentum_nms: float
    wheel_saturation_pct: float
    peak_rwa_torque_mnm: float
    total_adcs_energy_wh: float
    mtq_momentum_dumped_nms: float
    mean_disturbance_torque_unm: float
    timeline: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_leo_disturbance_torques(
    q_ib: np.ndarray,
    inertia_mat: np.ndarray,
    altitude_km: float,
    kp: float,
    f10_7: float,
    cp_cg_offset_m: np.ndarray,
    drag_area_m2: float = 2.4,
    cd: float = 2.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute environmental disturbance torques (Gravity Gradient + Aerodynamic + SRP)
    and local geomagnetic field vector B_body (Tesla) in spacecraft body coordinates.
    """
    dcm = quaternion_to_dcm(q_ib)
    r_m = (EARTH_RADIUS_KM + altitude_km) * 1000.0
    mu_m3_s2 = MU_EARTH_KM3_S2 * 1e9

    # 1. Gravity-Gradient torque: tau_gg = (3*mu / r^3) * (r_hat_b x (I * r_hat_b))
    r_hat_i = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    r_hat_b = dcm @ r_hat_i
    tau_gg = (3.0 * mu_m3_s2 / (r_m**3)) * np.cross(r_hat_b, inertia_mat @ r_hat_b)

    # 2. Storm-responsive Aerodynamic Weathercock Torque
    rho = compute_atmospheric_density(altitude_km, kp=kp, f10_7=f10_7)
    v_orb_m_s = math.sqrt(mu_m3_s2 / r_m)
    v_hat_i = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    v_hat_b = dcm @ v_hat_i
    f_drag_b = -0.5 * rho * cd * drag_area_m2 * (v_orb_m_s**2) * v_hat_b
    tau_aero = np.cross(cp_cg_offset_m, f_drag_b)

    # 3. Solar Radiation Pressure torque
    p_srp = 4.56e-6  # N/m^2 at 1 AU
    s_hat_b = dcm @ np.array([0.577, 0.577, 0.577], dtype=np.float64)
    f_srp_b = -p_srp * 1.3 * drag_area_m2 * s_hat_b
    tau_srp = np.cross(cp_cg_offset_m * 0.7, f_srp_b)

    # Local dipolar Earth magnetic field in body frame (~30-45 uT in LEO)
    b0_tesla = 3.12e-5 * ((EARTH_RADIUS_KM / (EARTH_RADIUS_KM + altitude_km)) ** 3)
    b_inertial = np.array([0.35 * b0_tesla, 0.25 * b0_tesla, -0.90 * b0_tesla], dtype=np.float64)
    b_body = dcm @ b_inertial

    total_dist = tau_gg + tau_aero + tau_srp
    return total_dist, b_body


def simulate_attitude_slew(
    target_roll_deg: float = 0.0,
    target_pitch_deg: float = 35.0,
    target_yaw_deg: float = -20.0,
    initial_omega_deg_s: Tuple[float, float, float] = (0.4, -0.3, 0.2),
    altitude_km: float = 550.0,
    kp_storm: float = 6.0,
    f10_7: float = 180.0,
    duration_s: float = 90.0,
    dt_s: float = 0.5,
    config: Optional[SpacecraftADCSConfig] = None,
) -> ADCSSlewReport:
    """
    Simulate 6-DOF quaternion slew-to-burn maneuver with 4-wheel RWA pyramid
    and continuous magnetorquer cross-product momentum desaturation.
    """
    cfg = config or SpacecraftADCSConfig()
    I_mat = np.diag(np.asarray(cfg.inertia_diag_kg_m2, dtype=np.float64))
    I_inv = np.diag(1.0 / np.asarray(cfg.inertia_diag_kg_m2, dtype=np.float64))
    W, W_pinv = build_pyramid_rwa_matrix()
    cp_offset = np.asarray(cfg.cp_cg_offset_m, dtype=np.float64)

    q_curr = euler_to_quaternion(0.0, 0.0, 0.0)
    q_target = euler_to_quaternion(target_roll_deg, target_pitch_deg, target_yaw_deg)
    omega = np.radians(np.asarray(initial_omega_deg_s, dtype=np.float64))
    h_rw_4 = np.array([0.15, -0.12, 0.10, -0.13], dtype=np.float64)  # Initial stored wheel momentum (N*m*s)

    def _pointing_angle_deg(q_a: np.ndarray, q_b: np.ndarray) -> float:
        qe = quaternion_error(q_a, q_b)
        qw_clamped = min(1.0, max(-1.0, abs(float(qe[3]))))
        return math.degrees(2.0 * math.acos(qw_clamped))

    init_err_deg = _pointing_angle_deg(q_curr, q_target)
    steps = max(10, int(duration_s / dt_s))

    settling_time_s = duration_s
    settled = False
    peak_rpm = 0.0
    peak_h_nms = 0.0
    peak_tau_mnm = 0.0
    total_energy_joules = 0.0
    total_mtq_dumped_nms = 0.0
    dist_torques_unm: List[float] = []
    timeline: List[Dict[str, float]] = []

    for step in range(steps + 1):
        t = step * dt_s
        err_deg = _pointing_angle_deg(q_curr, q_target)
        omega_deg_s = float(np.degrees(np.linalg.norm(omega)))

        # Check settling criterion (< 0.1 deg pointing error and < 0.02 deg/s rate)
        if err_deg < 0.10 and omega_deg_s < 0.025:
            if not settled:
                settling_time_s = t
                settled = True
        else:
            settled = False
            settling_time_s = duration_s

        # Quaternion error & PD + gyroscopic decoupling law
        q_e = quaternion_error(q_curr, q_target)
        sign_w = 1.0 if q_e[3] >= 0.0 else -1.0
        q_e_vec = sign_w * q_e[:3]

        h_rw_body = W @ h_rw_4
        gyro_term = np.cross(omega, I_mat @ omega + h_rw_body)
        tau_desired_body = -cfg.kp_gain * q_e_vec - cfg.kd_gain * omega + gyro_term

        # Allocate across 4 pyramid reaction wheels and apply actuator saturation
        tau_rw_cmd_4 = W_pinv @ (-tau_desired_body)
        tau_rw_cmd_4 = np.clip(tau_rw_cmd_4, -cfg.max_wheel_torque_nm, cfg.max_wheel_torque_nm)

        # Prevent wheel momentum saturation
        for w_idx in range(4):
            if abs(h_rw_4[w_idx]) >= cfg.max_wheel_momentum_nms and (h_rw_4[w_idx] * tau_rw_cmd_4[w_idx] > 0):
                tau_rw_cmd_4[w_idx] = 0.0

        # Environmental disturbance & geomagnetic field
        tau_dist, b_body = compute_leo_disturbance_torques(
            q_ib=q_curr,
            inertia_mat=I_mat,
            altitude_km=altitude_km,
            kp=kp_storm,
            f10_7=f10_7,
            cp_cg_offset_m=cp_offset,
            drag_area_m2=cfg.drag_area_m2,
            cd=cfg.cd,
        )

        # Magnetorquer cross-product momentum dumping: m_dump = k * (h_rw_body x B) / ||B||^2
        b_norm_sq = float(np.dot(b_body, b_body))
        if b_norm_sq > 1e-16:
            m_cmd = cfg.mtq_dump_gain * np.cross(h_rw_body, b_body) / b_norm_sq
            m_cmd = np.clip(m_cmd, -cfg.max_mtq_dipole_am2, cfg.max_mtq_dipole_am2)
            tau_mtq = np.cross(m_cmd, b_body)
        else:
            tau_mtq = np.zeros(3, dtype=np.float64)

        # Record telemetry metrics
        wheel_omega_rad_s = h_rw_4 / cfg.wheel_inertia_kg_m2
        wheel_rpm = np.abs(wheel_omega_rad_s) * 60.0 / (2.0 * math.pi)
        curr_max_rpm = float(np.max(wheel_rpm))
        curr_max_h = float(np.max(np.abs(h_rw_4)))
        curr_tau_mnm = float(np.linalg.norm(W @ tau_rw_cmd_4) * 1000.0)
        curr_dist_unm = float(np.linalg.norm(tau_dist) * 1e6)

        peak_rpm = max(peak_rpm, curr_max_rpm)
        peak_h_nms = max(peak_h_nms, curr_max_h)
        peak_tau_mnm = max(peak_tau_mnm, curr_tau_mnm)
        dist_torques_unm.append(curr_dist_unm)

        if step % max(1, steps // 30) == 0 or step == steps:
            timeline.append(
                {
                    "time_s": round(t, 2),
                    "pointing_error_deg": round(err_deg, 4),
                    "omega_norm_deg_s": round(omega_deg_s, 4),
                    "max_wheel_rpm": round(curr_max_rpm, 1),
                    "rwa_torque_mnm": round(curr_tau_mnm, 2),
                    "disturbance_torque_unm": round(curr_dist_unm, 2),
                }
            )

        if step == steps:
            break

        # Electrical power: P = sum(|tau_i * omega_w_i|) + bus quiescent (6W) + MTQ power (2W)
        mech_power_w = float(np.sum(np.abs(tau_rw_cmd_4 * wheel_omega_rad_s))) + 8.0
        total_energy_joules += mech_power_w * dt_s
        total_mtq_dumped_nms += float(np.linalg.norm(tau_mtq)) * dt_s

        # RK4 integration of (q, omega)
        tau_ext = -(W @ tau_rw_cmd_4) + tau_mtq + tau_dist

        def _q_dot(q_val: np.ndarray, w_val: np.ndarray) -> np.ndarray:
            wx, wy, wz = w_val
            omega_mat = np.array(
                [
                    [0.0, wz, -wy, wx],
                    [-wz, 0.0, wx, wy],
                    [wy, -wx, 0.0, wz],
                    [-wx, -wy, -wz, 0.0],
                ],
                dtype=np.float64,
            )
            return 0.5 * (omega_mat @ q_val)

        def _w_dot(w_val: np.ndarray) -> np.ndarray:
            return I_inv @ (tau_ext - np.cross(w_val, I_mat @ w_val + h_rw_body))

        k1_q = _q_dot(q_curr, omega)
        k1_w = _w_dot(omega)

        k2_q = _q_dot(normalize_quaternion(q_curr + 0.5 * dt_s * k1_q), omega + 0.5 * dt_s * k1_w)
        k2_w = _w_dot(omega + 0.5 * dt_s * k1_w)

        k3_q = _q_dot(normalize_quaternion(q_curr + 0.5 * dt_s * k2_q), omega + 0.5 * dt_s * k2_w)
        k3_w = _w_dot(omega + 0.5 * dt_s * k2_w)

        k4_q = _q_dot(normalize_quaternion(q_curr + dt_s * k3_q), omega + dt_s * k3_w)
        k4_w = _w_dot(omega + dt_s * k3_w)

        q_curr = normalize_quaternion(q_curr + (dt_s / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q))
        omega = omega + (dt_s / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)
        h_rw_4 = h_rw_4 + (tau_rw_cmd_4 + W_pinv @ tau_mtq) * dt_s

    final_err_deg = _pointing_angle_deg(q_curr, q_target)
    final_err_arcsec = final_err_deg * 3600.0
    sat_pct = min(100.0, (peak_h_nms / cfg.max_wheel_momentum_nms) * 100.0)
    converged = final_err_deg < 0.25

    return ADCSSlewReport(
        target_euler_deg=(target_roll_deg, target_pitch_deg, target_yaw_deg),
        initial_error_deg=round(init_err_deg, 3),
        final_pointing_error_arcsec=round(final_err_arcsec, 2),
        settling_time_s=round(settling_time_s, 2),
        slew_converged=bool(converged),
        burn_attitude_ready=bool(converged and sat_pct < 90.0),
        peak_wheel_speed_rpm=round(peak_rpm, 1),
        peak_wheel_momentum_nms=round(peak_h_nms, 4),
        wheel_saturation_pct=round(sat_pct, 2),
        peak_rwa_torque_mnm=round(peak_tau_mnm, 2),
        total_adcs_energy_wh=round(total_energy_joules / 3600.0, 4),
        mtq_momentum_dumped_nms=round(total_mtq_dumped_nms, 5),
        mean_disturbance_torque_unm=round(float(np.mean(dist_torques_unm)), 3),
        timeline=timeline,
    )
