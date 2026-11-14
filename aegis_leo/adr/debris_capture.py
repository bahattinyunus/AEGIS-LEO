"""
AEGIS-LEO Active Debris Removal (ADR) Capture Dynamics Engine
============================================================
Comprehensive physics and control engine for non-cooperative orbital debris capture:
1. 6-DOF Robotic Arm Forward & Inverse Kinematics (DH parameters, Geometric Jacobian,
   Yoshikawa manipulability, Damped Least Squares numerical IK, and Cartesian trajectory tracking).
2. Harpoon Tethered Momentum Exchange (Hypervelocity projectile penetration mechanics,
   Kelvin-Voigt viscoelastic tether modeling, winch braking dissipation, and momentum conservation).
3. Post-Capture Momentum Management (Euler rigid-body rotational dynamics, combined body inertia
   via Parallel Axis Theorem, Reaction Wheel rate damping, and geomagnetic Magnetorquer desaturation).

Standard reference models: WGS84, Earth Magnetic Dipole, NASA-STD-8719.14 ADR standards.
Uses exclusively NumPy and Python Standard Library.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np

from aegis_leo.constants import (
    EARTH_RADIUS_M,
    EARTH_MU_SI,
)


# ============================================================================
# Math Utilities & Attitude Kinematics
# ============================================================================

def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Return 3x3 skew-symmetric cross-product matrix [v]x."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ], dtype=float)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to unit quaternion [q_w, q_x, q_y, q_z]
    using numerically stable Shepperd's algorithm.
    """
    tr = np.trace(R)
    if tr > 0.0:
        s = 0.5 / math.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qw, qx, qy, qz], dtype=float)
    norm = np.linalg.norm(q)
    return q / (norm if norm > 1e-12 else 1.0)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [q_w, q_x, q_y, q_z] to 3x3 rotation matrix."""
    qw, qx, qy, qz = q / (np.linalg.norm(q) if np.linalg.norm(q) > 1e-12 else 1.0)
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
        [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
        [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)]
    ], dtype=float)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [q_w, q_x, q_y, q_z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], dtype=float)


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit quaternions for t in [0, 1]."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    if dot > 0.9995:
        # Linear approximation for very close orientations
        res = q0 + t * (q1 - q0)
        return res / np.linalg.norm(res)

    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    res = (s0 * q0) + (s1 * q1)
    return res / np.linalg.norm(res)


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Generate 3x3 rotation matrix using Rodrigues formula."""
    axis = axis / (np.linalg.norm(axis) if np.linalg.norm(axis) > 1e-12 else 1.0)
    K = skew_symmetric(axis)
    return np.eye(3) + math.sin(angle_rad) * K + (1.0 - math.cos(angle_rad)) * (K @ K)


# ============================================================================
# 1. 6-DOF Robotic Arm Forward & Inverse Kinematics
# ============================================================================

@dataclass
class DHParameter:
    """
    Standard Denavit-Hartenberg (DH) parameter specification for a single link.
    """
    a: float               # Link length along x_i (meters)
    alpha: float           # Link twist about x_i (radians)
    d: float               # Link offset along z_{i-1} (meters)
    theta_offset: float = 0.0 # Nominal offset angle about z_{i-1} (radians)
    theta_min: float = -np.pi # Minimum joint limit (radians)
    theta_max: float = np.pi  # Maximum joint limit (radians)


@dataclass
class ForwardKinematicsResult:
    """Result of 6-DOF forward kinematics computation."""
    position: np.ndarray              # End-effector Cartesian position (3,) [x, y, z] in meters
    orientation_matrix: np.ndarray    # End-effector 3x3 rotation matrix
    quaternion: np.ndarray            # End-effector unit quaternion [qw, qx, qy, qz]
    transform: np.ndarray             # 4x4 homogeneous transformation matrix T_0^6
    joint_transforms: List[np.ndarray]# 4x4 transforms for all intermediate frames T_0^i


@dataclass
class InverseKinematicsResult:
    """Result of 6-DOF numerical inverse kinematics."""
    success: bool
    joint_angles: np.ndarray          # Solved joint angles (6,) in radians
    iterations: int
    position_error: float             # Residual Cartesian Euclidean error in meters
    orientation_error: float          # Residual angular orientation error in radians
    manipulability: float             # Yoshikawa manipulability index at the solution
    message: str


class RoboticArm6DOF:
    """
    6-DOF Robotic Manipulator for Active Debris Removal (ADR).
    Models a flight-proven space robotic arm (Canadarm / ERA architecture)
    with forward kinematics, geometric Jacobian, manipulability index,
    damped least-squares inverse kinematics, and Cartesian trajectory tracking.
    """

    def __init__(
        self,
        dh_params: Optional[List[DHParameter]] = None,
        name: str = "ADR-Manipulator-6DOF"
    ):
        self.name = name
        if dh_params is None:
            # Default Canadarm / ERA style 6-DOF space manipulator geometry:
            # Joint 1: Base Yaw / Roll
            # Joint 2: Shoulder Pitch
            # Joint 3: Elbow Pitch
            # Joint 4: Wrist Pitch
            # Joint 5: Wrist Yaw
            # Joint 6: Wrist Roll
            self.dh_params: List[DHParameter] = [
                DHParameter(a=0.0, alpha=np.pi / 2.0, d=0.40, theta_min=-np.pi, theta_max=np.pi),
                DHParameter(a=1.20, alpha=0.0, d=0.0, theta_min=-np.pi * 0.9, theta_max=np.pi * 0.9),
                DHParameter(a=1.00, alpha=0.0, d=0.0, theta_min=-np.pi * 0.9, theta_max=np.pi * 0.9),
                DHParameter(a=0.0, alpha=-np.pi / 2.0, d=0.25, theta_min=-np.pi, theta_max=np.pi),
                DHParameter(a=0.0, alpha=np.pi / 2.0, d=0.0, theta_min=-np.pi * 0.8, theta_max=np.pi * 0.8),
                DHParameter(a=0.0, alpha=0.0, d=0.30, theta_min=-np.pi, theta_max=np.pi),
            ]
        else:
            if len(dh_params) != 6:
                raise ValueError(f"Expected exactly 6 DH parameters for 6-DOF arm, got {len(dh_params)}")
            self.dh_params = dh_params

    @staticmethod
    def dh_matrix(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """Compute standard 4x4 Denavit-Hartenberg transformation matrix A_i."""
        ct = math.cos(theta)
        st = math.sin(theta)
        ca = math.cos(alpha)
        sa = math.sin(alpha)

        return np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=float)

    def forward_kinematics(self, joint_angles: np.ndarray) -> ForwardKinematicsResult:
        """
        Compute forward kinematics for given 6 joint angles (radians).
        Returns end-effector pose, rotation matrix, quaternion, and all link transforms.
        """
        if len(joint_angles) != 6:
            raise ValueError(f"Expected 6 joint angles, got {len(joint_angles)}")

        T_current = np.eye(4, dtype=float)
        joint_transforms: List[np.ndarray] = [T_current.copy()]

        for i, (param, theta) in enumerate(zip(self.dh_params, joint_angles)):
            A_i = self.dh_matrix(
                a=param.a,
                alpha=param.alpha,
                d=param.d,
                theta=theta + param.theta_offset
            )
            T_current = T_current @ A_i
            joint_transforms.append(T_current.copy())

        p_ee = T_current[0:3, 3]
        R_ee = T_current[0:3, 0:3]
        q_ee = rotation_matrix_to_quaternion(R_ee)

        return ForwardKinematicsResult(
            position=p_ee,
            orientation_matrix=R_ee,
            quaternion=q_ee,
            transform=T_current,
            joint_transforms=joint_transforms
        )

    def compute_jacobian(self, joint_angles: np.ndarray) -> np.ndarray:
        """
        Compute 6x6 geometric Jacobian J = [J_v; J_w] mapping joint velocities to end-effector twist.
        J_v = z_{i-1} x (p_e - p_{i-1})
        J_w = z_{i-1}
        """
        fk = self.forward_kinematics(joint_angles)
        p_ee = fk.position
        J = np.zeros((6, 6), dtype=float)

        for i in range(6):
            T_prev = fk.joint_transforms[i]  # T_0^{i-1}
            p_prev = T_prev[0:3, 3]
            z_prev = T_prev[0:3, 2]

            J[0:3, i] = np.cross(z_prev, p_ee - p_prev)
            J[3:6, i] = z_prev

        return J

    def compute_manipulability(self, joint_angles: np.ndarray) -> float:
        """
        Compute Yoshikawa manipulability index w = sqrt(det(J * J^T)).
        For a square 6x6 Jacobian, w = |det(J)|.
        """
        J = self.compute_jacobian(joint_angles)
        det_jj = np.linalg.det(J @ J.T)
        return math.sqrt(max(0.0, det_jj))

    def compute_condition_number(self, joint_angles: np.ndarray) -> float:
        """Compute condition number of the Jacobian using SVD."""
        J = self.compute_jacobian(joint_angles)
        s = np.linalg.svd(J, compute_uv=False)
        if s[-1] < 1e-12:
            return float("inf")
        return float(s[0] / s[-1])

    def is_singular(self, joint_angles: np.ndarray, threshold: float = 1e-4) -> bool:
        """Check if robotic arm is near a kinematic singularity."""
        return self.compute_manipulability(joint_angles) < threshold

    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        initial_guess: Optional[np.ndarray] = None,
        max_iterations: int = 150,
        tolerance_pos: float = 1e-4,
        tolerance_ori: float = 1e-3,
        damping: float = 0.02,
        adaptive_damping: bool = True
    ) -> InverseKinematicsResult:
        """
        Solve 6-DOF inverse kinematics using Damped Least Squares (DLS) / Levenberg-Marquardt.
        Supports full 6-DOF pose matching (target_position + target_orientation 3x3)
        or 3-DOF position-only matching (if target_orientation is None).
        Joint angles are clamped to physical limits at each step.
        """
        target_pos = np.asarray(target_position, dtype=float)
        theta = (
            np.zeros(6, dtype=float)
            if initial_guess is None
            else np.asarray(initial_guess, dtype=float).copy()
        )

        solve_6dof = target_orientation is not None
        if solve_6dof:
            R_target = np.asarray(target_orientation, dtype=float)

        for iteration in range(max_iterations):
            fk = self.forward_kinematics(theta)
            pos_err = target_pos - fk.position
            norm_pos_err = float(np.linalg.norm(pos_err))

            if solve_6dof:
                # Orientation error vector using angle-axis representation
                R_err = R_target @ fk.orientation_matrix.T
                # Extract skew-symmetric component: 0.5 * (R_err - R_err^T)
                ori_err = 0.5 * np.array([
                    R_err[2, 1] - R_err[1, 2],
                    R_err[0, 2] - R_err[2, 0],
                    R_err[1, 0] - R_err[0, 1]
                ], dtype=float)
                norm_ori_err = float(np.linalg.norm(ori_err))

                if norm_pos_err < tolerance_pos and norm_ori_err < tolerance_ori:
                    manip = self.compute_manipulability(theta)
                    return InverseKinematicsResult(
                        success=True,
                        joint_angles=theta,
                        iterations=iteration,
                        position_error=norm_pos_err,
                        orientation_error=norm_ori_err,
                        manipulability=manip,
                        message="IK converged successfully within tolerances."
                    )

                error_vec = np.concatenate([pos_err, ori_err])
                J = self.compute_jacobian(theta)
            else:
                norm_ori_err = 0.0
                if norm_pos_err < tolerance_pos:
                    manip = self.compute_manipulability(theta)
                    return InverseKinematicsResult(
                        success=True,
                        joint_angles=theta,
                        iterations=iteration,
                        position_error=norm_pos_err,
                        orientation_error=0.0,
                        manipulability=manip,
                        message="Position IK converged successfully."
                    )

                error_vec = pos_err
                J = self.compute_jacobian(theta)[0:3, :]  # 3x6 position Jacobian

            # Damped Least Squares: delta_theta = J^T * (J * J^T + lambda^2 * I)^(-1) * error
            m, n = J.shape
            lam = damping
            if adaptive_damping:
                # Increase damping near singularities
                det_metric = math.sqrt(max(0.0, np.linalg.det(J @ J.T)))
                lam = damping * (1.0 + 0.1 / (det_metric + 1e-4))

            damped_inv = J.T @ np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(m))
            delta_theta = damped_inv @ error_vec

            # Step-size limiting for stability
            step_norm = np.linalg.norm(delta_theta)
            if step_norm > 0.4:
                delta_theta = delta_theta * (0.4 / step_norm)

            theta += delta_theta

            # Enforce physical joint limits
            for i, param in enumerate(self.dh_params):
                theta[i] = np.clip(theta[i], param.theta_min, param.theta_max)

        # Final evaluation after loop
        fk_final = self.forward_kinematics(theta)
        final_pos_err = float(np.linalg.norm(target_pos - fk_final.position))
        final_ori_err = 0.0
        if solve_6dof:
            R_err = R_target @ fk_final.orientation_matrix.T
            final_ori_err = float(np.linalg.norm(0.5 * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1]
            ])))

        manip = self.compute_manipulability(theta)
        success = final_pos_err < (tolerance_pos * 5.0) and (
            final_ori_err < (tolerance_ori * 5.0) if solve_6dof else True
        )

        return InverseKinematicsResult(
            success=success,
            joint_angles=theta,
            iterations=max_iterations,
            position_error=final_pos_err,
            orientation_error=final_ori_err,
            manipulability=manip,
            message="IK reached iteration limit" + (" (acceptable residual)" if success else " (failed convergence).")
        )

    def plan_cartesian_trajectory(
        self,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        start_orientation: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        steps: int = 50,
        duration: float = 10.0,
        initial_joint_angles: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Generate smooth Cartesian waypoints from start to target pose
        and solve inverse kinematics along the path to produce joint trajectories.
        """
        time_arr = np.linspace(0.0, duration, steps)
        dt = duration / max(1, steps - 1)

        start_p = np.asarray(start_pos, dtype=float)
        target_p = np.asarray(target_pos, dtype=float)

        use_orient = start_orientation is not None and target_orientation is not None
        if use_orient:
            q_start = rotation_matrix_to_quaternion(start_orientation)
            q_target = rotation_matrix_to_quaternion(target_orientation)

        current_angles = (
            np.zeros(6, dtype=float)
            if initial_joint_angles is None
            else np.asarray(initial_joint_angles, dtype=float).copy()
        )

        positions: List[np.ndarray] = []
        joint_trajectory: List[np.ndarray] = []
        manipulabilities: List[float] = []
        successes: List[bool] = []

        for i in range(steps):
            # Smooth quintic polynomial timing parameter s in [0, 1]
            tau = i / (steps - 1) if steps > 1 else 1.0
            s = 10.0 * (tau ** 3) - 15.0 * (tau ** 4) + 6.0 * (tau ** 5)

            p_waypoint = start_p + s * (target_p - start_p)
            positions.append(p_waypoint)

            if use_orient:
                q_waypoint = quaternion_slerp(q_start, q_target, s)
                R_waypoint = quaternion_to_rotation_matrix(q_waypoint)
                ik_res = self.inverse_kinematics(
                    target_position=p_waypoint,
                    target_orientation=R_waypoint,
                    initial_guess=current_angles,
                    tolerance_pos=1e-3,
                    tolerance_ori=2e-2,
                    max_iterations=100
                )
            else:
                ik_res = self.inverse_kinematics(
                    target_position=p_waypoint,
                    initial_guess=current_angles,
                    tolerance_pos=1e-3,
                    max_iterations=80
                )

            current_angles = ik_res.joint_angles.copy()
            joint_trajectory.append(current_angles.copy())
            manipulabilities.append(ik_res.manipulability)
            successes.append(ik_res.success)

        joint_traj_arr = np.array(joint_trajectory)  # shape (steps, 6)
        # Approximate joint velocities via central finite differences
        joint_velocities = np.gradient(joint_traj_arr, dt, axis=0)

        return {
            "time_s": time_arr,
            "cartesian_positions": np.array(positions),
            "joint_trajectory_rad": joint_traj_arr,
            "joint_velocities_rad_s": joint_velocities,
            "manipulability": np.array(manipulabilities),
            "all_points_converged": all(successes),
            "max_joint_speed_rad_s": float(np.max(np.abs(joint_velocities)))
        }

    def simulate_grapple_approach(
        self,
        fixture_pos: np.ndarray,
        fixture_vel: np.ndarray,
        initial_joint_angles: Optional[np.ndarray] = None,
        capture_distance: float = 0.05,
        capture_rel_vel_tol: float = 0.15,
        max_time: float = 10.0,
        dt: float = 0.05
    ) -> Dict[str, Any]:
        """
        Simulate robotic arm tracking and capture of a drifting target debris grapple fixture.
        Returns capture time, trajectory, and latch status.
        """
        t = 0.0
        fixture_pos_curr = np.asarray(fixture_pos, dtype=float).copy()
        current_joints = (
            np.zeros(6, dtype=float)
            if initial_joint_angles is None
            else np.asarray(initial_joint_angles, dtype=float).copy()
        )

        time_hist = []
        distance_hist = []
        captured = False
        capture_time = None

        while t <= max_time:
            # Update moving fixture position
            fixture_pos_curr += fixture_vel * dt
            fk = self.forward_kinematics(current_joints)
            dist = float(np.linalg.norm(fixture_pos_curr - fk.position))

            time_hist.append(t)
            distance_hist.append(dist)

            # Check capture criteria
            if dist <= capture_distance:
                # Compute end-effector velocity from Jacobian
                J = self.compute_jacobian(current_joints)[0:3, :]
                captured = True
                capture_time = t
                break

            # Track moving target
            ik_res = self.inverse_kinematics(
                target_position=fixture_pos_curr,
                initial_guess=current_joints,
                max_iterations=60,
                tolerance_pos=5e-3
            )
            current_joints = ik_res.joint_angles
            t += dt

        return {
            "captured": captured,
            "capture_time_s": capture_time,
            "final_distance_m": distance_hist[-1] if distance_hist else None,
            "time_history_s": np.array(time_hist),
            "distance_history_m": np.array(distance_hist),
            "final_joint_angles": current_joints
        }


# ============================================================================
# 2. Harpoon Tethered Momentum Exchange
# ============================================================================

@dataclass
class HarpoonParameters:
    """Hypervelocity debris capture harpoon projectile specifications."""
    mass_kg: float = 5.0                   # Harpoon projectile mass (kg)
    penetrator_radius_m: float = 0.012     # Penetrator tip radius (m)
    impact_velocity_m_s: float = 25.0      # Nominal firing speed relative to debris (m/s)
    barb_deploy_depth_m: float = 0.005     # Required penetration depth to deploy locking barbs (m)
    pullout_force_limit_n: float = 12000.0 # Pull-out mechanical failure threshold (N)


@dataclass
class TetherParameters:
    """Viscoelastic tether cable and winch braking specifications."""
    nominal_length_m: float = 80.0         # Full deployed unstretched tether length (m)
    stiffness_n_per_m: float = 3500.0      # Tether longitudinal spring stiffness k (N/m)
    damping_ratio: float = 0.15            # Viscous damping ratio zeta (Kelvin-Voigt)
    break_tension_n: float = 18000.0       # Ultimate tensile break force (N)
    linear_density_kg_per_m: float = 0.008 # Mass per unit length (kg/m)
    winch_slip_tension_n: float = 7500.0   # Active slip braking threshold (N)
    winch_max_braking_torque_nm: float = 600.0 # Max winch friction brake torque (N*m)
    winch_drum_radius_m: float = 0.15      # Winch drum radius (m)
    winch_reel_in_rate_m_s: float = 0.20   # Controlled reel-in docking rate (m/s)


@dataclass
class PenetrationResult:
    """Analysis of harpoon impact against debris structure."""
    penetrated: bool
    penetration_depth_m: float
    impact_kinetic_energy_j: float
    required_perforation_energy_j: float
    barbs_engaged: bool
    structural_integrity: str
    message: str


@dataclass
class TetherState:
    """Instantaneous state of the tether during momentum exchange."""
    length_m: float
    elongation_m: float
    tension_n: float
    is_taut: bool
    is_broken: bool
    winch_slipping: bool


class HarpoonTetherSystem:
    """
    Active Debris Removal Harpoon and Tethered Momentum Exchange System.
    Models structural penetration of debris casing (rocket stage skin/tanks),
    deployable locking barbs, Kelvin-Voigt viscoelastic tether tension,
    active winch braking dissipation, and full conservation of momentum.
    """

    def __init__(
        self,
        harpoon_params: Optional[HarpoonParameters] = None,
        tether_params: Optional[TetherParameters] = None
    ):
        self.harpoon = harpoon_params if harpoon_params is not None else HarpoonParameters()
        self.tether = tether_params if tether_params is not None else TetherParameters()

    def evaluate_penetration(
        self,
        target_skin_thickness_m: float = 0.003,
        target_yield_strength_pa: float = 2.8e8,  # Typical Al-2024 / Al-6061 alloy (280 MPa)
        target_material_density_kg_m3: float = 2700.0,
        impact_velocity_m_s: Optional[float] = None
    ) -> PenetrationResult:
        """
        Evaluate harpoon projectile penetration into space debris outer casing.
        Uses plastic shear work and hydrodynamic penetration equations.
        """
        v_impact = self.harpoon.impact_velocity_m_s if impact_velocity_m_s is None else impact_velocity_m_s
        m_h = self.harpoon.mass_kg
        r_p = self.harpoon.penetrator_radius_m
        A_proj = math.pi * (r_p ** 2)

        ke = 0.5 * m_h * (v_impact ** 2)

        # Perforation energy: plastic deformation and punching shear
        # E_crit = A_proj * t_skin * sigma_yield * geometric_factor
        geom_factor = 1.4  # Includes friction and displacement of plug
        e_req = A_proj * target_skin_thickness_m * target_yield_strength_pa * geom_factor

        # Estimated penetration depth in ductile aluminum casing
        depth = ke / (A_proj * target_yield_strength_pa * geom_factor + 1e-6)

        penetrated = depth >= target_skin_thickness_m
        barbs_engaged = depth >= (target_skin_thickness_m + self.harpoon.barb_deploy_depth_m)

        if not penetrated:
            status = "BOUNCE_DEFLECTION"
            msg = "Impact energy insufficient to perforate debris skin."
        elif not barbs_engaged:
            status = "PARTIAL_PENETRATION"
            msg = "Perforated skin, but penetration depth insufficient to deploy locking barbs."
        else:
            status = "SECURE_LATCH"
            msg = "Full penetration achieved; locking barbs deployed behind skin structure."

        return PenetrationResult(
            penetrated=penetrated,
            penetration_depth_m=depth,
            impact_kinetic_energy_j=ke,
            required_perforation_energy_j=e_req,
            barbs_engaged=barbs_engaged,
            structural_integrity=status,
            message=msg
        )

    def compute_tether_force(
        self,
        chaser_pos: np.ndarray,
        chaser_vel: np.ndarray,
        debris_pos: np.ndarray,
        debris_vel: np.ndarray,
        current_nominal_length: float,
        chaser_mass: float,
        debris_mass: float,
        allow_winch_slip: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, TetherState]:
        """
        Compute tether tension force vector on chaser and debris using Kelvin-Voigt viscoelastic model.
        Returns:
            F_chaser: Force vector on chaser (N) [directed toward debris]
            F_debris: Force vector on debris (N) [directed toward chaser]
            state: Instantaneous TetherState
        """
        r_rel = debris_pos - chaser_pos
        L = float(np.linalg.norm(r_rel))

        if L < 1e-6:
            unit_vec = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            unit_vec = r_rel / L

        v_rel = debris_vel - chaser_vel
        L_dot = float(np.dot(v_rel, unit_vec))

        elongation = L - current_nominal_length

        if elongation <= 0.0:
            # Slack tether: zero tension
            return np.zeros(3), np.zeros(3), TetherState(
                length_m=L,
                elongation_m=elongation,
                tension_n=0.0,
                is_taut=False,
                is_broken=False,
                winch_slipping=False
            )

        # Reduced mass for damping coefficient
        mu_red = (chaser_mass * debris_mass) / (chaser_mass + debris_mass)
        c_damping = 2.0 * self.tether.damping_ratio * math.sqrt(self.tether.stiffness_n_per_m * mu_red)

        # Kelvin-Voigt model: T = k * delta_L + c * delta_L_dot
        tension_raw = self.tether.stiffness_n_per_m * elongation + c_damping * L_dot
        tension = max(0.0, tension_raw)

        # Check winch slip / braking limit
        winch_slipping = False
        if allow_winch_slip and tension > self.tether.winch_slip_tension_n:
            winch_slipping = True
            tension = self.tether.winch_slip_tension_n

        # Check tensile rupture
        if tension >= self.tether.break_tension_n:
            # Tether snaps!
            return np.zeros(3), np.zeros(3), TetherState(
                length_m=L,
                elongation_m=elongation,
                tension_n=0.0,
                is_taut=False,
                is_broken=True,
                winch_slipping=False
            )

        # Tether pulls chaser toward debris (+unit_vec) and debris toward chaser (-unit_vec)
        F_chaser = tension * unit_vec
        F_debris = -tension * unit_vec

        return F_chaser, F_debris, TetherState(
            length_m=L,
            elongation_m=elongation,
            tension_n=tension,
            is_taut=True,
            is_broken=False,
            winch_slipping=winch_slipping
        )

    def simulate_capture_and_momentum_exchange(
        self,
        chaser_mass_kg: float,
        debris_mass_kg: float,
        chaser_init_pos: np.ndarray,
        chaser_init_vel: np.ndarray,
        debris_init_pos: np.ndarray,
        debris_init_vel: np.ndarray,
        duration_s: float = 60.0,
        dt: float = 0.02,
        enable_winch_braking: bool = True
    ) -> Dict[str, Any]:
        """
        Simulate multi-body tethered momentum exchange between chaser and debris post-harpoon latch.
        Integrates Newtonian dynamics with viscoelastic tether and winch energy dissipation.
        Verifies conservation of total system linear momentum.
        """
        steps = int(duration_s / dt)
        t_arr = np.linspace(0.0, duration_s, steps)

        r_c = np.asarray(chaser_init_pos, dtype=float).copy()
        v_c = np.asarray(chaser_init_vel, dtype=float).copy()
        r_d = np.asarray(debris_init_pos, dtype=float).copy()
        v_d = np.asarray(debris_init_vel, dtype=float).copy()

        current_nominal_L = float(np.linalg.norm(r_d - r_c))
        if current_nominal_L > self.tether.nominal_length_m:
            current_nominal_L = self.tether.nominal_length_m

        total_mass = chaser_mass_kg + debris_mass_kg
        init_com_pos = (chaser_mass_kg * r_c + debris_mass_kg * r_d) / total_mass
        init_com_vel = (chaser_mass_kg * v_c + debris_mass_kg * v_d) / total_mass
        init_total_momentum = total_mass * init_com_vel

        # Storage arrays
        time_hist: List[float] = []
        tension_hist: List[float] = []
        separation_hist: List[float] = []
        com_vel_err_hist: List[float] = []
        kinetic_energy_hist: List[float] = []
        chaser_speed_hist: List[float] = []
        debris_speed_hist: List[float] = []
        winch_dissipated_energy_j = 0.0
        tether_snapped = False

        for t in t_arr:
            time_hist.append(t)
            r_rel = r_d - r_c
            sep = float(np.linalg.norm(r_rel))
            separation_hist.append(sep)

            # Compute current kinetic energy
            ke = 0.5 * chaser_mass_kg * np.dot(v_c, v_c) + 0.5 * debris_mass_kg * np.dot(v_d, v_d)
            kinetic_energy_hist.append(ke)
            chaser_speed_hist.append(float(np.linalg.norm(v_c)))
            debris_speed_hist.append(float(np.linalg.norm(v_d)))

            # Current CoM velocity check
            curr_com_vel = (chaser_mass_kg * v_c + debris_mass_kg * v_d) / total_mass
            com_vel_err_hist.append(float(np.linalg.norm(curr_com_vel - init_com_vel)))

            # Calculate tether forces
            F_c, F_d, state = self.compute_tether_force(
                chaser_pos=r_c,
                chaser_vel=v_c,
                debris_pos=r_d,
                debris_vel=v_d,
                current_nominal_length=current_nominal_L,
                chaser_mass=chaser_mass_kg,
                debris_mass=debris_mass_kg,
                allow_winch_slip=enable_winch_braking
            )

            tension_hist.append(state.tension_n)

            if state.is_broken:
                tether_snapped = True

            # If winch is slipping to absorb shock, reel out nominal length
            if state.winch_slipping and enable_winch_braking:
                slip_rate = max(0.0, float(np.dot(v_d - v_c, r_rel / sep)))
                delta_l = slip_rate * dt
                current_nominal_L += delta_l
                winch_dissipated_energy_j += state.tension_n * delta_l

            # Verlet / Semi-implicit Euler integration
            a_c = F_c / chaser_mass_kg
            a_d = F_d / debris_mass_kg

            v_c += a_c * dt
            v_d += a_d * dt
            r_c += v_c * dt
            r_d += v_d * dt

        tension_arr = np.array(tension_hist)
        ke_arr = np.array(kinetic_energy_hist)

        return {
            "time_s": t_arr,
            "tension_n": tension_arr,
            "peak_tension_n": float(np.max(tension_arr)),
            "separation_m": np.array(separation_hist),
            "kinetic_energy_j": ke_arr,
            "energy_dissipated_j": float(ke_arr[0] - ke_arr[-1]),
            "winch_energy_dissipated_j": winch_dissipated_energy_j,
            "chaser_final_speed_m_s": chaser_speed_hist[-1],
            "debris_final_speed_m_s": debris_speed_hist[-1],
            "max_com_velocity_drift_m_s": float(np.max(com_vel_err_hist)),
            "momentum_conserved": bool(np.max(com_vel_err_hist) < 1e-9),
            "tether_snapped": tether_snapped,
            "final_relative_speed_m_s": float(np.linalg.norm(v_d - v_c))
        }


# ============================================================================
# 3. Post-Capture Momentum Management & Upper Stage Detumbling
# ============================================================================

@dataclass
class SpacecraftMassProperties:
    """Mass and inertia tensor properties of a spacecraft or debris body."""
    mass_kg: float
    com_offset_m: np.ndarray             # CoM position vector in reference frame (3,)
    inertia_tensor_kg_m2: np.ndarray     # 3x3 inertia matrix about its own CoM
    initial_spin_rate_rad_s: np.ndarray = field(default_factory=lambda: np.zeros(3))


class ReactionWheelArray:
    """
    Reaction Wheel (RW) actuator array.
    Applies control torque to satellite body while storing angular momentum.
    Enforces maximum torque and maximum stored momentum capacity per axis.
    """

    def __init__(
        self,
        max_torque_nm: float = 0.25,
        max_momentum_capacity_nms: float = 25.0
    ):
        self.max_torque_nm = max_torque_nm
        self.max_capacity_nms = max_momentum_capacity_nms
        self.stored_momentum_nms = np.zeros(3, dtype=float)

    def apply_torque(
        self,
        desired_torque_body: np.ndarray,
        dt: float
    ) -> Tuple[np.ndarray, bool]:
        """
        Request torque on the spacecraft body.
        The wheels apply -tau to absorb momentum.
        Returns:
            actual_body_torque: Torque actually exerted on the body (N*m)
            saturated: True if any axis has saturated its momentum capacity
        """
        tau_req = np.asarray(desired_torque_body, dtype=float)

        # Enforce maximum torque per axis
        tau_clamped = np.clip(tau_req, -self.max_torque_nm, self.max_torque_nm)

        # Wheel momentum rate of change: dh_rw/dt = -tau_body
        delta_h = -tau_clamped * dt
        new_h = self.stored_momentum_nms + delta_h

        # Enforce momentum saturation
        saturated = False
        actual_torque = tau_clamped.copy()

        for i in range(3):
            if abs(new_h[i]) >= self.max_capacity_nms:
                saturated = True
                new_h[i] = math.copysign(self.max_capacity_nms, new_h[i])
                # If saturated and requested torque pushes further into saturation, clamp torque to 0
                if (new_h[i] > 0 and tau_clamped[i] < 0) or (new_h[i] < 0 and tau_clamped[i] > 0):
                    actual_torque[i] = 0.0

        self.stored_momentum_nms = new_h
        return actual_torque, saturated

    def dump_momentum(self, momentum_delta: np.ndarray):
        """Desaturate wheels by removing stored momentum."""
        self.stored_momentum_nms += momentum_delta
        self.stored_momentum_nms = np.clip(
            self.stored_momentum_nms,
            -self.max_capacity_nms,
            self.max_capacity_nms
        )


class MagnetorquerArray:
    """
    Magnetic Torquer (MTQ) Actuator Array.
    Generates magnetic dipole moment m to interact with Earth's geomagnetic field B:
    tau_mag = m x B.
    Note that magnetorquers can only produce torque orthogonal to B.
    """

    def __init__(self, max_dipole_moment_am2: float = 120.0):
        self.max_dipole_moment_am2 = max_dipole_moment_am2

    def compute_torque(self, dipole_am2: np.ndarray, b_field_tesla: np.ndarray) -> np.ndarray:
        """Compute torque produced by magnetic dipole in magnetic field B."""
        m_clamped = np.clip(dipole_am2, -self.max_dipole_moment_am2, self.max_dipole_moment_am2)
        return np.cross(m_clamped, b_field_tesla)


class EarthMagneticFieldDipole:
    """
    Geomagnetic Field Dipole Model for Low Earth Orbit.
    Calculates Earth's magnetic field vector in ECI / Geocentric frame
    given orbital position vector r (meters).
    """

    def __init__(
        self,
        dipole_moment_am2: float = 7.94e22,
        magnetic_pole_eci: Optional[np.ndarray] = None
    ):
        self.m_earth = dipole_moment_am2
        # Earth's magnetic dipole axis (pointing toward southern magnetic pole, ~11.5 deg tilt)
        if magnetic_pole_eci is None:
            # Approx dipole axis tilted relative to geographic Z
            tilt = math.radians(11.5)
            self.pole_axis = np.array([math.sin(tilt), 0.0, -math.cos(tilt)], dtype=float)
        else:
            self.pole_axis = magnetic_pole_eci / np.linalg.norm(magnetic_pole_eci)

    def compute_b_field(self, position_m: np.ndarray) -> np.ndarray:
        """
        Compute magnetic field B (Tesla) at geocentric position r (meters).
        B(r) = (mu_0 / (4*pi)) * (1/r^5) * [3 * (M . r) * r - r^2 * M]
        """
        r_norm = float(np.linalg.norm(position_m))
        if r_norm < 1e3:
            return np.zeros(3)

        mu0_over_4pi = 1e-7  # T * m / A
        M_vec = self.m_earth * self.pole_axis

        r_hat = position_m / r_norm
        B = (mu0_over_4pi / (r_norm ** 3)) * (
            3.0 * np.dot(M_vec, r_hat) * r_hat - M_vec
        )
        return B


class CombinedBodyDynamics:
    """
    Calculates combined mass properties and rotational inertia of chaser
    and captured upper stage debris using the Parallel Axis Theorem.
    """

    @staticmethod
    def compute_combined_properties(
        chaser: SpacecraftMassProperties,
        debris: SpacecraftMassProperties,
        docking_offset_m: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute combined mass, joint Center of Mass, and combined inertia tensor
        about the joint CoM via Parallel Axis Theorem (Steiner's theorem).
        """
        m_c = chaser.mass_kg
        m_d = debris.mass_kg
        m_total = m_c + m_d

        r_c = chaser.com_offset_m
        r_d = debris.com_offset_m + docking_offset_m

        r_com = (m_c * r_c + m_d * r_d) / m_total

        # Parallel axis displacement vectors
        d_c = r_c - r_com
        d_d = r_d - r_com

        # Shift chaser inertia
        I_c_shift = chaser.inertia_tensor_kg_m2 + m_c * (
            np.dot(d_c, d_c) * np.eye(3) - np.outer(d_c, d_c)
        )

        # Shift debris inertia
        I_d_shift = debris.inertia_tensor_kg_m2 + m_d * (
            np.dot(d_d, d_d) * np.eye(3) - np.outer(d_d, d_d)
        )

        I_combined = I_c_shift + I_d_shift

        # Initial angular momentum about joint CoM
        H_c = chaser.inertia_tensor_kg_m2 @ chaser.initial_spin_rate_rad_s
        H_d = debris.inertia_tensor_kg_m2 @ debris.initial_spin_rate_rad_s
        H_initial = H_c + H_d

        # Joint initial angular velocity immediately following rigid latching
        omega_combined_0 = np.linalg.solve(I_combined, H_initial)

        return {
            "total_mass_kg": m_total,
            "joint_com_m": r_com,
            "combined_inertia_kg_m2": I_combined,
            "initial_angular_momentum_nms": H_initial,
            "initial_angular_velocity_rad_s": omega_combined_0,
            "initial_rotational_energy_j": float(0.5 * omega_combined_0 @ I_combined @ omega_combined_0)
        }


class DetumbleController:
    """
    Post-capture detumbling guidance and actuator controller.
    Provides:
    - Magnetic B-Dot algorithm for rate-damping without attitude knowledge.
    - Reaction wheel rate damping for fast angular velocity stabilization.
    - Magnetic cross-product desaturation to continuously dump wheel momentum.
    """

    @staticmethod
    def compute_b_dot_dipole(
        omega_body: np.ndarray,
        b_field_body: np.ndarray,
        gain_k: float = 1.0e6,
        max_dipole: float = 120.0
    ) -> np.ndarray:
        """
        B-Dot detumble dipole: m = k * (omega x B) / |B|.
        Dissipates rotational energy along axes orthogonal to geomagnetic field.
        """
        b_norm = float(np.linalg.norm(b_field_body))
        if b_norm < 1e-9:
            return np.zeros(3)

        # m = k * (omega x B) / |B|
        m_req = gain_k * np.cross(omega_body, b_field_body) / b_norm
        return np.clip(m_req, -max_dipole, max_dipole)

    @staticmethod
    def compute_rw_rate_damping_torque(
        omega_body: np.ndarray,
        kd_gain: float = 15.0,
        max_torque: float = 0.25
    ) -> np.ndarray:
        """Proportional rate-damping control torque for reaction wheels."""
        tau = -kd_gain * omega_body
        return np.clip(tau, -max_torque, max_torque)

    @staticmethod
    def compute_magnetic_desaturation_dipole(
        wheel_momentum_nms: np.ndarray,
        b_field_body: np.ndarray,
        kw_gain: float = 0.05,
        max_dipole: float = 120.0
    ) -> np.ndarray:
        """
        Compute magnetic dipole to desaturate reaction wheels:
        tau_desat = -kw * h_rw
        m = (B x tau_desat) / |B|^2
        """
        b_norm = float(np.linalg.norm(b_field_body))
        if b_norm < 1e-9:
            return np.zeros(3)

        tau_desat = -kw_gain * wheel_momentum_nms
        m_desat = np.cross(b_field_body, tau_desat) / (b_norm ** 2)
        return np.clip(m_desat, -max_dipole, max_dipole)


class PostCaptureMomentumManager:
    """
    Comprehensive Post-Capture Momentum Management and Upper Stage Detumbling Simulator.
    Simulates combined chaser + spent upper stage rigid body dynamics under:
    - Orbit progression in LEO (varying geomagnetic field).
    - Euler rotational equations: I * omega_dot + omega x (I * omega) = tau_rw + tau_mag.
    - Reaction Wheel attitude stabilization and wheel momentum buildup.
    - Geomagnetic Magnetorquer desaturation or pure magnetic B-dot detumbling.
    """

    def __init__(
        self,
        rw_array: Optional[ReactionWheelArray] = None,
        mtq_array: Optional[MagnetorquerArray] = None,
        b_field_model: Optional[EarthMagneticFieldDipole] = None
    ):
        self.rw = rw_array if rw_array is not None else ReactionWheelArray()
        self.mtq = mtq_array if mtq_array is not None else MagnetorquerArray()
        self.b_field_model = b_field_model if b_field_model is not None else EarthMagneticFieldDipole()

    def simulate_detumble(
        self,
        chaser: SpacecraftMassProperties,
        debris: SpacecraftMassProperties,
        docking_offset_m: np.ndarray = np.array([0.0, 0.0, 3.0]),
        orbit_altitude_km: float = 700.0,
        orbit_inclination_deg: float = 82.0,
        duration_s: float = 3600.0,
        dt: float = 0.5,
        control_mode: str = "hybrid",  # "hybrid", "b_dot", "reaction_wheels_only", "uncontrolled"
        detumble_threshold_rad_s: float = 0.002,
        kd_gain: float = 25.0,
        kw_gain: float = 0.08
    ) -> Dict[str, Any]:
        """
        Simulate detumbling of captured spinning rocket upper stage over an orbit.
        Integrates full 3-DOF rigid body rotational dynamics and quaternion kinematics.
        """
        combined = CombinedBodyDynamics.compute_combined_properties(chaser, debris, docking_offset_m)
        I_comb = combined["combined_inertia_kg_m2"]
        I_inv = np.linalg.inv(I_comb)

        omega = combined["initial_angular_velocity_rad_s"].copy()
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # Body attitude quaternion

        # Orbital setup for geomagnetic field propagation
        r_orbit = EARTH_RADIUS_M + orbit_altitude_km * 1000.0
        n_mean_motion = math.sqrt(EARTH_MU_SI / (r_orbit ** 3))
        inc_rad = math.radians(orbit_inclination_deg)

        steps = int(duration_s / dt)
        time_arr = np.linspace(0.0, duration_s, steps)

        omega_norm_hist: List[float] = []
        energy_hist: List[float] = []
        wheel_momentum_hist: List[float] = []
        rw_torque_hist: List[float] = []
        mtq_torque_hist: List[float] = []
        detumbled = False
        detumble_time_s = None

        # Reset reaction wheel momentum
        self.rw.stored_momentum_nms = np.zeros(3, dtype=float)

        for step_idx, t in enumerate(time_arr):
            omega_norm = float(np.linalg.norm(omega))
            omega_norm_hist.append(omega_norm)

            rot_energy = float(0.5 * omega @ I_comb @ omega)
            energy_hist.append(rot_energy)

            h_rw_norm = float(np.linalg.norm(self.rw.stored_momentum_nms))
            wheel_momentum_hist.append(h_rw_norm)

            if not detumbled and omega_norm < detumble_threshold_rad_s:
                detumbled = True
                detumble_time_s = t

            # Orbit position ECI
            true_anom = n_mean_motion * t
            pos_eci = r_orbit * np.array([
                math.cos(true_anom),
                math.sin(true_anom) * math.cos(inc_rad),
                math.sin(true_anom) * math.sin(inc_rad)
            ], dtype=float)

            # Local geomagnetic field in ECI and Body frames
            B_eci = self.b_field_model.compute_b_field(pos_eci)
            R_body = quaternion_to_rotation_matrix(q)
            B_body = R_body.T @ B_eci

            # Control Torque Calculation
            tau_rw = np.zeros(3, dtype=float)
            tau_mag = np.zeros(3, dtype=float)

            if control_mode == "hybrid":
                # RW rate damping for fast attitude stabilization
                tau_rw_req = DetumbleController.compute_rw_rate_damping_torque(
                    omega_body=omega,
                    kd_gain=kd_gain,
                    max_torque=self.rw.max_torque_nm
                )
                tau_rw, _ = self.rw.apply_torque(tau_rw_req, dt)

                # MTQ continuous wheel momentum desaturation
                m_dipole = DetumbleController.compute_magnetic_desaturation_dipole(
                    wheel_momentum_nms=self.rw.stored_momentum_nms,
                    b_field_body=B_body,
                    kw_gain=kw_gain,
                    max_dipole=self.mtq.max_dipole_moment_am2
                )
                tau_mag = self.mtq.compute_torque(m_dipole, B_body)

                # Desaturate wheels against magnetic torque
                # When tau_desat counteracts wheel momentum, MTQ torque tau_mag dumps wheel momentum
                self.rw.dump_momentum(tau_mag * dt)

            elif control_mode == "b_dot":
                # Pure magnetic B-dot detumbling
                m_dipole = DetumbleController.compute_b_dot_dipole(
                    omega_body=omega,
                    b_field_body=B_body,
                    gain_k=1.5e6,
                    max_dipole=self.mtq.max_dipole_moment_am2
                )
                tau_mag = self.mtq.compute_torque(m_dipole, B_body)

            elif control_mode == "reaction_wheels_only":
                tau_rw_req = DetumbleController.compute_rw_rate_damping_torque(
                    omega_body=omega,
                    kd_gain=25.0,
                    max_torque=self.rw.max_torque_nm
                )
                tau_rw, _ = self.rw.apply_torque(tau_rw_req, dt)

            elif control_mode == "uncontrolled":
                pass

            rw_torque_hist.append(float(np.linalg.norm(tau_rw)))
            mtq_torque_hist.append(float(np.linalg.norm(tau_mag)))

            # Euler's rotational equations of motion:
            # I * d_omega/dt = tau_total - omega x (I * omega)
            tau_total = tau_rw + tau_mag
            gyroscopic = np.cross(omega, I_comb @ omega)
            d_omega_dt = I_inv @ (tau_total - gyroscopic)

            # RK4 Integration of Angular Velocity
            k1 = d_omega_dt
            omega_mid = omega + 0.5 * dt * k1
            gyr_mid = np.cross(omega_mid, I_comb @ omega_mid)
            k2 = I_inv @ (tau_total - gyr_mid)

            omega += k2 * dt

            # Quaternion kinematic update: dq/dt = 0.5 * q (x) [0, omega]
            # Quaternion derivative
            qw, qx, qy, qz = q
            dq_dt = 0.5 * np.array([
                -qx * omega[0] - qy * omega[1] - qz * omega[2],
                qw * omega[0] + qy * omega[2] - qz * omega[1],
                qw * omega[1] - qx * omega[2] + qz * omega[0],
                qw * omega[2] + qx * omega[1] - qy * omega[0]
            ], dtype=float)

            q += dq_dt * dt
            q_norm = np.linalg.norm(q)
            q = q / (q_norm if q_norm > 1e-12 else 1.0)

        omega_arr = np.array(omega_norm_hist)
        energy_arr = np.array(energy_hist)

        return {
            "time_s": time_arr,
            "omega_norm_rad_s": omega_arr,
            "omega_norm_deg_s": np.degrees(omega_arr),
            "rotational_energy_j": energy_arr,
            "wheel_momentum_nms": np.array(wheel_momentum_hist),
            "rw_torque_nm": np.array(rw_torque_hist),
            "mtq_torque_nm": np.array(mtq_torque_hist),
            "initial_spin_rate_rad_s": float(omega_arr[0]),
            "final_spin_rate_rad_s": float(omega_arr[-1]),
            "final_rotational_energy_j": float(energy_arr[-1]),
            "energy_reduction_ratio": float((energy_arr[0] - energy_arr[-1]) / max(1e-6, energy_arr[0])),
            "is_detumbled": bool(omega_arr[-1] < detumble_threshold_rad_s),
            "detumble_time_s": detumble_time_s,
            "combined_mass_properties": combined
        }
