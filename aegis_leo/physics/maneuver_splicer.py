"""
Finite-Duration Maneuver Splicer & Powered Flight Integrator
Models continuous thrust acceleration, mass depletion rate m_dot = -T / (Isp * g0),
and splices pre-burn, powered arc, and post-burn ballistic trajectories.
"""

from typing import Dict, Any, Tuple
import numpy as np
from aegis_leo.constants import STANDARD_GRAVITY
from aegis_leo.physics.propagator import compute_gravitational_and_j2_acceleration
from aegis_leo.physics.atmosphere import compute_drag_acceleration


class ManeuverSplicer:
    """
    Simulates finite-thrust burn execution and generates continuous trajectory telemetry.
    """

    def __init__(
        self,
        wet_mass_kg: float = 260.0,
        thrust_newtons: float = 0.5,
        isp_seconds: float = 250.0,
        cd: float = 2.2,
        area_m2: float = 2.2,
    ):
        self.mass = wet_mass_kg
        self.thrust = thrust_newtons
        self.isp = isp_seconds
        self.cd = cd
        self.area = area_m2
        self.mdot = thrust_newtons / (isp_seconds * STANDARD_GRAVITY)  # kg / s

    def simulate_powered_burn(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        burn_duration_seconds: float,
        burn_direction_unit: np.ndarray,
        dt_seconds: float = 1.0,
        kp: float = 8.0,
        f10_7: float = 240.0,
    ) -> Dict[str, Any]:
        """
        Numerically integrate the powered flight arc during engine burn.
        """
        num_steps = int(burn_duration_seconds / dt_seconds) + 1
        times = np.linspace(0, burn_duration_seconds, num_steps)

        positions = np.zeros((num_steps, 3), dtype=float)
        velocities = np.zeros((num_steps, 3), dtype=float)
        masses = np.zeros(num_steps, dtype=float)

        curr_r = np.array(r0, dtype=float).copy()
        curr_v = np.array(v0, dtype=float).copy()
        curr_m = float(self.mass)

        positions[0] = curr_r
        velocities[0] = curr_v
        masses[0] = curr_m

        burn_dir = burn_direction_unit / np.linalg.norm(burn_direction_unit)

        for i in range(1, num_steps):
            # Thrust acceleration in km/s^2: (Thrust / mass) * 1e-3
            a_thrust_km_s2 = ((self.thrust / curr_m) * 1e-3) * burn_dir
            a_grav = compute_gravitational_and_j2_acceleration(curr_r)
            a_drag = compute_drag_acceleration(
                curr_r, curr_v, self.cd, self.area, curr_m, kp=kp, f10_7=f10_7
            )

            a_total = a_grav + a_drag + a_thrust_km_s2

            # RK2 / Midpoint step
            curr_v += a_total * dt_seconds
            curr_r += curr_v * dt_seconds
            curr_m = max(curr_m - self.mdot * dt_seconds, 1.0)

            positions[i] = curr_r
            velocities[i] = curr_v
            masses[i] = curr_m

        delta_v_delivered_m_s = float(
            np.linalg.norm(velocities[-1] - velocities[0]) * 1000.0
        )
        propellant_spent_kg = float(masses[0] - masses[-1])

        return {
            "burn_duration_s": burn_duration_seconds,
            "final_position": positions[-1],
            "final_velocity": velocities[-1],
            "final_mass_kg": masses[-1],
            "delta_v_delivered_m_s": delta_v_delivered_m_s,
            "propellant_spent_kg": propellant_spent_kg,
            "time_series": times,
            "positions": positions,
            "velocities": velocities,
        }
