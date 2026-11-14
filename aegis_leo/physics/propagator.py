"""
Orbital Propagation Engine
Includes high-precision RK4 numerical integration with J2 and dynamic atmospheric drag,
plus SGP4 analytic propagation for NORAD TLE catalog objects.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from datetime import datetime, timezone
try:
    from sgp4.api import Satrec, jday
    HAS_SGP4 = True
except ImportError:
    HAS_SGP4 = False

from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    EARTH_MU,
    EARTH_J2,
    QUIET_KP,
    QUIET_F10_7,
)
from aegis_leo.physics.atmosphere import compute_drag_acceleration
from aegis_leo.physics.coordinates import cartesian_to_kepler, eci_to_geodetic


def compute_gravitational_and_j2_acceleration(r_eci: np.ndarray) -> np.ndarray:
    """
    Calculate primary two-body gravity + Earth J2 oblateness acceleration (km/s^2).
    """
    x, y, z = r_eci[0], r_eci[1], r_eci[2]
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    if r < 1.0:
        return np.zeros(3)

    # Two-body point mass acceleration
    mu_over_r3 = EARTH_MU / (r2 * r)
    a_grav = -mu_over_r3 * r_eci

    # J2 zonal harmonic perturbation
    r5 = r2 * r2 * r
    z2_over_r2 = (z**2) / r2
    factor_j2 = 1.5 * EARTH_J2 * EARTH_MU * (EARTH_RADIUS_KM**2) / r5

    a_j2_x = factor_j2 * x * (5.0 * z2_over_r2 - 1.0)
    a_j2_y = factor_j2 * y * (5.0 * z2_over_r2 - 1.0)
    a_j2_z = factor_j2 * z * (5.0 * z2_over_r2 - 3.0)
    a_j2 = np.array([a_j2_x, a_j2_y, a_j2_z], dtype=float)

    return a_grav + a_j2


def rk4_step(
    r: np.ndarray,
    v: np.ndarray,
    dt: float,
    cd: float,
    area_m2: float,
    mass_kg: float,
    kp: float,
    f10_7: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute a single 4th-order Runge-Kutta numerical step.
    State vector: [r, v]
    """
    def derivatives(pos: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dr_dt = vel
        a_grav = compute_gravitational_and_j2_acceleration(pos)
        a_drag = compute_drag_acceleration(pos, vel, cd, area_m2, mass_kg, kp, f10_7)
        dv_dt = a_grav + a_drag
        return dr_dt, dv_dt

    # RK4 coefficients
    k1_r, k1_v = derivatives(r, v)
    k2_r, k2_v = derivatives(r + 0.5 * dt * k1_r, v + 0.5 * dt * k1_v)
    k3_r, k3_v = derivatives(r + 0.5 * dt * k2_r, v + 0.5 * dt * k2_v)
    k4_r, k4_v = derivatives(r + dt * k3_r, v + dt * k3_v)

    r_next = r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
    v_next = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return r_next, v_next


class OrbitalPropagator:
    """
    Multi-fidelity orbital state propagator supporting both Cowell numerical integration
    with dynamic space weather drag and SGP4 analytic propagation.
    """

    def __init__(
        self,
        cd: float = 2.2,
        area_m2: float = 1.5,
        mass_kg: float = 120.0,
    ):
        self.cd = float(cd)
        self.area_m2 = float(area_m2)
        self.mass_kg = float(mass_kg)

    def propagate_numerical(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        duration_hours: float,
        step_seconds: float = 30.0,
        kp: float = QUIET_KP,
        f10_7: float = QUIET_F10_7,
    ) -> Dict[str, Any]:
        """
        Numerically integrate the trajectory using RK4 under specified space weather conditions.
        Returns time history of positions, velocities, altitudes, and keplerian elements.
        """
        total_seconds = float(duration_hours * 3600.0)
        num_steps = int(total_seconds / step_seconds) + 1

        t_series = np.linspace(0.0, total_seconds, num_steps)
        positions = np.zeros((num_steps, 3), dtype=float)
        velocities = np.zeros((num_steps, 3), dtype=float)
        altitudes = np.zeros(num_steps, dtype=float)

        curr_r = np.array(r0, dtype=float).copy()
        curr_v = np.array(v0, dtype=float).copy()

        positions[0] = curr_r
        velocities[0] = curr_v
        altitudes[0] = np.linalg.norm(curr_r) - EARTH_RADIUS_KM

        for i in range(1, num_steps):
            curr_r, curr_v = rk4_step(
                curr_r,
                curr_v,
                step_seconds,
                self.cd,
                self.area_m2,
                self.mass_kg,
                kp,
                f10_7,
            )
            positions[i] = curr_r
            velocities[i] = curr_v
            altitudes[i] = np.linalg.norm(curr_r) - EARTH_RADIUS_KM

        final_kepler = cartesian_to_kepler(curr_r, curr_v)

        return {
            "time_seconds": t_series,
            "positions_km": positions,
            "velocities_km_s": velocities,
            "altitudes_km": altitudes,
            "final_position": curr_r,
            "final_velocity": curr_v,
            "final_altitude_km": float(altitudes[-1]),
            "final_kepler": final_kepler,
            "altitude_decay_meters": float((altitudes[0] - altitudes[-1]) * 1000.0),
        }

    def simulate_storm_divergence(
        self,
        r0: np.ndarray,
        v0: np.ndarray,
        duration_hours: float = 36.0,
        step_seconds: float = 60.0,
        storm_kp: float = 8.0,
        storm_f10_7: float = 240.0,
    ) -> Dict[str, Any]:
        """
        Quantify the along-track position divergence and orbital decay between
        a nominal (quiet) CDM forecast vs actual geomagnetic storm expansion.
        """
        # Baseline quiet forecast (standard operational prediction)
        res_quiet = self.propagate_numerical(
            r0, v0, duration_hours, step_seconds, kp=QUIET_KP, f10_7=QUIET_F10_7
        )

        # Actual trajectory under geomagnetic storm
        res_storm = self.propagate_numerical(
            r0, v0, duration_hours, step_seconds, kp=storm_kp, f10_7=storm_f10_7
        )

        # Compute displacements at final epoch
        pos_quiet = res_quiet["final_position"]
        pos_storm = res_storm["final_position"]
        vel_quiet = res_quiet["final_velocity"]

        # Radial, In-Track, Cross-Track (RIC) frame relative to quiet trajectory
        r_unit = pos_quiet / np.linalg.norm(pos_quiet)
        h_vec = np.cross(pos_quiet, vel_quiet)
        c_unit = h_vec / np.linalg.norm(h_vec)
        i_unit = np.cross(c_unit, r_unit)  # Along-track unit vector

        delta_r_vec = pos_storm - pos_quiet
        radial_disp_km = float(np.dot(delta_r_vec, r_unit))
        intrack_disp_km = float(np.dot(delta_r_vec, i_unit))
        crosstrack_disp_km = float(np.dot(delta_r_vec, c_unit))
        total_miss_km = float(np.linalg.norm(delta_r_vec))

        decay_diff_meters = float(
            (res_quiet["final_altitude_km"] - res_storm["final_altitude_km"]) * 1000.0
        )

        return {
            "duration_hours": duration_hours,
            "quiet_altitude_decay_m": res_quiet["altitude_decay_meters"],
            "storm_altitude_decay_m": res_storm["altitude_decay_meters"],
            "net_decay_meters": decay_diff_meters,
            "radial_displacement_km": radial_disp_km,
            "along_track_displacement_km": intrack_disp_km,
            "cross_track_displacement_km": crosstrack_disp_km,
            "total_divergence_km": total_miss_km,
            "quiet_history": res_quiet,
            "storm_history": res_storm,
        }

    @staticmethod
    def propagate_tle(
        line1: str,
        line2: str,
        epoch_datetime: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Analytic SGP4 propagation from Two-Line Element (TLE).
        """
        if not HAS_SGP4:
            raise RuntimeError("sgp4 package is required for TLE propagation.")

        sat = Satrec.twoline2rv(line1, line2)
        dt = epoch_datetime or datetime.now(timezone.utc)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6)

        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            raise ValueError(f"SGP4 error code {e} during propagation.")

        r_vec = np.array(r, dtype=float)
        v_vec = np.array(v, dtype=float)
        alt_km = float(np.linalg.norm(r_vec) - EARTH_RADIUS_KM)

        return {
            "epoch": dt.isoformat(),
            "position_km": r_vec,
            "velocity_km_s": v_vec,
            "altitude_km": alt_km,
            "satnum": sat.satnum,
            "bstar": sat.bstar,
            "inclination_deg": float(np.degrees(sat.inclo)),
            "eccentricity": float(sat.ecco),
            "mean_motion_rev_day": float(sat.no_kozai * 1440.0 / (2.0 * np.pi)),
        }
