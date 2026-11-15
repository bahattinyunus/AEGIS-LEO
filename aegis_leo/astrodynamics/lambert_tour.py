"""
Lambert Rendezvous Solver & Multi-Target Active Debris Removal (ADR) Tour Planner
=================================================================================

Implements:
  1. Universal Variable (Bate-Mueller-White / Vallado Stumpff C(z), S(z)) Lambert
     two-point boundary value solver (r1, r2, tof_s -> v1_trans, v2_trans).
  2. Secular J2 Nodal Precession (RAAN drift rate) & Coplanar Phasing Estimator
     for low-fuel orbital plane matching in LEO.
  3. Combinatorial Multi-Target ADR Tour Optimizer: sequences rendezvous and
     deorbit burns across multiple high-risk LEO debris objects within the
     servicer spacecraft's propellant budget (Tsiolkovsky rocket equation).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aegis_leo.constants import EARTH_RADIUS_KM, G0_M_S2, J2_EARTH, MU_EARTH_KM3_S2


def stumpff_c(z: float) -> float:
    """Stumpff function C(z) for universal variable formulation."""
    if z > 1e-6:
        sz = math.sqrt(z)
        return (1.0 - math.cos(sz)) / z
    if z < -1e-6:
        sz = math.sqrt(-z)
        return (math.cosh(sz) - 1.0) / (-z)
    return 0.5 - z / 24.0 + (z * z) / 720.0


def stumpff_s(z: float) -> float:
    """Stumpff function S(z) for universal variable formulation."""
    if z > 1e-6:
        sz = math.sqrt(z)
        return (sz - math.sin(sz)) / (sz**3)
    if z < -1e-6:
        sz = math.sqrt(-z)
        return (math.sinh(sz) - sz) / (sz**3)
    return 1.0 / 6.0 - z / 120.0 + (z * z) / 5040.0


@dataclass
class LambertSolution:
    v1_km_s: np.ndarray
    v2_km_s: np.ndarray
    tof_s: float
    transfer_angle_deg: float
    semi_major_axis_km: float
    converged: bool
    iterations: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v1_km_s": [round(float(x), 5) for x in self.v1_km_s],
            "v2_km_s": [round(float(x), 5) for x in self.v2_km_s],
            "tof_s": round(self.tof_s, 2),
            "transfer_angle_deg": round(self.transfer_angle_deg, 3),
            "semi_major_axis_km": round(self.semi_major_axis_km, 2),
            "converged": self.converged,
            "iterations": self.iterations,
        }


def solve_lambert_universal(
    r1_km: np.ndarray,
    r2_km: np.ndarray,
    tof_s: float,
    mu_km3_s2: float = MU_EARTH_KM3_S2,
    prograde: bool = True,
    max_iter: int = 80,
    tol: float = 1e-7,
) -> LambertSolution:
    """
    Solve Lambert's orbital boundary value problem using the Universal Variable
    and Stumpff functions C(z), S(z) via damped Newton-Raphson / bisection hybrid.
    """
    r1 = np.asarray(r1_km, dtype=np.float64).flatten()
    r2 = np.asarray(r2_km, dtype=np.float64).flatten()
    if tof_s <= 0.0:
        raise ValueError("Time of flight tof_s must be strictly positive.")

    r1_norm = float(np.linalg.norm(r1))
    r2_norm = float(np.linalg.norm(r2))
    cos_dnu = float(np.clip(np.dot(r1, r2) / (r1_norm * r2_norm), -1.0, 1.0))
    cross_z = float(np.cross(r1, r2)[2])

    if prograde:
        dnu = math.acos(cos_dnu) if cross_z >= 0.0 else (2.0 * math.pi - math.acos(cos_dnu))
    else:
        dnu = math.acos(cos_dnu) if cross_z < 0.0 else (2.0 * math.pi - math.acos(cos_dnu))

    # Guard against exact 0 or 180 deg singularity
    if abs(math.sin(dnu)) < 1e-6:
        dnu += 1e-4

    A = math.sin(dnu) * math.sqrt(r1_norm * r2_norm / max(1e-9, 1.0 - math.cos(dnu)))
    sqrt_mu = math.sqrt(mu_km3_s2)

    def _y(z_val: float) -> float:
        cz = stumpff_c(z_val)
        sz = stumpff_s(z_val)
        if cz < 1e-12:
            return 1e9
        return r1_norm + r2_norm + A * (z_val * sz - 1.0) / math.sqrt(cz)

    # Bracket z in [-4*pi, (2*pi)^2]
    z_low = -4.0 * math.pi
    z_high = 4.0 * (math.pi**2) - 0.01
    while _y(z_low) < 0.0 and z_low < z_high:
        z_low += 0.25

    z = 0.0
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        yz = _y(z)
        if yz <= 0.0:
            z_low = z
            z = 0.5 * (z_low + z_high)
            continue

        cz = stumpff_c(z)
        sz = stumpff_s(z)
        xz = math.sqrt(yz / cz)
        tof_calc = ((xz**3) * sz + A * math.sqrt(yz)) / sqrt_mu
        err = tof_calc - tof_s
        if abs(err) < tol:
            converged = True
            break

        if err < 0.0:
            z_low = max(z_low, z)
        else:
            z_high = min(z_high, z)

        # Derivative d(tof)/dz
        if abs(z) > 1e-5:
            dtof_dz = (
                (xz**3) * (stumpff_c(z) - 1.5 * sz / cz) / (2.0 * z)
                + (A / 8.0) * (3.0 * sz * math.sqrt(yz) / cz + A / xz)
            ) / sqrt_mu
        else:
            y0 = _y(0.0)
            dtof_dz = (math.sqrt(2.0) / 40.0 * (y0**1.5) + (A / 8.0) * (math.sqrt(y0) + A * math.sqrt(1.0 / (2.0 * y0)))) / sqrt_mu

        z_next = z - err / dtof_dz if abs(dtof_dz) > 1e-12 else 0.5 * (z_low + z_high)
        if z_next <= z_low or z_next >= z_high:
            z = 0.5 * (z_low + z_high)
        else:
            z = z_next

    yz = max(1e-8, _y(z))
    f = 1.0 - yz / r1_norm
    g = A * math.sqrt(yz / mu_km3_s2)
    g_dot = 1.0 - yz / r2_norm

    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g

    spec_energy = 0.5 * float(np.dot(v1, v1)) - mu_km3_s2 / r1_norm
    sma = -mu_km3_s2 / (2.0 * spec_energy) if abs(spec_energy) > 1e-9 else 1e6

    return LambertSolution(
        v1_km_s=v1,
        v2_km_s=v2,
        tof_s=tof_s,
        transfer_angle_deg=math.degrees(dnu),
        semi_major_axis_km=sma,
        converged=converged,
        iterations=it,
    )


def compute_j2_raan_drift_rate_deg_day(altitude_km: float, inclination_deg: float, ecc: float = 0.001) -> float:
    """
    Compute secular J2 nodal precession rate d(RAAN)/dt in degrees per day.
    """
    a_km = EARTH_RADIUS_KM + altitude_km
    n_rad_s = math.sqrt(MU_EARTH_KM3_S2 / (a_km**3))
    p_km = a_km * (1.0 - ecc * ecc)
    inc_rad = math.radians(inclination_deg)
    omega_dot_rad_s = -1.5 * n_rad_s * J2_EARTH * ((EARTH_RADIUS_KM / p_km) ** 2) * math.cos(inc_rad)
    return math.degrees(omega_dot_rad_s) * 86400.0


@dataclass
class DebrisTourTarget:
    target_id: str
    name: str
    altitude_km: float
    inclination_deg: float
    raan_deg: float
    mass_kg: float
    kessler_priority_score: float


DEFAULT_ADR_TOUR_TARGETS: List[DebrisTourTarget] = [
    DebrisTourTarget("DEB-01", "COSMOS-2251-CORE", 785.0, 74.0, 42.0, 900.0, 9.6),
    DebrisTourTarget("DEB-02", "SL-16-UPPER-STAGE", 830.0, 71.0, 45.2, 1400.0, 9.8),
    DebrisTourTarget("DEB-03", "IRIDIUM-33-BUS", 775.0, 86.4, 48.0, 560.0, 8.9),
    DebrisTourTarget("DEB-04", "FENGYUN-1C-FRAG", 845.0, 98.6, 52.5, 420.0, 9.2),
    DebrisTourTarget("DEB-05", "METEOR-2-ROCKET", 810.0, 81.2, 44.0, 1150.0, 9.4),
]


@dataclass
class ADRTourLeg:
    leg_index: int
    target_id: str
    target_name: str
    from_altitude_km: float
    to_altitude_km: float
    raan_drift_wait_days: float
    lambert_rendezvous_dv_m_s: float
    deorbit_kit_impulse_dv_m_s: float
    total_leg_dv_m_s: float
    propellant_used_kg: float
    remaining_fuel_kg: float
    kessler_risk_removed: float


@dataclass
class MultiDebrisTourReport:
    servicer_dry_mass_kg: float
    initial_fuel_kg: float
    remaining_fuel_kg: float
    isp_s: float
    targets_visited: int
    total_delta_v_m_s: float
    total_fuel_consumed_kg: float
    total_mission_duration_days: float
    total_kessler_risk_removed: float
    sample_lambert_transfer: Dict[str, Any]
    legs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def plan_multi_debris_adr_tour(
    servicer_dry_mass_kg: float = 450.0,
    initial_fuel_kg: float = 180.0,
    isp_s: float = 320.0,
    start_altitude_km: float = 750.0,
    start_inclination_deg: float = 74.0,
    start_raan_deg: float = 40.0,
    max_targets: int = 4,
    targets: Optional[List[DebrisTourTarget]] = None,
) -> MultiDebrisTourReport:
    """
    Optimize a multi-target Active Debris Removal (ADR) orbital tour using
    J2 nodal drift phasing + Universal-Variable Lambert rendezvous transfers.
    """
    pool = list(targets) if targets is not None else list(DEFAULT_ADR_TOUR_TARGETS)
    curr_alt = start_altitude_km
    curr_inc = start_inclination_deg
    curr_raan = start_raan_deg
    rem_fuel = initial_fuel_kg
    v_exh = isp_s * G0_M_S2

    visited_legs: List[Dict[str, Any]] = []
    total_dv = 0.0
    total_days = 0.0
    total_risk_removed = 0.0
    sample_lambert_dict: Dict[str, Any] = {}

    unvisited = pool.copy()

    for leg_idx in range(1, min(max_targets, len(pool)) + 1):
        best_candidate: Optional[DebrisTourTarget] = None
        best_eval: Optional[Tuple[float, float, float, float, LambertSolution]] = None
        best_cost_per_risk = 1e12

        for cand in unvisited:
            # 1. Compute J2 differential nodal drift wait time vs active RAAN trim
            drift_curr = compute_j2_raan_drift_rate_deg_day(curr_alt, curr_inc)
            drift_cand = compute_j2_raan_drift_rate_deg_day(cand.altitude_km, cand.inclination_deg)
            diff_drift = abs(drift_curr - drift_cand)
            delta_raan = abs((cand.raan_deg - curr_raan + 180.0) % 360.0 - 180.0)

            wait_days = min(25.0, delta_raan / max(0.15, diff_drift))
            residual_raan_deg = max(0.0, delta_raan - diff_drift * wait_days)

            # 2. Construct ECI departure r1 and arrival r2 for half-orbit Lambert transfer
            r1_mag = EARTH_RADIUS_KM + curr_alt
            r2_mag = EARTH_RADIUS_KM + cand.altitude_km
            a_trans = 0.5 * (r1_mag + r2_mag)
            tof_lambert_s = math.pi * math.sqrt((a_trans**3) / MU_EARTH_KM3_S2) * 0.88

            r1_vec = np.array([r1_mag, 0.0, 0.0], dtype=np.float64)
            ang = math.radians(155.0)
            inc_diff_rad = math.radians( (cand.inclination_deg - curr_inc) * 0.18 + residual_raan_deg * 0.12 )
            r2_vec = np.array(
                [
                    r2_mag * math.cos(ang),
                    r2_mag * math.sin(ang) * math.cos(inc_diff_rad),
                    r2_mag * math.sin(ang) * math.sin(inc_diff_rad),
                ],
                dtype=np.float64,
            )

            lam = solve_lambert_universal(r1_vec, r2_vec, tof_lambert_s)
            v_circ1 = np.array([0.0, math.sqrt(MU_EARTH_KM3_S2 / r1_mag), 0.0], dtype=np.float64)
            v_dir2 = np.array([-math.sin(ang), math.cos(ang), 0.0], dtype=np.float64)
            v_circ2 = math.sqrt(MU_EARTH_KM3_S2 / r2_mag) * v_dir2

            dv_lambert_m_s = (float(np.linalg.norm(lam.v1_km_s - v_circ1)) + float(np.linalg.norm(v_circ2 - lam.v2_km_s))) * 1000.0
            # Scale realistic LEO J2-assisted phasing + rendezvous delta-v
            dv_rendezvous_m_s = min(145.0, 0.22 * dv_lambert_m_s + 18.0)
            dv_deorbit_kit_m_s = 32.0  # Attached solid/EDT deorbit kit deployment + separation trim
            dv_total_leg = dv_rendezvous_m_s + dv_deorbit_kit_m_s

            metric = dv_total_leg / max(1.0, cand.kessler_priority_score)
            if metric < best_cost_per_risk:
                best_cost_per_risk = metric
                best_candidate = cand
                best_eval = (wait_days, dv_rendezvous_m_s, dv_deorbit_kit_m_s, dv_total_leg, lam)

        if best_candidate is None or best_eval is None:
            break

        wait_days, dv_rend, dv_deorb, dv_leg, lam_sol = best_eval
        current_total_mass = servicer_dry_mass_kg + rem_fuel
        prop_used = current_total_mass * (1.0 - math.exp(-dv_leg / v_exh))

        if prop_used > rem_fuel:
            break

        rem_fuel -= prop_used
        total_dv += dv_leg
        total_days += wait_days + 1.5  # 1.5 days proximity ops & capture
        total_risk_removed += best_candidate.kessler_priority_score

        if not sample_lambert_dict:
            sample_lambert_dict = lam_sol.to_dict()

        leg_obj = ADRTourLeg(
            leg_index=leg_idx,
            target_id=best_candidate.target_id,
            target_name=best_candidate.name,
            from_altitude_km=round(curr_alt, 1),
            to_altitude_km=round(best_candidate.altitude_km, 1),
            raan_drift_wait_days=round(wait_days, 2),
            lambert_rendezvous_dv_m_s=round(dv_rend, 2),
            deorbit_kit_impulse_dv_m_s=round(dv_deorb, 2),
            total_leg_dv_m_s=round(dv_leg, 2),
            propellant_used_kg=round(prop_used, 2),
            remaining_fuel_kg=round(rem_fuel, 2),
            kessler_risk_removed=round(best_candidate.kessler_priority_score, 2),
        )
        visited_legs.append(asdict(leg_obj))

        curr_alt = best_candidate.altitude_km
        curr_inc = best_candidate.inclination_deg
        curr_raan = best_candidate.raan_deg
        unvisited.remove(best_candidate)

    return MultiDebrisTourReport(
        servicer_dry_mass_kg=round(servicer_dry_mass_kg, 1),
        initial_fuel_kg=round(initial_fuel_kg, 2),
        remaining_fuel_kg=round(rem_fuel, 2),
        isp_s=round(isp_s, 1),
        targets_visited=len(visited_legs),
        total_delta_v_m_s=round(total_dv, 2),
        total_fuel_consumed_kg=round(initial_fuel_kg - rem_fuel, 2),
        total_mission_duration_days=round(total_days, 2),
        total_kessler_risk_removed=round(total_risk_removed, 2),
        sample_lambert_transfer=sample_lambert_dict,
        legs=visited_legs,
    )
