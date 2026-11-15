"""
Impulsive Orbit Transfer Planner
================================

Closed-form two-body transfer design between circular LEO orbits:

  1. Hohmann two-impulse transfer.
  2. Bi-elliptic three-impulse transfer through an intermediate apogee.
  3. Pure plane change and combined Hohmann + plane change with the optimal
     split of the inclination change between the two burns.
  4. Coplanar phasing orbit for along-track slot re-positioning.
  5. Tsiolkovsky propellant accounting and a strategy trade-off report used
     for orbit raising, graveyard / disposal lowering and slot changes.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from scipy.optimize import minimize_scalar

from aegis_leo.constants import EARTH_RADIUS_KM, G0_M_S2, MU_EARTH_KM3_S2


def circular_velocity_km_s(radius_km: float, mu: float = MU_EARTH_KM3_S2) -> float:
    return math.sqrt(mu / radius_km)


def orbital_period_s(semi_major_axis_km: float, mu: float = MU_EARTH_KM3_S2) -> float:
    return 2.0 * math.pi * math.sqrt(semi_major_axis_km**3 / mu)


def _vis_viva(radius_km: float, semi_major_axis_km: float, mu: float = MU_EARTH_KM3_S2) -> float:
    return math.sqrt(mu * (2.0 / radius_km - 1.0 / semi_major_axis_km))


def _vector_dv(v_a: float, v_b: float, angle_rad: float) -> float:
    """Magnitude of the velocity change between two speeds separated by a plane rotation."""
    return math.sqrt(max(0.0, v_a**2 + v_b**2 - 2.0 * v_a * v_b * math.cos(angle_rad)))


def propellant_for_delta_v(initial_mass_kg: float, delta_v_m_s: float, isp_s: float) -> float:
    """Tsiolkovsky propellant mass (kg) needed for a delta-v starting from initial_mass_kg."""
    if isp_s <= 0.0:
        raise ValueError("isp_s must be positive.")
    return initial_mass_kg * (1.0 - math.exp(-abs(delta_v_m_s) / (isp_s * G0_M_S2)))


@dataclass
class TransferSolution:
    strategy: str
    burns_m_s: List[float]
    total_delta_v_m_s: float
    time_of_flight_s: float
    details: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def hohmann_transfer(r1_km: float, r2_km: float, mu: float = MU_EARTH_KM3_S2) -> TransferSolution:
    """Two-impulse Hohmann transfer between coplanar circular orbits of radius r1 and r2."""
    if r1_km <= 0.0 or r2_km <= 0.0:
        raise ValueError("Orbit radii must be positive.")
    a_t = 0.5 * (r1_km + r2_km)
    dv1 = abs(_vis_viva(r1_km, a_t, mu) - circular_velocity_km_s(r1_km, mu))
    dv2 = abs(circular_velocity_km_s(r2_km, mu) - _vis_viva(r2_km, a_t, mu))
    return TransferSolution(
        strategy="HOHMANN",
        burns_m_s=[dv1 * 1000.0, dv2 * 1000.0],
        total_delta_v_m_s=(dv1 + dv2) * 1000.0,
        time_of_flight_s=0.5 * orbital_period_s(a_t, mu),
        details={"transfer_sma_km": a_t, "transfer_eccentricity": abs(r2_km - r1_km) / (r1_km + r2_km)},
    )


def bielliptic_transfer(r1_km: float, r2_km: float, rb_km: float, mu: float = MU_EARTH_KM3_S2) -> TransferSolution:
    """Three-impulse bi-elliptic transfer through an intermediate apoapsis radius rb >= max(r1, r2)."""
    if rb_km < max(r1_km, r2_km):
        raise ValueError("Intermediate radius rb_km must be >= max(r1_km, r2_km).")
    a1 = 0.5 * (r1_km + rb_km)
    a2 = 0.5 * (r2_km + rb_km)
    dv1 = abs(_vis_viva(r1_km, a1, mu) - circular_velocity_km_s(r1_km, mu))
    dv2 = abs(_vis_viva(rb_km, a2, mu) - _vis_viva(rb_km, a1, mu))
    dv3 = abs(circular_velocity_km_s(r2_km, mu) - _vis_viva(r2_km, a2, mu))
    return TransferSolution(
        strategy="BI_ELLIPTIC",
        burns_m_s=[dv1 * 1000.0, dv2 * 1000.0, dv3 * 1000.0],
        total_delta_v_m_s=(dv1 + dv2 + dv3) * 1000.0,
        time_of_flight_s=0.5 * (orbital_period_s(a1, mu) + orbital_period_s(a2, mu)),
        details={"intermediate_radius_km": rb_km},
    )


def plane_change_delta_v_m_s(radius_km: float, delta_inclination_deg: float, mu: float = MU_EARTH_KM3_S2) -> float:
    """Pure plane rotation of a circular orbit: dv = 2 v sin(di / 2)."""
    return 2000.0 * circular_velocity_km_s(radius_km, mu) * abs(math.sin(math.radians(delta_inclination_deg) / 2.0))


def combined_plane_change_transfer(
    r1_km: float,
    r2_km: float,
    delta_inclination_deg: float,
    mu: float = MU_EARTH_KM3_S2,
) -> TransferSolution:
    """
    Hohmann transfer with the inclination change split optimally between the
    departure and arrival burns (most of it goes to the slower, higher burn).
    """
    di = math.radians(abs(delta_inclination_deg))
    a_t = 0.5 * (r1_km + r2_km)
    vc1, vc2 = circular_velocity_km_s(r1_km, mu), circular_velocity_km_s(r2_km, mu)
    vt1, vt2 = _vis_viva(r1_km, a_t, mu), _vis_viva(r2_km, a_t, mu)

    def total(split: float) -> float:
        return _vector_dv(vc1, vt1, split * di) + _vector_dv(vt2, vc2, (1.0 - split) * di)

    split = 0.0
    if di > 0.0:
        split = float(minimize_scalar(total, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-10}).x)
        split = min((0.0, split, 1.0), key=total)
    dv1 = _vector_dv(vc1, vt1, split * di)
    dv2 = _vector_dv(vt2, vc2, (1.0 - split) * di)
    return TransferSolution(
        strategy="COMBINED_PLANE_CHANGE_HOHMANN",
        burns_m_s=[dv1 * 1000.0, dv2 * 1000.0],
        total_delta_v_m_s=(dv1 + dv2) * 1000.0,
        time_of_flight_s=0.5 * orbital_period_s(a_t, mu),
        details={
            "plane_change_at_departure_deg": math.degrees(split * di),
            "plane_change_at_arrival_deg": math.degrees((1.0 - split) * di),
        },
    )


def phasing_maneuver(
    radius_km: float,
    phase_angle_deg: float,
    phasing_orbits: int = 1,
    min_perigee_altitude_km: float = 180.0,
    mu: float = MU_EARTH_KM3_S2,
) -> TransferSolution:
    """
    Two-burn phasing orbit returning to the same circular orbit shifted by
    phase_angle_deg along track (positive = move ahead, uses a shorter-period
    inner orbit) after phasing_orbits revolutions of the phasing orbit.
    """
    if phasing_orbits < 1:
        raise ValueError("phasing_orbits must be >= 1.")
    period = orbital_period_s(radius_km, mu)
    phasing_period = period * (1.0 - math.radians(phase_angle_deg) / (2.0 * math.pi * phasing_orbits))
    if phasing_period <= 0.0:
        raise ValueError("Phase angle too large for the requested number of phasing orbits.")
    a_ph = (mu * (phasing_period / (2.0 * math.pi)) ** 2) ** (1.0 / 3.0)
    perigee_alt = min(radius_km, 2.0 * a_ph - radius_km) - EARTH_RADIUS_KM
    if perigee_alt < min_perigee_altitude_km:
        raise ValueError(
            f"Phasing orbit perigee altitude {perigee_alt:.1f} km is below {min_perigee_altitude_km:.1f} km; "
            "increase phasing_orbits."
        )
    dv = abs(_vis_viva(radius_km, a_ph, mu) - circular_velocity_km_s(radius_km, mu)) * 1000.0
    return TransferSolution(
        strategy="PHASING",
        burns_m_s=[dv, dv],
        total_delta_v_m_s=2.0 * dv,
        time_of_flight_s=phasing_period * phasing_orbits,
        details={
            "phasing_sma_km": a_ph,
            "phasing_perigee_altitude_km": perigee_alt,
            "phasing_apogee_altitude_km": max(radius_km, 2.0 * a_ph - radius_km) - EARTH_RADIUS_KM,
        },
    )


@dataclass
class OrbitTransferReport:
    initial_altitude_km: float
    target_altitude_km: float
    delta_inclination_deg: float
    recommended_strategy: str
    total_delta_v_m_s: float
    time_of_flight_min: float
    propellant_required_kg: float
    propellant_available_kg: float
    propellant_remaining_kg: float
    feasible: bool
    burn_durations_s: List[float]
    separate_plane_change_penalty_m_s: float
    strategies: List[Dict[str, Any]] = field(default_factory=list)
    phasing: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def plan_orbit_transfer(
    initial_altitude_km: float = 550.0,
    target_altitude_km: float = 600.0,
    delta_inclination_deg: float = 0.0,
    wet_mass_kg: float = 260.0,
    propellant_kg: float = 15.4,
    isp_s: float = 250.0,
    thrust_n: float = 0.0,
    phase_angle_deg: float = 0.0,
    phasing_orbits: int = 10,
    bielliptic_apogee_altitude_km: Optional[float] = None,
) -> OrbitTransferReport:
    """
    Compare impulsive strategies for a circular-to-circular LEO transfer and
    report the cheapest one together with its propellant feasibility.
    """
    if min(initial_altitude_km, target_altitude_km) <= 100.0:
        raise ValueError("Altitudes must be above 100 km.")
    if wet_mass_kg <= 0.0 or not 0.0 <= propellant_kg < wet_mass_kg:
        raise ValueError("wet_mass_kg must be positive and propellant_kg must be within [0, wet_mass_kg).")

    r1 = EARTH_RADIUS_KM + initial_altitude_km
    r2 = EARTH_RADIUS_KM + target_altitude_km
    di = abs(delta_inclination_deg)

    hohmann = hohmann_transfer(r1, r2)
    # Naive sequence: Hohmann, then rotate the plane at whichever end is slower (higher)
    separate = hohmann.total_delta_v_m_s + plane_change_delta_v_m_s(max(r1, r2), di)
    candidates: List[TransferSolution] = [combined_plane_change_transfer(r1, r2, di) if di > 0.0 else hohmann]

    rb = EARTH_RADIUS_KM + (bielliptic_apogee_altitude_km or 0.0)
    if bielliptic_apogee_altitude_km is not None and rb >= max(r1, r2):
        bi = bielliptic_transfer(r1, r2, rb)
        if di > 0.0:
            # Perform the whole plane rotation at the slow intermediate apoapsis
            a1, a2 = 0.5 * (r1 + rb), 0.5 * (r2 + rb)
            mid = _vector_dv(_vis_viva(rb, a1), _vis_viva(rb, a2), math.radians(di)) * 1000.0
            bi.burns_m_s[1] = mid
            bi.total_delta_v_m_s = sum(bi.burns_m_s)
            bi.strategy = "BI_ELLIPTIC_APOGEE_PLANE_CHANGE"
        candidates.append(bi)

    best = min(candidates, key=lambda c: c.total_delta_v_m_s)

    phasing = None
    total_dv = best.total_delta_v_m_s
    total_tof = best.time_of_flight_s
    burns = list(best.burns_m_s)
    if abs(phase_angle_deg) > 0.0:
        ph = phasing_maneuver(r2, phase_angle_deg, phasing_orbits)
        phasing = ph.to_dict()
        total_dv += ph.total_delta_v_m_s
        total_tof += ph.time_of_flight_s
        burns += ph.burns_m_s

    mass = wet_mass_kg
    durations: List[float] = []
    for dv in burns:
        used = propellant_for_delta_v(mass, dv, isp_s)
        durations.append(round(used * isp_s * G0_M_S2 / thrust_n, 2) if thrust_n > 0.0 else 0.0)
        mass -= used
    required = wet_mass_kg - mass

    return OrbitTransferReport(
        initial_altitude_km=initial_altitude_km,
        target_altitude_km=target_altitude_km,
        delta_inclination_deg=delta_inclination_deg,
        recommended_strategy=best.strategy,
        total_delta_v_m_s=round(total_dv, 4),
        time_of_flight_min=round(total_tof / 60.0, 3),
        propellant_required_kg=round(required, 5),
        propellant_available_kg=propellant_kg,
        propellant_remaining_kg=round(propellant_kg - required, 5),
        feasible=required <= propellant_kg,
        burn_durations_s=durations,
        separate_plane_change_penalty_m_s=round(separate - candidates[0].total_delta_v_m_s, 4),
        strategies=[c.to_dict() for c in candidates],
        phasing=phasing,
    )
