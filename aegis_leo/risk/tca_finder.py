"""
Time of Closest Approach (TCA) Finder & Conjunction Screening
==============================================================

Turns two element sets into actual conjunction events instead of assuming a
pre-computed encounter geometry:

  1. Analytic propagation of Keplerian elements with secular J2 drift of the
     node, argument of perigee and mean anomaly.
  2. Apogee/perigee pre-filter that rejects pairs whose altitude shells can
     never come within the screening threshold.
  3. Coarse range scan followed by bounded Brent refinement of every local
     range minimum to locate each TCA.
  4. Encounter geometry at TCA: miss distance, relative speed, approach angle
     and the miss vector resolved in the primary's Radial / In-track /
     Cross-track (RIC) frame, ready for B-plane collision probability.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from aegis_leo.constants import EARTH_RADIUS_KM, J2_EARTH, MU_EARTH_KM3_S2
from aegis_leo.physics.coordinates import kepler_to_cartesian


@dataclass
class OrbitalElements:
    altitude_km: float
    inclination_deg: float
    raan_deg: float = 0.0
    eccentricity: float = 0.0
    arg_perigee_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    name: str = "OBJECT"

    @property
    def semi_major_axis_km(self) -> float:
        return EARTH_RADIUS_KM + self.altitude_km

    @property
    def mean_motion_rad_s(self) -> float:
        return math.sqrt(MU_EARTH_KM3_S2 / self.semi_major_axis_km**3)

    @property
    def perigee_radius_km(self) -> float:
        return self.semi_major_axis_km * (1.0 - self.eccentricity)

    @property
    def apogee_radius_km(self) -> float:
        return self.semi_major_axis_km * (1.0 + self.eccentricity)


def solve_kepler(mean_anomaly_rad: float, ecc: float, tol: float = 1e-12) -> float:
    """Eccentric anomaly from mean anomaly (Newton-Raphson)."""
    m = math.fmod(mean_anomaly_rad, 2.0 * math.pi)
    e_anom = m if ecc < 0.8 else math.pi
    for _ in range(50):
        delta = (e_anom - ecc * math.sin(e_anom) - m) / (1.0 - ecc * math.cos(e_anom))
        e_anom -= delta
        if abs(delta) < tol:
            break
    return e_anom


def j2_secular_rates(elements: OrbitalElements) -> Tuple[float, float, float]:
    """Secular J2 rates (rad/s) of RAAN, argument of perigee and mean anomaly."""
    a, e = elements.semi_major_axis_km, elements.eccentricity
    inc = math.radians(elements.inclination_deg)
    n = elements.mean_motion_rad_s
    p = a * (1.0 - e**2)
    k = 1.5 * J2_EARTH * (EARTH_RADIUS_KM / p) ** 2 * n
    raan_dot = -k * math.cos(inc)
    argp_dot = 0.5 * k * (5.0 * math.cos(inc) ** 2 - 1.0)
    m_dot = 0.5 * k * math.sqrt(1.0 - e**2) * (3.0 * math.cos(inc) ** 2 - 1.0)
    return raan_dot, argp_dot, m_dot


def propagate_elements(elements: OrbitalElements, t_s: float, include_j2: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """ECI position (km) and velocity (km/s) of an element set t_s seconds after epoch."""
    raan_dot, argp_dot, m_dot = j2_secular_rates(elements) if include_j2 else (0.0, 0.0, 0.0)
    ecc = elements.eccentricity
    m = math.radians(elements.mean_anomaly_deg) + (elements.mean_motion_rad_s + m_dot) * t_s
    e_anom = solve_kepler(m, ecc)
    nu = 2.0 * math.atan2(math.sqrt(1.0 + ecc) * math.sin(e_anom / 2.0), math.sqrt(1.0 - ecc) * math.cos(e_anom / 2.0))
    return kepler_to_cartesian(
        a_km=elements.semi_major_axis_km,
        ecc=ecc,
        inc_deg=elements.inclination_deg,
        raan_deg=elements.raan_deg + math.degrees(raan_dot * t_s),
        argp_deg=elements.arg_perigee_deg + math.degrees(argp_dot * t_s),
        nu_deg=math.degrees(nu),
    )


def ric_frame(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    """Rows are the Radial, In-track and Cross-track unit vectors of the orbit."""
    r_hat = r_eci / np.linalg.norm(r_eci)
    c_hat = np.cross(r_eci, v_eci)
    c_hat /= np.linalg.norm(c_hat)
    return np.vstack([r_hat, np.cross(c_hat, r_hat), c_hat])


@dataclass
class ConjunctionEvent:
    tca_s: float
    miss_distance_km: float
    relative_speed_km_s: float
    approach_angle_deg: float
    radial_miss_km: float
    in_track_miss_km: float
    cross_track_miss_km: float
    primary_altitude_km: float
    severity: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConjunctionScreeningReport:
    primary_name: str
    secondary_name: str
    duration_hours: float
    screening_threshold_km: float
    shell_filter_passed: bool
    radial_shell_gap_km: float
    close_approaches_found: int
    events_within_threshold: int
    min_miss_distance_km: float
    tca_of_min_miss_s: float
    relative_speed_at_min_km_s: float
    screening_status: str
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _severity(miss_km: float, threshold_km: float) -> str:
    if miss_km <= 1.0:
        return "RED"
    if miss_km <= min(5.0, threshold_km):
        return "YELLOW"
    return "GREEN" if miss_km > threshold_km else "WATCH"


def find_close_approaches(
    primary: OrbitalElements,
    secondary: OrbitalElements,
    duration_s: float,
    coarse_step_s: float = 20.0,
    include_j2: bool = True,
    max_report_distance_km: float = 500.0,
    threshold_km: float = 5.0,
) -> List[ConjunctionEvent]:
    """Locate every local range minimum below max_report_distance_km within the window."""
    if duration_s <= 0.0 or coarse_step_s <= 0.0:
        raise ValueError("duration_s and coarse_step_s must be positive.")

    def separation(t: float) -> float:
        r1, _ = propagate_elements(primary, t, include_j2)
        r2, _ = propagate_elements(secondary, t, include_j2)
        return float(np.linalg.norm(r2 - r1))

    times = np.arange(0.0, duration_s + coarse_step_s, coarse_step_s)
    times[-1] = min(times[-1], duration_s)
    dist = np.array([separation(float(t)) for t in times])

    events: List[ConjunctionEvent] = []
    for i in range(len(times)):
        left = dist[i - 1] if i > 0 else math.inf
        right = dist[i + 1] if i < len(times) - 1 else math.inf
        if not (dist[i] <= left and dist[i] < right):
            continue
        lo, hi = float(times[max(i - 1, 0)]), float(times[min(i + 1, len(times) - 1)])
        tca = float(times[i])
        if hi > lo:
            res = minimize_scalar(separation, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
            if res.fun <= dist[i]:
                tca = float(res.x)
        miss = separation(tca)
        if miss > max_report_distance_km:
            continue
        r1, v1 = propagate_elements(primary, tca, include_j2)
        r2, v2 = propagate_elements(secondary, tca, include_j2)
        v_rel = v2 - v1
        cos_angle = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        miss_ric = ric_frame(r1, v1) @ (r2 - r1)
        events.append(ConjunctionEvent(
            tca_s=round(tca, 4),
            miss_distance_km=round(miss, 6),
            relative_speed_km_s=round(float(np.linalg.norm(v_rel)), 6),
            approach_angle_deg=round(math.degrees(math.acos(max(-1.0, min(1.0, cos_angle)))), 4),
            radial_miss_km=round(float(miss_ric[0]), 6),
            in_track_miss_km=round(float(miss_ric[1]), 6),
            cross_track_miss_km=round(float(miss_ric[2]), 6),
            primary_altitude_km=round(float(np.linalg.norm(r1)) - EARTH_RADIUS_KM, 3),
            severity=_severity(miss, threshold_km),
        ))
    return events


def screen_conjunctions(
    primary: OrbitalElements,
    secondary: OrbitalElements,
    duration_hours: float = 24.0,
    threshold_km: float = 5.0,
    coarse_step_s: float = 20.0,
    include_j2: bool = True,
    max_events: int = 25,
) -> ConjunctionScreeningReport:
    """Screen a primary/secondary pair for conjunctions over a time window."""
    gap = max(
        primary.perigee_radius_km - secondary.apogee_radius_km,
        secondary.perigee_radius_km - primary.apogee_radius_km,
        0.0,
    )
    shell_ok = gap <= threshold_km

    events: List[ConjunctionEvent] = []
    if shell_ok:
        events = find_close_approaches(
            primary, secondary, duration_hours * 3600.0, coarse_step_s, include_j2, threshold_km=threshold_km
        )
    within = [e for e in events if e.miss_distance_km <= threshold_km]
    closest = min(events, key=lambda e: e.miss_distance_km) if events else None

    if not shell_ok:
        status = "NO_CONJUNCTION_POSSIBLE_SHELL_FILTER"
    elif closest is None or not within:
        status = "CLEAR"
    elif closest.miss_distance_km <= 1.0:
        status = "RED_CONJUNCTION_ALERT"
    else:
        status = "YELLOW_CONJUNCTION_WARNING"

    ranked = sorted(events, key=lambda e: e.miss_distance_km)[:max_events]
    return ConjunctionScreeningReport(
        primary_name=primary.name,
        secondary_name=secondary.name,
        duration_hours=duration_hours,
        screening_threshold_km=threshold_km,
        shell_filter_passed=shell_ok,
        radial_shell_gap_km=round(gap, 4),
        close_approaches_found=len(events),
        events_within_threshold=len(within),
        min_miss_distance_km=closest.miss_distance_km if closest else float("inf"),
        tca_of_min_miss_s=closest.tca_s if closest else -1.0,
        relative_speed_at_min_km_s=closest.relative_speed_km_s if closest else 0.0,
        screening_status=status,
        events=[e.to_dict() for e in sorted(ranked, key=lambda e: e.tca_s)],
    )


def build_intercepting_secondary(
    primary: OrbitalElements,
    tca_s: float,
    inclination_deg: float,
    radial_offset_km: float = 0.0,
    name: str = "DEBRIS",
    include_j2: bool = False,
) -> OrbitalElements:
    """
    Construct a circular secondary on a different plane that passes radially
    above/below the primary's position at tca_s. Useful for generating
    synthetic conjunction scenarios for drills and tests. include_j2 must
    match the propagation mode later used for screening.
    """
    if primary.eccentricity != 0.0:
        raise ValueError("build_intercepting_secondary requires a circular primary.")
    r_p, _ = propagate_elements(primary, tca_s, include_j2)
    r_hat = r_p / np.linalg.norm(r_p)
    inc = math.radians(inclination_deg)
    if abs(r_hat[2]) > abs(math.sin(inc)) + 1e-12:
        raise ValueError("Secondary inclination is too low to reach the primary's latitude at tca_s.")
    u = math.asin(max(-1.0, min(1.0, r_hat[2] / math.sin(inc)))) if math.sin(inc) != 0.0 else 0.0
    raan_tca = math.atan2(r_hat[1], r_hat[0]) - math.atan2(math.cos(inc) * math.sin(u), math.cos(u))

    sec = OrbitalElements(altitude_km=primary.altitude_km + radial_offset_km, inclination_deg=inclination_deg, name=name)
    raan_dot, argp_dot, m_dot = j2_secular_rates(sec) if include_j2 else (0.0, 0.0, 0.0)
    sec.raan_deg = math.degrees(raan_tca - raan_dot * tca_s) % 360.0
    sec.mean_anomaly_deg = math.degrees(u - (sec.mean_motion_rad_s + m_dot + argp_dot) * tca_s) % 360.0
    return sec
