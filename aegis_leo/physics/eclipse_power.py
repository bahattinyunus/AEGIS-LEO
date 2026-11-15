"""
Orbital Eclipse, Beta Angle & Solar/Battery Power Budget Engine
================================================================

Models:
1. Astronomical Sun direction vector from day-of-year (solar declination & right ascension).
2. Orbital Beta angle (beta) between the orbit plane and the Sun vector, plus
   Sun-Synchronous Orbit (SSO) J2 inclination solver.
3. Dual-cone Earth shadow geometry (Umbra, Penumbra, Sunlit) along a full orbit.
4. Spacecraft Solar Array generation, payload/bus power consumption, battery
   State-of-Charge (SoC) & Depth-of-Discharge (DoD %), and thermal swing estimation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List

import numpy as np

from aegis_leo.constants import EARTH_J2, MU_EARTH_KM3_S2, EARTH_RADIUS_KM

SOLAR_CONSTANT_W_M2 = 1361.0
SUN_ANGULAR_RADIUS_RAD = math.radians(0.2666)
EARTH_OBLIQUITY_RAD = math.radians(23.4393)
SSO_PRECESSION_RAD_S = 2.0 * math.pi / (365.2422 * 86400.0)


@dataclass
class EclipseOrbitSample:
    time_s: float
    orbit_angle_deg: float
    shadow_state: str  # SUNLIT | PENUMBRA | UMBRA
    sunlight_fraction: float
    solar_power_w: float
    net_power_w: float
    battery_soc_pct: float
    estimated_temp_c: float


@dataclass
class EclipsePowerReport:
    altitude_km: float
    inclination_deg: float
    raan_deg: float
    day_of_year: int
    beta_angle_deg: float
    orbital_period_s: float
    sunlit_duration_s: float
    eclipse_duration_s: float
    eclipse_fraction: float
    sso_required_inclination_deg: float
    peak_solar_power_w: float
    mean_generated_power_w: float
    load_power_w: float
    net_orbit_energy_wh: float
    battery_capacity_wh: float
    max_dod_pct: float
    min_soc_pct: float
    power_margin_positive: bool
    temp_min_c: float
    temp_max_c: float
    timeline: List[EclipseOrbitSample] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def compute_sun_unit_vector(day_of_year: int) -> np.ndarray:
    """
    Compute ECI unit vector pointing from Earth to the Sun for a given day of year (1..366).
    """
    # Mean solar longitude from vernal equinox (~day 80)
    lambda_sun = 2.0 * math.pi * ((day_of_year - 80) % 365.25) / 365.25
    x = math.cos(lambda_sun)
    y = math.sin(lambda_sun) * math.cos(EARTH_OBLIQUITY_RAD)
    z = math.sin(lambda_sun) * math.sin(EARTH_OBLIQUITY_RAD)
    vec = np.array([x, y, z], dtype=np.float64)
    return vec / np.linalg.norm(vec)


def compute_sso_inclination_deg(altitude_km: float, eccentricity: float = 0.0) -> float:
    """
    Solve the J2 nodal precession equation for a Sun-Synchronous Orbit (SSO) inclination.
    Returns inclination in degrees (typically ~97-99 deg for LEO).
    """
    a_km = EARTH_RADIUS_KM + altitude_km
    n = math.sqrt(MU_EARTH_KM3_S2 / (a_km ** 3))
    p = a_km * (1.0 - eccentricity ** 2)
    cos_i = -2.0 * SSO_PRECESSION_RAD_S * (p ** 2) / (3.0 * n * EARTH_J2 * (EARTH_RADIUS_KM ** 2))
    cos_i = max(-1.0, min(1.0, cos_i))
    return round(math.degrees(math.acos(cos_i)), 3)


def compute_beta_angle_deg(
    inclination_deg: float,
    raan_deg: float,
    sun_vec_eci: np.ndarray,
) -> float:
    """
    Compute the orbital Beta angle (degrees) between the orbit plane and the Sun vector.
    """
    inc = math.radians(inclination_deg)
    raan = math.radians(raan_deg)
    # Orbit normal unit vector h_hat
    h_hat = np.array([
        math.sin(inc) * math.sin(raan),
        -math.sin(inc) * math.cos(raan),
        math.cos(inc),
    ], dtype=np.float64)
    sin_beta = float(np.clip(np.dot(h_hat, sun_vec_eci), -1.0, 1.0))
    return math.degrees(math.asin(sin_beta))


def evaluate_shadow_state(sat_pos_eci_km: np.ndarray, sun_unit_vec: np.ndarray) -> tuple[str, float]:
    """
    Dual-cone Earth shadow model returning (shadow_state, sunlight_fraction in [0, 1]).
    """
    proj = float(np.dot(sat_pos_eci_km, sun_unit_vec))
    if proj >= 0.0:
        return "SUNLIT", 1.0

    # Perpendicular distance from Earth-Sun shadow axis
    perp_vec = sat_pos_eci_km - proj * sun_unit_vec
    d_perp = float(np.linalg.norm(perp_vec))
    d_anti = -proj  # distance behind Earth center

    r_umbra = EARTH_RADIUS_KM - d_anti * math.tan(SUN_ANGULAR_RADIUS_RAD)
    r_penumbra = EARTH_RADIUS_KM + d_anti * math.tan(SUN_ANGULAR_RADIUS_RAD)

    if d_perp <= max(0.0, r_umbra):
        return "UMBRA", 0.0
    if d_perp < r_penumbra:
        frac = (d_perp - max(0.0, r_umbra)) / max(r_penumbra - max(0.0, r_umbra), 1e-6)
        return "PENUMBRA", float(np.clip(frac, 0.0, 1.0))
    return "SUNLIT", 1.0


def simulate_eclipse_and_power(
    altitude_km: float = 550.0,
    inclination_deg: float = 97.6,
    raan_deg: float = 15.0,
    day_of_year: int = 172,
    solar_array_area_m2: float = 2.4,
    solar_cell_efficiency: float = 0.295,
    bus_load_w: float = 180.0,
    battery_capacity_wh: float = 420.0,
    initial_soc_pct: float = 92.0,
    num_steps: int = 120,
) -> EclipsePowerReport:
    """
    Simulate one full orbital period for solar eclipse geometry, solar array power
    generation, battery SoC/DoD cycle, and thermal equilibrium swing.
    """
    r_km = EARTH_RADIUS_KM + altitude_km
    period_s = 2.0 * math.pi * math.sqrt((r_km ** 3) / MU_EARTH_KM3_S2)
    dt_s = period_s / num_steps

    sun_vec = compute_sun_unit_vector(day_of_year)
    beta_deg = compute_beta_angle_deg(inclination_deg, raan_deg, sun_vec)
    sso_inc = compute_sso_inclination_deg(altitude_km)

    inc = math.radians(inclination_deg)
    raan = math.radians(raan_deg)
    c_o, s_o = math.cos(raan), math.sin(raan)
    c_i, s_i = math.cos(inc), math.sin(inc)

    peak_solar_w = SOLAR_CONSTANT_W_M2 * solar_array_area_m2 * solar_cell_efficiency
    # Effective cosine factor from beta angle (assuming 1-axis tracking or body-mounted)
    cos_beta = max(0.25, math.cos(math.radians(beta_deg)))

    soc_wh = battery_capacity_wh * (initial_soc_pct / 100.0)
    min_soc_wh = soc_wh
    sunlit_steps = 0.0
    eclipse_steps = 0.0
    total_gen_wh = 0.0
    total_load_wh = 0.0

    timeline: List[EclipseOrbitSample] = []
    temps: List[float] = []

    for i in range(num_steps):
        t = i * dt_s
        u = 2.0 * math.pi * (i / num_steps)
        c_u, s_u = math.cos(u), math.sin(u)

        pos = np.array([
            r_km * (c_o * c_u - s_o * s_u * c_i),
            r_km * (s_o * c_u + c_o * s_u * c_i),
            r_km * (s_u * s_i),
        ], dtype=np.float64)

        state, sun_frac = evaluate_shadow_state(pos, sun_vec)
        sunlit_steps += sun_frac
        eclipse_steps += (1.0 - sun_frac)

        # Solar incidence modulation along orbit
        solar_w = peak_solar_w * sun_frac * cos_beta
        net_w = solar_w - bus_load_w

        dt_h = dt_s / 3600.0
        total_gen_wh += solar_w * dt_h
        total_load_wh += bus_load_w * dt_h

        # Battery charge/discharge (92% round-trip efficiency on charge)
        if net_w >= 0.0:
            soc_wh = min(battery_capacity_wh, soc_wh + net_w * dt_h * 0.92)
        else:
            soc_wh = max(0.0, soc_wh + net_w * dt_h)

        min_soc_wh = min(min_soc_wh, soc_wh)
        soc_pct = (soc_wh / battery_capacity_wh) * 100.0

        # Simplified lumped spacecraft thermal node (-42 C in deep umbra, +58 C in full sun)
        temp_c = -42.0 + 100.0 * sun_frac * (0.85 + 0.15 * cos_beta)
        temps.append(temp_c)

        timeline.append(
            EclipseOrbitSample(
                time_s=round(t, 1),
                orbit_angle_deg=round(math.degrees(u), 1),
                shadow_state=state,
                sunlight_fraction=round(sun_frac, 3),
                solar_power_w=round(solar_w, 2),
                net_power_w=round(net_w, 2),
                battery_soc_pct=round(soc_pct, 2),
                estimated_temp_c=round(temp_c, 1),
            )
        )

    sunlit_s = sunlit_steps * dt_s
    eclipse_s = eclipse_steps * dt_s
    eclipse_frac = eclipse_s / period_s
    min_soc_pct = (min_soc_wh / battery_capacity_wh) * 100.0
    max_dod_pct = max(0.0, 100.0 - min_soc_pct)
    mean_gen_w = total_gen_wh / (period_s / 3600.0)
    net_orbit_wh = total_gen_wh - total_load_wh

    return EclipsePowerReport(
        altitude_km=round(altitude_km, 2),
        inclination_deg=round(inclination_deg, 2),
        raan_deg=round(raan_deg, 2),
        day_of_year=day_of_year,
        beta_angle_deg=round(beta_deg, 2),
        orbital_period_s=round(period_s, 1),
        sunlit_duration_s=round(sunlit_s, 1),
        eclipse_duration_s=round(eclipse_s, 1),
        eclipse_fraction=round(eclipse_frac, 4),
        sso_required_inclination_deg=sso_inc,
        peak_solar_power_w=round(peak_solar_w, 2),
        mean_generated_power_w=round(mean_gen_w, 2),
        load_power_w=round(bus_load_w, 2),
        net_orbit_energy_wh=round(net_orbit_wh, 2),
        battery_capacity_wh=round(battery_capacity_wh, 2),
        max_dod_pct=round(max_dod_pct, 2),
        min_soc_pct=round(min_soc_pct, 2),
        power_margin_positive=net_orbit_wh >= 0.0,
        temp_min_c=round(min(temps), 1),
        temp_max_c=round(max(temps), 1),
        timeline=timeline,
    )
