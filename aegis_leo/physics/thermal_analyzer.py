"""
Multi-Node Spacecraft Orbital Thermal Control & Radiative Equilibrium Engine
============================================================================

Simulates coupled transient radiative, conductive, and free-molecular aerodynamic
heat transfer across five spacecraft thermal nodes over a complete LEO orbit:
  1. SOLAR_ARRAY       - Deployable photovoltaic wings
  2. BUS_STRUCTURE     - MLI-blanketed primary spacecraft bus & optical radiator
  3. EDGE_AI_AVIONICS  - Radiation-hardened Edge-AI flight computer & NPU
  4. BATTERY_PACK      - Li-Ion energy storage with survival thermostat heater
  5. PROPULSION_UNIT   - Propellant tank, feedlines & thruster chamber during burn

Heat Balance Governing Equation for Node i:
  C_i * dT_i/dt = Q_sun,i + Q_albedo,i + Q_earth_ir,i + Q_aero,i + Q_int,i + Q_heater,i
                  - sum_j G_ij * (T_i - T_j) - sigma * eps_i * A_rad,i * (T_i^4 - T_space^4)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

import numpy as np

from aegis_leo.constants import EARTH_RADIUS_KM, MU_EARTH_KM3_S2
from aegis_leo.physics.atmosphere import compute_atmospheric_density

STEFAN_BOLTZMANN = 5.670374419e-8  # W / (m^2 * K^4)
SOLAR_CONSTANT_W_M2 = 1361.0
EARTH_IR_FLUX_W_M2 = 237.0
EARTH_ALBEDO_COEFF = 0.30
DEEP_SPACE_TEMP_K = 2.7


@dataclass
class NodeThermalLimits:
    node_name: str
    min_operational_c: float
    max_operational_c: float
    min_temp_c: float
    max_temp_c: float
    mean_temp_c: float
    within_limits: bool


@dataclass
class ThermalAnalysisReport:
    altitude_km: float
    beta_angle_deg: float
    kp_storm: float
    avionics_load_w: float
    burn_active: bool
    peak_aero_heating_w: float
    mean_solar_absorbed_w: float
    mean_earth_ir_absorbed_w: float
    heater_energy_wh: float
    thermal_status: str
    all_nodes_within_limits: bool
    node_summaries: Dict[str, Dict[str, Any]]
    timeline: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def simulate_spacecraft_thermal_balance(
    altitude_km: float = 550.0,
    beta_angle_deg: float = 22.0,
    kp_storm: float = 5.0,
    f10_7: float = 165.0,
    avionics_load_w: float = 85.0,
    burn_active: bool = True,
    burn_duration_s: float = 120.0,
    solar_array_area_m2: float = 2.4,
    num_steps: int = 72,
) -> ThermalAnalysisReport:
    """
    Integrate 5-node spacecraft thermal network over a full LEO orbital period
    including eclipse shadow entry/exit, storm thermospheric aerodynamic heating,
    and thruster firing heat soak.
    """
    r_km = EARTH_RADIUS_KM + altitude_km
    period_s = 2.0 * math.pi * math.sqrt((r_km**3) / MU_EARTH_KM3_S2)
    dt_s = period_s / float(num_steps)
    v_orb_m_s = math.sqrt(MU_EARTH_KM3_S2 / r_km) * 1000.0

    # Earth spherical view factor F_E = (R_E / r)^2
    view_factor_earth = (EARTH_RADIUS_KM / r_km) ** 2

    # Critical beta angle for cylindrical shadow
    beta_rad = math.radians(abs(beta_angle_deg))
    sin_rho = min(1.0, EARTH_RADIUS_KM / r_km)
    if math.sin(beta_rad) >= sin_rho:
        half_eclipse_deg = 0.0
    else:
        cos_eta = math.sqrt(max(0.0, sin_rho**2 - math.sin(beta_rad) ** 2)) / max(1e-6, math.cos(beta_rad))
        half_eclipse_deg = math.degrees(math.acos(min(1.0, max(-1.0, cos_eta))))

    # Free-molecular aerodynamic heating during storm
    rho = compute_atmospheric_density(altitude_km, kp=kp_storm, f10_7=f10_7)
    accommodation_ch = 0.88
    ram_area_m2 = 1.15
    q_aero_w = 0.5 * rho * (v_orb_m_s**3) * accommodation_ch * ram_area_m2

    # Node thermal capacitances C_i (J/K)
    # [0: SOLAR_ARRAY, 1: BUS_STRUCTURE, 2: EDGE_AI_AVIONICS, 3: BATTERY_PACK, 4: PROPULSION_UNIT]
    c_nodes = np.array([3600.0, 14500.0, 2800.0, 5200.0, 4200.0], dtype=np.float64)

    # Conductive couplings G_ij (W/K) to primary BUS_STRUCTURE (index 1)
    g_to_bus = np.array([1.5, 0.0, 9.5, 6.0, 5.5], dtype=np.float64)

    # Effective radiator area * IR emissivity (eps * A_rad in m^2)
    # Internal units (Avionics, Battery, Propulsion) are enclosed inside MLI and radiate minimally to space directly
    eps_area_rad = np.array(
        [
            0.80 * (1.8 * solar_array_area_m2),
            0.76 * 1.35,
            0.80 * 0.12,
            0.25 * 0.06,
            0.30 * 0.08,
        ],
        dtype=np.float64,
    )

    # Initial temperatures in Kelvin
    temps_k = np.array([292.15, 293.15, 301.15, 291.15, 294.15], dtype=np.float64)

    node_keys = [
        "SOLAR_ARRAY",
        "BUS_STRUCTURE",
        "EDGE_AI_AVIONICS",
        "BATTERY_PACK",
        "PROPULSION_UNIT",
    ]
    limits_c = {
        "SOLAR_ARRAY": (-85.0, 95.0),
        "BUS_STRUCTURE": (-25.0, 60.0),
        "EDGE_AI_AVIONICS": (-15.0, 70.0),
        "BATTERY_PACK": (0.0, 38.0),
        "PROPULSION_UNIT": (5.0, 65.0),
    }

    history_c: Dict[str, List[float]] = {k: [] for k in node_keys}
    timeline: List[Dict[str, float]] = []
    solar_abs_list: List[float] = []
    earth_ir_list: List[float] = []
    heater_energy_j = 0.0

    # Run 2 orbits so initial transient settles into periodic orbital equilibrium
    total_sim_steps = num_steps * 2
    for s_idx in range(total_sim_steps):
        orb_step = s_idx % num_steps
        angle_deg = (orb_step / float(num_steps)) * 360.0
        t_orb_s = orb_step * dt_s

        # Determine shadow factor (eclipse centered around 180 deg)
        dist_from_180 = abs(angle_deg - 180.0)
        if half_eclipse_deg <= 0.0:
            sun_factor = 1.0
        elif dist_from_180 < half_eclipse_deg:
            sun_factor = 0.0
        elif dist_from_180 < half_eclipse_deg + 4.0:
            sun_factor = 0.5
        else:
            sun_factor = 1.0

        cos_subsolar = max(0.0, math.cos(math.radians(angle_deg))) * sun_factor

        # External absorbed solar + albedo + Earth IR per node (W)
        q_sun_arr = SOLAR_CONSTANT_W_M2 * 0.68 * solar_array_area_m2 * sun_factor * math.cos(beta_rad * 0.5)
        q_sun_bus = SOLAR_CONSTANT_W_M2 * 0.24 * 0.85 * sun_factor

        q_alb_arr = SOLAR_CONSTANT_W_M2 * EARTH_ALBEDO_COEFF * 0.68 * solar_array_area_m2 * view_factor_earth * cos_subsolar
        q_alb_bus = SOLAR_CONSTANT_W_M2 * EARTH_ALBEDO_COEFF * 0.24 * 0.90 * view_factor_earth * cos_subsolar

        q_ir_arr = EARTH_IR_FLUX_W_M2 * 0.80 * solar_array_area_m2 * view_factor_earth
        q_ir_bus = EARTH_IR_FLUX_W_M2 * 0.75 * 1.10 * view_factor_earth

        # Thruster firing pulse around orbital angle 45 deg
        is_burning_now = burn_active and (0.0 <= t_orb_s <= burn_duration_s)
        q_thruster_w = 65.0 if is_burning_now else 4.0

        # Active survival heaters on Battery (< 8 C) and Propulsion (< 12 C)
        q_heat_batt = 18.0 if (temps_k[3] - 273.15) < 8.0 else 0.0
        q_heat_prop = 14.0 if (temps_k[4] - 273.15) < 12.0 else 0.0

        q_ext = np.array(
            [
                q_sun_arr + q_alb_arr + q_ir_arr + 0.65 * q_aero_w,
                q_sun_bus + q_alb_bus + q_ir_bus + 0.35 * q_aero_w + 22.0,
                avionics_load_w,
                6.0 + q_heat_batt,
                q_thruster_w + q_heat_prop,
            ],
            dtype=np.float64,
        )

        # Sub-step integration for numerical stability of T^4 radiation
        sub_steps = 6
        dt_sub = dt_s / float(sub_steps)
        for _ in range(sub_steps):
            t_bus = temps_k[1]
            q_cond = np.zeros(5, dtype=np.float64)
            for i in [0, 2, 3, 4]:
                flux_i_to_bus = g_to_bus[i] * (temps_k[i] - t_bus)
                q_cond[i] -= flux_i_to_bus
                q_cond[1] += flux_i_to_bus

            q_rad = STEFAN_BOLTZMANN * eps_area_rad * (temps_k**4 - DEEP_SPACE_TEMP_K**4)
            d_temp = ((q_ext + q_cond - q_rad) / c_nodes) * dt_sub
            temps_k = np.clip(temps_k + d_temp, 160.0, 420.0)

        # Record metrics during second (equilibrium) orbit
        if s_idx >= num_steps:
            heater_energy_j += (q_heat_batt + q_heat_prop) * dt_s
            solar_abs_list.append(q_sun_arr + q_sun_bus + q_alb_arr + q_alb_bus)
            earth_ir_list.append(q_ir_arr + q_ir_bus)
            temps_c = temps_k - 273.15
            for idx, k_name in enumerate(node_keys):
                history_c[k_name].append(float(temps_c[idx]))

            timeline.append(
                {
                    "orbit_angle_deg": round(angle_deg, 1),
                    "sun_factor": round(sun_factor, 2),
                    "solar_array_c": round(float(temps_c[0]), 2),
                    "bus_structure_c": round(float(temps_c[1]), 2),
                    "edge_ai_avionics_c": round(float(temps_c[2]), 2),
                    "battery_pack_c": round(float(temps_c[3]), 2),
                    "propulsion_unit_c": round(float(temps_c[4]), 2),
                }
            )

    node_summaries: Dict[str, Dict[str, Any]] = {}
    all_ok = True
    for k_name in node_keys:
        arr = np.asarray(history_c[k_name], dtype=np.float64)
        min_lim, max_lim = limits_c[k_name]
        t_min = float(np.min(arr))
        t_max = float(np.max(arr))
        t_mean = float(np.mean(arr))
        ok = (t_min >= min_lim) and (t_max <= max_lim)
        if not ok:
            all_ok = False
        node_summaries[k_name] = asdict(
            NodeThermalLimits(
                node_name=k_name,
                min_operational_c=min_lim,
                max_operational_c=max_lim,
                min_temp_c=round(t_min, 2),
                max_temp_c=round(t_max, 2),
                mean_temp_c=round(t_mean, 2),
                within_limits=bool(ok),
            )
        )

    status = "NOMINAL_THERMAL_EQUILIBRIUM" if all_ok else "THERMAL_MARGIN_EXCEEDED"

    return ThermalAnalysisReport(
        altitude_km=round(altitude_km, 1),
        beta_angle_deg=round(beta_angle_deg, 2),
        kp_storm=round(kp_storm, 2),
        avionics_load_w=round(avionics_load_w, 1),
        burn_active=bool(burn_active),
        peak_aero_heating_w=round(q_aero_w, 4),
        mean_solar_absorbed_w=round(float(np.mean(solar_abs_list)), 2),
        mean_earth_ir_absorbed_w=round(float(np.mean(earth_ir_list)), 2),
        heater_energy_wh=round(heater_energy_j / 3600.0, 3),
        thermal_status=status,
        all_nodes_within_limits=bool(all_ok),
        node_summaries=node_summaries,
        timeline=timeline,
    )
