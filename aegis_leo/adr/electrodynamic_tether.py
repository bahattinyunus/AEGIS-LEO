"""
AEGIS-LEO: Electrodynamic Tether (EDT) & Propellantless Deorbit Physics Module
Active Debris Removal (ADR) & Zero-Propellant End-of-Life Disposal.

Physical Principles:
1. Earth's Geomagnetic Dipole:
   B_r = 2 * (M_E / r^3) * cos(theta_m)
   B_theta = (M_E / r^3) * sin(theta_m)
2. Induced Electromotive Force (EMF):
   E = (v_rel x B) . L
3. Orbital Motion Limited (OML) Plasma Collection:
   I_oml = (2 * e * N_e * r_w * L / pi) * sqrt(2 * e * V_bias / m_e)
4. Lorentz Retarding Force:
   F_drag = integral (I(s) * (dL x B))
5. Comparative Trade-off:
   Chemical Impulse vs Low-Thrust EP vs Deployable Drag Sail vs Electrodynamic Tether.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from aegis_leo.constants import EARTH_RADIUS_KM, MU_EARTH_KM3_S2


class DeorbitMethod(str, Enum):
    NATURAL_DRAG = "NATURAL_DRAG"
    DEPLOYABLE_DRAG_SAIL = "DEPLOYABLE_DRAG_SAIL"
    CHEMICAL_RETRO_PROPULSION = "CHEMICAL_RETRO_PROPULSION"
    ELECTRIC_PROPULSION = "ELECTRIC_PROPULSION"
    ELECTRODYNAMIC_TETHER = "ELECTRODYNAMIC_TETHER"


@dataclass
class ElectrodynamicTetherSpecs:
    """Design parameters for a bare electrodynamic tether system."""
    length_m: float = 3000.0          # 3 km conductive tether
    wire_radius_mm: float = 0.4       # 0.4 mm tape / wire radius
    material_resistivity_ohm_m: float = 2.82e-8  # Aluminum (2.82e-8 ohm*m)
    mass_density_kg_m3: float = 2700.0           # Aluminum density
    cathode_impedance_ohms: float = 120.0        # Field-emission cathode impedance
    bias_efficiency: float = 0.85

    @property
    def tether_mass_kg(self) -> float:
        r_m = self.wire_radius_mm * 1e-3
        area = math.pi * (r_m ** 2)
        return self.length_m * area * self.mass_density_kg_m3

    @property
    def ohmic_resistance_ohms(self) -> float:
        r_m = self.wire_radius_mm * 1e-3
        area = math.pi * (r_m ** 2)
        return self.material_resistivity_ohm_m * self.length_m / max(1e-9, area)


@dataclass
class TetherDeorbitResult:
    """Simulation output of electrodynamic tether decay dynamics."""
    initial_altitude_km: float
    final_altitude_km: float
    deorbit_time_days: float
    average_lorentz_drag_force_mn: float
    peak_induced_emf_volts: float
    average_current_amps: float
    tether_mass_kg: float
    propellant_saved_kg: float
    altitude_history_km: List[float] = field(default_factory=list)
    time_history_days: List[float] = field(default_factory=list)
    status: str = "TETHER_DEORBIT_COMPLETE"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_altitude_km": self.initial_altitude_km,
            "final_altitude_km": self.final_altitude_km,
            "deorbit_time_days": round(self.deorbit_time_days, 1),
            "average_lorentz_drag_force_mn": round(self.average_lorentz_drag_force_mn, 2),
            "peak_induced_emf_volts": round(self.peak_induced_emf_volts, 1),
            "average_current_amps": round(self.average_current_amps, 3),
            "tether_mass_kg": round(self.tether_mass_kg, 2),
            "propellant_saved_kg": round(self.propellant_saved_kg, 2),
            "status": self.status,
            "data_points": len(self.time_history_days),
        }


def compute_geomagnetic_field_b(altitude_km: float, colatitude_rad: float = math.pi / 2.0) -> Tuple[float, float, float]:
    """
    Computes Earth's magnetic dipole components (B_radial, B_theta, B_total) in Tesla.
    Dipole moment M_E ~ 7.94e15 T*m^3.
    """
    r_m = (EARTH_RADIUS_KM + altitude_km) * 1000.0
    m_dipole = 7.94e15  # T * m^3

    b_r = 2.0 * (m_dipole / (r_m ** 3)) * math.cos(colatitude_rad)
    b_theta = (m_dipole / (r_m ** 3)) * math.sin(colatitude_rad)
    b_total = math.sqrt(b_r ** 2 + b_theta ** 2)
    return b_r, b_theta, b_total


def compute_plasma_density(altitude_km: float) -> float:
    """
    Approximates daytime LEO electron plasma density N_e in m^-3.
    Peak ionosphere F2-layer occurs around 300-350 km.
    """
    if altitude_km < 150.0:
        return 1.0e10
    elif altitude_km < 350.0:
        # Rising to F2 peak
        return 1.0e11 * math.exp((altitude_km - 150.0) / 100.0)
    elif altitude_km < 600.0:
        # F2 decay
        return 5.0e11 * math.exp(-(altitude_km - 350.0) / 150.0)
    else:
        # Upper topside ionosphere
        return 1.0e11 * math.exp(-(altitude_km - 600.0) / 250.0)


def simulate_electrodynamic_tether_decay(
    satellite_mass_kg: float = 300.0,
    initial_altitude_km: float = 800.0,
    target_altitude_km: float = 250.0,
    inclination_deg: float = 51.6,
    tether_specs: Optional[ElectrodynamicTetherSpecs] = None,
    dt_days: float = 2.0,
) -> TetherDeorbitResult:
    """
    Simulates secular orbit descent of an uncooperative satellite using an Electrodynamic Tether.
    """
    if tether_specs is None:
        tether_specs = ElectrodynamicTetherSpecs()

    mu_m = MU_EARTH_KM3_S2 * 1e9
    total_mass = satellite_mass_kg + tether_specs.tether_mass_kg

    curr_alt = float(initial_altitude_km)
    curr_time_days = 0.0

    time_hist = [0.0]
    alt_hist = [curr_alt]

    accum_force_n = 0.0
    accum_current_a = 0.0
    peak_emf_v = 0.0
    steps = 0

    colat = math.radians(90.0 - (inclination_deg * 0.6))  # representative orbit-average magnetic latitude

    while curr_alt > target_altitude_km and curr_time_days < 1000.0:
        r_m = (EARTH_RADIUS_KM + curr_alt) * 1000.0
        v_orb = math.sqrt(mu_m / r_m)

        # Geomagnetic field
        _, _, b_total = compute_geomagnetic_field_b(curr_alt, colat)

        # Induced EMF across vertically deployed tether: E = v_orb * B_total * L_tether * cos(i)
        emf = v_orb * b_total * tether_specs.length_m * math.cos(math.radians(inclination_deg * 0.3))
        if emf > peak_emf_v:
            peak_emf_v = emf

        # Plasma OML current estimation
        n_e = compute_plasma_density(curr_alt)
        e_charge = 1.602e-19
        m_electron = 9.109e-31

        r_w_m = tether_specs.wire_radius_mm * 1e-3
        # OML collection current capability:
        v_bias = max(1.0, emf * tether_specs.bias_efficiency)
        i_oml_limit = (2.0 * e_charge * n_e * r_w_m * tether_specs.length_m / math.pi) * math.sqrt(2.0 * e_charge * v_bias / m_electron)

        # Ohmic limitation
        total_resistance = tether_specs.ohmic_resistance_ohms + tether_specs.cathode_impedance_ohms
        i_circuit_limit = emf / max(1.0, total_resistance)

        # Effective current is limited by either plasma collection or circuit loop impedance
        i_eff = float(np.clip(min(i_oml_limit, i_circuit_limit), 0.01, 2.5))

        # Lorentz Drag Force: F_L = I * L * B (in direction opposite to orbital velocity)
        f_lorentz_n = i_eff * tether_specs.length_m * b_total

        accum_force_n += f_lorentz_n
        accum_current_a += i_eff
        steps += 1

        # Rate of energy dissipation: dE/dt = - F_L * v_orb
        # Since E_orb = - mu * m / (2 * r), da/dt = - 2 * sqrt(r^3 / mu) * (F_L / m)
        da_dt_m_s = - 2.0 * math.sqrt((r_m ** 3) / mu_m) * (f_lorentz_n / total_mass)
        delta_alt_km = (da_dt_m_s * (dt_days * 86400.0)) / 1000.0

        curr_alt += delta_alt_km
        curr_time_days += dt_days

        time_hist.append(round(curr_time_days, 1))
        alt_hist.append(max(target_altitude_km, round(curr_alt, 2)))

    avg_force_mn = (accum_force_n / max(1, steps)) * 1000.0
    avg_current = accum_current_a / max(1, steps)

    # Chemical propellant saved: Delta-V equivalent for Hohmann transfer from 800 km to 250 km:
    # Delta-V ~ 150 m/s. For Isp = 300 s, m_prop = m * (1 - exp(-dv / (Isp * g0)))
    delta_v_equiv = 150.0  # m/s
    prop_saved = satellite_mass_kg * (1.0 - math.exp(-delta_v_equiv / (300.0 * 9.80665)))

    return TetherDeorbitResult(
        initial_altitude_km=initial_altitude_km,
        final_altitude_km=curr_alt,
        deorbit_time_days=curr_time_days,
        average_lorentz_drag_force_mn=avg_force_mn,
        peak_induced_emf_volts=peak_emf_v,
        average_current_amps=avg_current,
        tether_mass_kg=tether_specs.tether_mass_kg,
        propellant_saved_kg=prop_saved,
        altitude_history_km=alt_hist,
        time_history_days=time_hist,
    )


def compare_all_deorbit_methods(
    satellite_mass_kg: float = 250.0,
    initial_altitude_km: float = 750.0,
) -> Dict[str, Any]:
    """
    Comprehensive multi-technology trade-off for satellite deorbiting.
    """
    # 1. Natural Drag (years -> days)
    # At 750 km, natural decay takes ~15-20 years
    nat_days = 18.0 * 365.25

    # 2. Deployable Drag Sail (5 m^2 sail)
    # Area increase x5 -> decay ~3.5 years
    sail_days = 3.5 * 365.25
    sail_mass_kg = 4.5

    # 3. Chemical Propulsion (Monopropellant Hydrazine, Isp = 220s)
    # Instantaneous burn, but high dry/propellant mass penalty
    # Delta-v = ~145 m/s
    dv = 145.0
    chem_prop_mass = satellite_mass_kg * (math.exp(dv / (220.0 * 9.80665)) - 1.0)
    chem_burn_time_min = 12.0

    # 4. Electric Propulsion (Hall Effect Thruster, Isp = 1600s, Thrust = 40 mN)
    ep_thrust_n = 0.040
    ep_prop_mass = satellite_mass_kg * (math.exp(dv / (1600.0 * 9.80665)) - 1.0)
    ep_burn_days = (satellite_mass_kg * dv) / (ep_thrust_n * 86400.0)

    # 5. Electrodynamic Tether (3 km aluminum tape)
    edt_sim = simulate_electrodynamic_tether_decay(
        satellite_mass_kg=satellite_mass_kg,
        initial_altitude_km=initial_altitude_km,
        target_altitude_km=250.0,
    )

    return {
        "initial_altitude_km": initial_altitude_km,
        "satellite_mass_kg": satellite_mass_kg,
        "methods": {
            DeorbitMethod.NATURAL_DRAG.value: {
                "duration_days": round(nat_days, 1),
                "subsystem_mass_kg": 0.0,
                "propellant_mass_kg": 0.0,
                "power_required_w": 0.0,
                "reliability_score": 0.35,
            },
            DeorbitMethod.DEPLOYABLE_DRAG_SAIL.value: {
                "duration_days": round(sail_days, 1),
                "subsystem_mass_kg": sail_mass_kg,
                "propellant_mass_kg": 0.0,
                "power_required_w": 2.0,
                "reliability_score": 0.78,
            },
            DeorbitMethod.CHEMICAL_RETRO_PROPULSION.value: {
                "duration_days": round(chem_burn_time_min / 1440.0, 3),
                "subsystem_mass_kg": 12.0,
                "propellant_mass_kg": round(chem_prop_mass, 2),
                "power_required_w": 25.0,
                "reliability_score": 0.92,
            },
            DeorbitMethod.ELECTRIC_PROPULSION.value: {
                "duration_days": round(ep_burn_days, 1),
                "subsystem_mass_kg": 18.0,
                "propellant_mass_kg": round(ep_prop_mass, 2),
                "power_required_w": 450.0,
                "reliability_score": 0.88,
            },
            DeorbitMethod.ELECTRODYNAMIC_TETHER.value: {
                "duration_days": round(edt_sim.deorbit_time_days, 1),
                "subsystem_mass_kg": round(edt_sim.tether_mass_kg + 3.0, 2),
                "propellant_mass_kg": 0.0,
                "power_required_w": 15.0,
                "reliability_score": 0.85,
            },
        },
        "recommended_method": DeorbitMethod.ELECTRODYNAMIC_TETHER.value if initial_altitude_km > 650.0 else DeorbitMethod.DEPLOYABLE_DRAG_SAIL.value,
    }
