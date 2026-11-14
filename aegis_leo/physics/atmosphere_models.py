"""
Advanced Thermospheric Density Models & Comparative Physics Engine
Implements Exponential, Jacchia-77 / Jacchia-Roberts, and NRLMSISE-00 approximations
for multi-model atmospheric drag sensitivity analysis.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from aegis_leo.constants import EARTH_RADIUS_KM, QUIET_KP, QUIET_F10_7


class AtmosphereModelType:
    EXPONENTIAL = "EXPONENTIAL"
    JACCHIA_77 = "JACCHIA_77"
    NRLMSISE_00 = "NRLMSISE_00"


def jacchia_exospheric_temperature(
    f10_7: float,
    f10_7_avg: float,
    kp: float,
) -> float:
    """
    Calculate exospheric nighttime minimum temperature T_c and geomagnetic delta T_g
    using empirical Jacchia formulas:
    T_c = 379 + 3.24 * F_bar + 1.3 * (F10.7 - F_bar)
    Delta T_g = 28.0 * Kp + 0.03 * exp(Kp)
    T_inf = T_c * (1 + 0.3 * cos^4(theta_solar / 2)) + Delta T_g
    """
    t_c = 379.0 + 3.24 * f10_7_avg + 1.3 * (f10_7 - f10_7_avg)
    delta_t_g = 28.0 * kp + 0.03 * np.exp(min(kp, 9.0))
    # Diurnal bulge mean factor ~ 1.15
    t_inf = t_c * 1.15 + delta_t_g
    return float(t_inf)


def compute_jacchia_density(
    altitude_km: float,
    kp: float = QUIET_KP,
    f10_7: float = QUIET_F10_7,
    f10_7_avg: float = QUIET_F10_7,
) -> Dict[str, float]:
    """
    Jacchia-77 density approximation based on exospheric temperature T_inf and barometric integration.
    """
    alt = np.clip(altitude_km, 100.0, 1100.0)
    t_inf = jacchia_exospheric_temperature(f10_7, f10_7_avg, kp)

    # Reference density at 120 km (kg / m^3)
    rho_120 = 2.4e-8
    # Mean molecular weight varies from 28 at 120 km down to 16 at 500 km and 4 at 1000 km
    z = (alt - 120.0) / 880.0
    mean_mol_weight = 28.0 * (1.0 - z) + 4.0 * z

    # Effective scale height H = R* T / (M * g)
    # Calibrated scale height in km
    h_eff_km = (t_inf / 1000.0) * (55.0 / (mean_mol_weight / 20.0)) * (1.0 + alt / EARTH_RADIUS_KM)**2

    rho_jacchia = rho_120 * np.exp(-(alt - 120.0) / h_eff_km)

    return {
        "density_kg_m3": float(rho_jacchia),
        "exospheric_temp_k": float(t_inf),
        "effective_scale_height_km": float(h_eff_km),
        "mean_molecular_weight": float(mean_mol_weight),
    }


def compute_nrlmsise_density_approx(
    altitude_km: float,
    kp: float = QUIET_KP,
    f10_7: float = QUIET_F10_7,
) -> Dict[str, Any]:
    """
    Analytical NRLMSISE-00 style multi-species decomposition.
    Evaluates individual constituent number densities (O, N2, O2, He, H, Ar)
    and sums total mass density rho = sum(n_i * m_i).
    """
    alt = np.clip(altitude_km, 120.0, 1000.0)

    # Temperature profile
    t_inf = jacchia_exospheric_temperature(f10_7, f10_7, kp)
    t_local = t_inf - (t_inf - 200.0) * np.exp(-0.015 * (alt - 120.0))

    # Constituent species molecular masses in amu
    mass_amu = {"O": 16.0, "N2": 28.0, "O2": 32.0, "He": 4.0, "H": 1.0}
    amu_to_kg = 1.660539e-27

    # Base number densities at 120 km (particles / m^3)
    base_n = {
        "O": 1.1e17,
        "N2": 3.8e17,
        "O2": 6.2e16,
        "He": 4.5e13,
        "H": 1.0e11,
    }

    species_densities = {}
    total_mass_density = 0.0

    for sp, m_amu in mass_amu.items():
        # Diffusive equilibrium scale height H_i = k_B * T / (m_i * g)
        h_i = (t_local / 800.0) * (28.0 / m_amu) * 45.0  # km
        # Geomagnetic storm enhancement for atomic oxygen and heavy constituents
        storm_boost = 1.0 + (0.15 * (kp - 2.0) if kp > 2.0 else 0.0) * (m_amu / 20.0)
        n_sp = base_n[sp] * np.exp(-(alt - 120.0) / h_i) * max(0.5, storm_boost)
        rho_sp = n_sp * (m_amu * amu_to_kg)
        species_densities[sp] = float(rho_sp)
        total_mass_density += rho_sp

    return {
        "total_density_kg_m3": float(total_mass_density),
        "temperature_k": float(t_local),
        "species_mass_density_kg_m3": species_densities,
        "primary_constituent": "O" if alt < 600.0 else "He",
    }


def compare_atmospheric_models(
    altitudes_km: List[float],
    kp: float = 8.0,
    f10_7: float = 240.0,
) -> Dict[str, Any]:
    """
    Execute comparative benchmarking between Exponential, Jacchia-77, and NRLMSISE-00.
    """
    from aegis_leo.physics.atmosphere import compute_atmospheric_density

    results = []
    for alt in altitudes_km:
        rho_exp = compute_atmospheric_density(alt, kp=kp, f10_7=f10_7)
        res_jac = compute_jacchia_density(alt, kp=kp, f10_7=f10_7)
        res_msis = compute_nrlmsise_density_approx(alt, kp=kp, f10_7=f10_7)

        rho_jac = res_jac["density_kg_m3"]
        rho_msis = res_msis["total_density_kg_m3"]

        mean_rho = (rho_exp + rho_jac + rho_msis) / 3.0
        discrepancy_pct = (max(rho_exp, rho_jac, rho_msis) - min(rho_exp, rho_jac, rho_msis)) / max(mean_rho, 1e-25) * 100.0

        results.append({
            "altitude_km": alt,
            "exponential_kg_m3": rho_exp,
            "jacchia_77_kg_m3": rho_jac,
            "nrlmsise_00_kg_m3": rho_msis,
            "mean_density_kg_m3": mean_rho,
            "discrepancy_percentage": round(discrepancy_pct, 1),
            "dominant_species": res_msis["primary_constituent"],
            "exospheric_temp_k": res_jac["exospheric_temp_k"],
        })

    return {
        "kp_index": kp,
        "f10_7_flux": f10_7,
        "comparison_table": results,
    }
