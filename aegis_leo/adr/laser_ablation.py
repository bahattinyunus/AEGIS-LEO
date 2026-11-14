"""
AEGIS-LEO: Laser Orbital Debris Remediation (LODR) & Photon Ablation Physics
Space-based & Ground-based High-Power Pulsed Laser Momentum Impartation.

Physical Formulations:
1. Beam Propagation & Diffraction:
   theta_div = 1.22 * lambda / D_director
   spot_diameter = max(d_debris, 2.0 * slant_range * tan(theta_div / 2))
   spot_area = pi * (spot_diameter / 2)^2
2. Fluence on Target:
   F_target = E_pulse * tau_atm / spot_area  (J/cm^2 or J/m^2)
3. Momentum Coupling Coefficient (Recoil Impulse):
   C_m = C_m_peak * exp( - (ln(F / F_opt))^2 / (2 * sigma_f^2) )  (N*s / J or micro-N*s / J)
4. Impulse & Velocity Increment:
   delta_p = C_m * E_pulse * n_pulses
   delta_v = delta_p / m_debris
5. Perigee Drop (Gauss Variational / Vis-Viva):
   delta_r_p = 2 * r_a * r_p * delta_v / (v_p * (r_a + r_p))
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from aegis_leo.constants import EARTH_RADIUS_KM, MU_EARTH_KM3_S2


class LaserPlatformType(str, Enum):
    GROUND_BASED = "GROUND_BASED"
    SPACE_BASED_ORBITAL = "SPACE_BASED_ORBITAL"


@dataclass
class LaserStationSpecs:
    """Laser remediation system parameters."""
    platform: LaserPlatformType = LaserPlatformType.SPACE_BASED_ORBITAL
    wavelength_nm: float = 1064.0        # Nd:YAG / Fiber laser (1064 nm)
    pulse_energy_joules: float = 100.0   # 100 J per pulse
    repetition_rate_hz: float = 100.0    # 100 Hz (10 kW average beam power)
    director_diameter_m: float = 1.5     # 1.5 m beam director telescope
    beam_quality_m2: float = 1.15        # Near diffraction-limited M^2
    pointing_jitter_urad: float = 0.5    # 0.5 microradian tracking jitter

    @property
    def average_power_kw(self) -> float:
        return (self.pulse_energy_joules * self.repetition_rate_hz) / 1000.0


@dataclass
class DebrisTargetSpecs:
    """Characteristics of the untracked or hazardous debris target."""
    target_id: str = "DEBRIS-LEO-FRAG-441"
    mass_kg: float = 1.25                 # 1.25 kg lethal fragment
    effective_diameter_cm: float = 12.0   # 12 cm cross section
    material: str = "Aluminum 7075-T6"
    perigee_alt_km: float = 450.0
    apogee_alt_km: float = 780.0
    optical_reflectivity: float = 0.15

    @property
    def cross_section_area_m2(self) -> float:
        r_m = (self.effective_diameter_cm * 0.01) / 2.0
        return math.pi * (r_m ** 2)


@dataclass
class LaserEngagementResult:
    """Outcomes of laser engagement pass against debris."""
    target_id: str
    slant_range_km: float
    engagement_duration_s: float
    total_pulses_fired: int
    pulses_on_target: int
    average_fluence_j_cm2: float
    recoil_momentum_coupling_u_ns_j: float
    total_impulse_n_s: float
    delta_v_m_s: float
    perigee_drop_km: float
    new_perigee_alt_km: float
    deorbit_accelerated_years: float
    fragmentation_risk_score: float  # Must remain < 0.05 to avoid shattering
    status: str = "PASS_COMPLETED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "slant_range_km": round(self.slant_range_km, 1),
            "engagement_duration_s": round(self.engagement_duration_s, 1),
            "total_pulses_fired": self.total_pulses_fired,
            "pulses_on_target": self.pulses_on_target,
            "average_fluence_j_cm2": round(self.average_fluence_j_cm2, 2),
            "recoil_momentum_coupling_u_ns_j": round(self.recoil_momentum_coupling_u_ns_j, 1),
            "total_impulse_n_s": round(self.total_impulse_n_s, 3),
            "delta_v_m_s": round(self.delta_v_m_s, 3),
            "perigee_drop_km": round(self.perigee_drop_km, 2),
            "new_perigee_alt_km": round(self.new_perigee_alt_km, 2),
            "deorbit_accelerated_years": round(self.deorbit_accelerated_years, 1),
            "fragmentation_risk_score": round(self.fragmentation_risk_score, 4),
            "status": self.status,
        }


def compute_atmospheric_laser_transmission(zenith_angle_rad: float, wavelength_nm: float = 1064.0) -> float:
    """Atmospheric extinction for ground-to-space laser transmission."""
    # Optical depth tau0 ~ 0.08 at 1064 nm in clear sky
    tau_0 = 0.075 * (1064.0 / max(100.0, wavelength_nm)) ** 0.8
    sec_zeta = 1.0 / max(0.1, math.cos(zenith_angle_rad))
    return float(math.exp(-tau_0 * sec_zeta))


def compute_momentum_coupling(fluence_j_cm2: float) -> Tuple[float, float]:
    """
    Computes plasma ablation momentum coupling Cm (micro-N*s / J) and fragmentation risk.
    For aluminum, optimal fluence F_opt ~ 5.0 - 15.0 J/cm^2 yields Cm ~ 40-70 micro-N*s / J.
    Threshold for surface ablation is ~ 1.5 J/cm^2.
    """
    if fluence_j_cm2 < 1.0:
        # Pure radiation photon pressure: Cm = (1 + R) / c ~ 0.0067 micro-N*s / J
        return 0.007, 0.0001

    f_opt = 8.5   # J/cm^2
    cm_peak = 55.0  # micro-N*s / J

    # Gaussian log-bell response
    log_ratio = math.log(max(1e-4, fluence_j_cm2 / f_opt))
    cm = cm_peak * math.exp(-(log_ratio ** 2) / (2.0 * (1.1 ** 2)))

    # Fragmentation risk rises sharply if fluence exceeds 80 J/cm^2 (thermal shock / spallation)
    frag_risk = 0.01 + 0.99 / (1.0 + math.exp(-(fluence_j_cm2 - 75.0) / 8.0))

    return float(cm), float(frag_risk)


def simulate_laser_debris_engagement(
    laser: Optional[LaserStationSpecs] = None,
    debris: Optional[DebrisTargetSpecs] = None,
    slant_range_km: float = 350.0,
    engagement_duration_s: float = 45.0,
    zenith_angle_deg: float = 25.0,
) -> LaserEngagementResult:
    """
    Simulates a high-power pulsed laser engagement pass to decelerate and lower debris perigee.
    """
    if laser is None:
        laser = LaserStationSpecs()
    if debris is None:
        debris = DebrisTargetSpecs()

    wavelength_m = laser.wavelength_nm * 1e-9
    # Beam divergence with beam quality factor M^2
    theta_div_rad = laser.beam_quality_m2 * 1.22 * wavelength_m / laser.director_diameter_m

    # Spot diameter at debris range (meters)
    range_m = slant_range_km * 1000.0
    spot_diameter_m = 2.0 * range_m * math.tan(theta_div_rad / 2.0)
    spot_area_m2 = math.pi * ((spot_diameter_m / 2.0) ** 2)
    spot_area_cm2 = spot_area_m2 * 10000.0

    # Atmospheric transmission
    if laser.platform == LaserPlatformType.GROUND_BASED:
        tau_atm = compute_atmospheric_laser_transmission(math.radians(zenith_angle_deg), laser.wavelength_nm)
    else:
        tau_atm = 1.0  # Space-based has vacuum transmission

    # Energy delivered per pulse onto spot area
    energy_on_target_per_pulse = laser.pulse_energy_joules * tau_atm
    fluence_j_cm2 = energy_on_target_per_pulse / max(1.0, spot_area_cm2)

    cm_u_ns_j, frag_risk = compute_momentum_coupling(fluence_j_cm2)
    cm_si = cm_u_ns_j * 1e-6  # N*s / J

    total_pulses = int(laser.repetition_rate_hz * engagement_duration_s)

    # Cross section intercept fraction: fraction of spot intercepted by debris
    debris_area = debris.cross_section_area_m2
    intercept_ratio = min(1.0, debris_area / max(1e-6, spot_area_m2))
    effective_pulses = int(total_pulses * (1.0 - laser.pointing_jitter_urad * 0.1))

    # Total recoil impulse (N*s)
    energy_absorbed_total = effective_pulses * energy_on_target_per_pulse * intercept_ratio
    total_impulse = cm_si * energy_absorbed_total

    # Delta-V imparted (m/s)
    delta_v = total_impulse / max(0.01, debris.mass_kg)

    # Perigee lowering from vis-viva / Gauss variational equation:
    # At apogee (optimal deceleration point), a retro Delta-V lowers perigee:
    r_a = (EARTH_RADIUS_KM + debris.apogee_alt_km) * 1000.0
    r_p = (EARTH_RADIUS_KM + debris.perigee_alt_km) * 1000.0
    mu_m = MU_EARTH_KM3_S2 * 1e9

    v_a = math.sqrt(2.0 * mu_m * r_p / (r_a * (r_a + r_p)))
    # delta_r_p ~ 2 * r_p * r_a * delta_v / (v_a * (r_a + r_p))
    delta_rp_m = 2.0 * r_p * r_a * delta_v / max(1.0, (v_a * (r_a + r_p)))
    delta_rp_km = delta_rp_m / 1000.0

    new_perigee = max(100.0, debris.perigee_alt_km - delta_rp_km)

    # Natural lifetime reduction estimation:
    # Dropping perigee from 450 km to 350 km accelerates decay exponentially
    lifetime_initial = 12.0 * math.exp((debris.perigee_alt_km - 400.0) / 50.0)
    lifetime_final = 12.0 * math.exp((new_perigee - 400.0) / 50.0)
    decay_acceleration_years = max(0.1, lifetime_initial - lifetime_final)

    return LaserEngagementResult(
        target_id=debris.target_id,
        slant_range_km=slant_range_km,
        engagement_duration_s=engagement_duration_s,
        total_pulses_fired=total_pulses,
        pulses_on_target=effective_pulses,
        average_fluence_j_cm2=fluence_j_cm2,
        recoil_momentum_coupling_u_ns_j=cm_u_ns_j,
        total_impulse_n_s=total_impulse,
        delta_v_m_s=delta_v,
        perigee_drop_km=delta_rp_km,
        new_perigee_alt_km=new_perigee,
        deorbit_accelerated_years=decay_acceleration_years,
        fragmentation_risk_score=frag_risk,
    )
