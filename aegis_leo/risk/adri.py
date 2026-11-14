"""
Adaptive Debris & Drag Risk Index (A-DRI) Engine
Core mathematical risk model for AEGIS-LEO.
Unifies Collision Probability, Space Weather Drag Uncertainty, NASA ORDEM Density,
and Fuel Feasibility into a real-time risk score in [0.0, 1.0].
"""

from typing import Dict, Any, Optional
import numpy as np
from enum import Enum
from aegis_leo.constants import (
    COLLISION_PROBABILITY_THRESHOLD,
    QUIET_KP,
    QUIET_F10_7,
)
from aegis_leo.risk.collision import compute_collision_probability
from aegis_leo.risk.debris_density import get_spatial_debris_density


class RiskLevel(str, Enum):
    NOMINAL = "NOMINAL"        # < 0.30 (Green)
    ELEVATED = "ELEVATED"      # 0.30 - 0.59 (Yellow)
    HIGH = "HIGH"              # 0.60 - 0.79 (Orange)
    CRITICAL = "CRITICAL"      # >= 0.80 (Red)


class ActionCommand(str, Enum):
    HOLD_NOMINAL = "HOLD_NOMINAL"
    OBSERVE_CLOSE_TRACK = "OBSERVE_CLOSE_TRACK"
    SCHEDULE_PROACTIVE_MICRO_BURN = "SCHEDULE_PROACTIVE_MICRO_BURN"
    EMERGENCY_AUTONOMOUS_EVASION = "EMERGENCY_AUTONOMOUS_EVASION"


class ADRICalculator:
    """
    Computes real-time Adaptive Debris & Drag Risk Index (A-DRI).
    """

    def __init__(
        self,
        w1_cprob: float = 0.40,
        w2_wdrag: float = 0.25,
        w3_dord: float = 0.20,
        w4_fdelta_v: float = 0.15,
    ):
        total_w = w1_cprob + w2_wdrag + w3_dord + w4_fdelta_v
        self.w1 = w1_cprob / total_w
        self.w2 = w2_wdrag / total_w
        self.w3 = w3_dord / total_w
        self.w4 = w4_fdelta_v / total_w

    def compute_w_drag(
        self,
        area_m2: float,
        mass_kg: float,
        kp: float,
        f10_7: float,
        altitude_km: float,
        along_track_disp_km: float = 0.0,
    ) -> float:
        """
        Compute Space Weather orbital uncertainty index W_drag in [0.0, 1.0].
        Objects with high area-to-mass ratio (A/m) experience severe along-track drift
        during geomagnetic storms (Kp >= 5), expanding collision volume uncertainty.
        """
        area_to_mass = area_m2 / max(mass_kg, 1.0)  # m^2 / kg
        # Sensitivity scaling around typical 0.02 m^2/kg
        atm_factor = float(np.clip(0.6 + (area_to_mass / 0.02) * 0.4, 0.5, 1.8))

        # Geomagnetic disturbance factor [0.0, 1.0]
        kp_val = max(0.0, float(kp))
        kp_factor = float(np.clip((kp_val - 2.0) / 6.0, 0.0, 1.0))

        # Solar radio flux factor [0.0, 1.0]
        f_val = max(60.0, float(f10_7))
        flux_factor = float(np.clip((f_val - 70.0) / 200.0, 0.0, 1.0))

        # Altitude susceptibility in LEO [0.5, 1.4]
        alt_factor = float(np.clip(1.3 - (altitude_km - 300.0) / 600.0, 0.5, 1.4))

        # Along-track displacement contribution (normalized to 10 km)
        disp_factor = float(np.clip(abs(along_track_disp_km) / 8.0, 0.0, 1.0))

        raw_w_drag = (0.30 * kp_factor + 0.20 * flux_factor + 0.50 * disp_factor) * atm_factor * alt_factor
        return float(np.clip(raw_w_drag, 0.0, 1.0))

    def compute_f_delta_v(
        self,
        required_dv_m_s: float,
        remaining_propellant_kg: float,
        initial_mass_kg: float,
        isp_seconds: float = 220.0,
    ) -> float:
        """
        Compute Propellant Feasibility index F_Delta_v in [0.0, 1.0].
        Quantifies the urgency/penalty of propellant expenditure.
        Higher value indicates propellant scarcity or excessive delta-v requirement.
        """
        if initial_mass_kg <= 0 or remaining_propellant_kg < 0:
            return 1.0

        # Fractional propellant remaining
        fuel_ratio = remaining_propellant_kg / initial_mass_kg
        fuel_scarcity = 1.0 - np.clip(fuel_ratio / 0.15, 0.0, 1.0)  # 1.0 if fuel < 0.01

        # Required delta-v magnitude factor (1 m/s is nominal micro-burn, 15 m/s is large evasion)
        dv_penalty = np.clip(required_dv_m_s / 12.0, 0.0, 1.0)

        f_val = 0.5 * fuel_scarcity + 0.5 * dv_penalty
        return float(np.clip(f_val, 0.0, 1.0))

    def compute_c_prob_normalized(
        self,
        collision_prob: float,
        threshold: float = COLLISION_PROBABILITY_THRESHOLD,
    ) -> float:
        """
        Normalize collision probability Pc onto a smooth logarithmic [0.0, 1.0] scale.
        Pc = 1e-7 -> ~0.1
        Pc = 1e-4 (NASA screening threshold) -> 0.60
        Pc = 1e-3 -> 0.85
        Pc >= 1e-2 -> 1.0
        """
        if collision_prob <= 1e-9:
            return 0.0
        log_pc = np.log10(max(collision_prob, 1e-9))
        # Map log_pc from [-9, -2] to [0.0, 1.0]
        norm = (log_pc - (-9.0)) / ((-2.0) - (-9.0))
        return float(np.clip(norm, 0.0, 1.0))

    def evaluate_risk(
        self,
        miss_distance_km: float,
        altitude_km: float,
        inclination_deg: float,
        area_m2: float = 1.5,
        mass_kg: float = 120.0,
        kp: float = QUIET_KP,
        f10_7: float = QUIET_F10_7,
        along_track_disp_km: float = 0.0,
        required_dv_m_s: float = 0.5,
        remaining_propellant_kg: float = 12.0,
        sigma_pos_km: float = 0.3,
        combined_radius_m: float = 8.0,
    ) -> Dict[str, Any]:
        """
        Evaluate complete A-DRI score and decompose each factor.
        """
        # 1. Raw & Normalized Collision Probability
        raw_pc = compute_collision_probability(
            miss_distance_km=miss_distance_km,
            combined_radius_m=combined_radius_m,
            sigma_pos_km=sigma_pos_km,
        )
        c_prob_norm = self.compute_c_prob_normalized(raw_pc)

        # 2. Space Weather Drag Uncertainty
        w_drag = self.compute_w_drag(
            area_m2=area_m2,
            mass_kg=mass_kg,
            kp=kp,
            f10_7=f10_7,
            altitude_km=altitude_km,
            along_track_disp_km=along_track_disp_km,
        )

        # 3. Spatial Debris Density (NASA ORDEM)
        debris_info = get_spatial_debris_density(altitude_km, inclination_deg)
        d_ord = debris_info["normalized_debris_factor"]

        # 4. Propellant Feasibility Index
        f_delta_v = self.compute_f_delta_v(
            required_dv_m_s=required_dv_m_s,
            remaining_propellant_kg=remaining_propellant_kg,
            initial_mass_kg=mass_kg,
        )

        # Composite A-DRI
        adri_score = (
            self.w1 * c_prob_norm +
            self.w2 * w_drag +
            self.w3 * d_ord +
            self.w4 * f_delta_v
        )
        adri_score = float(np.clip(adri_score, 0.0, 1.0))

        # Risk Classification
        if adri_score >= 0.80 or raw_pc >= 1e-3:
            level = RiskLevel.CRITICAL
            action = ActionCommand.EMERGENCY_AUTONOMOUS_EVASION
        elif adri_score >= 0.60 or raw_pc >= 1e-4:
            level = RiskLevel.HIGH
            action = ActionCommand.SCHEDULE_PROACTIVE_MICRO_BURN
        elif adri_score >= 0.30:
            level = RiskLevel.ELEVATED
            action = ActionCommand.OBSERVE_CLOSE_TRACK
        else:
            level = RiskLevel.NOMINAL
            action = ActionCommand.HOLD_NOMINAL

        return {
            "a_dri": adri_score,
            "risk_level": level.value,
            "action_command": action.value,
            "components": {
                "c_prob_raw": raw_pc,
                "c_prob_normalized": c_prob_norm,
                "w_drag": w_drag,
                "d_ord": d_ord,
                "f_delta_v": f_delta_v,
            },
            "weights": {
                "w1_cprob": self.w1,
                "w2_wdrag": self.w2,
                "w3_dord": self.w3,
                "w4_fdelta_v": self.w4,
            },
            "debris_spatial_density_km3": debris_info["spatial_density_objects_per_km3"],
            "miss_distance_km": miss_distance_km,
            "altitude_km": altitude_km,
            "kp_index": kp,
        }
