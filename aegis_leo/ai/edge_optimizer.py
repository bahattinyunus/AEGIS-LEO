"""
Edge-AI Autonomous Collision Avoidance Optimizer
Calculates optimal, fuel-minimizing altitude and phasing burns (Delta-v)
using astrodynamics (Vis-Viva, phasing, Tsiolkovsky) for onboard flight computers.
"""

from typing import Dict, Any, List
import numpy as np
from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    EARTH_MU,
    STANDARD_GRAVITY,
)
from aegis_leo.risk.adri import ADRICalculator, RiskLevel


class ManeuverType:
    PROACTIVE_PHASING = "PROACTIVE_PHASING"
    ALTITUDE_RAISE = "ALTITUDE_RAISE"
    EMERGENCY_CROSS_TRACK = "EMERGENCY_CROSS_TRACK"
    STORM_COMPENSATION_MICRO_BURN = "STORM_COMPENSATION_MICRO_BURN"


class EdgeManeuverOptimizer:
    """
    Onboard autonomous collision avoidance and propellant optimization engine.
    """

    def __init__(
        self,
        satellite_mass_kg: float = 120.0,
        fuel_remaining_kg: float = 12.0,
        isp_seconds: float = 220.0,
        thrust_newtons: float = 0.5,
    ):
        self.mass_kg = float(satellite_mass_kg)
        self.fuel_kg = float(fuel_remaining_kg)
        self.isp_s = float(isp_seconds)
        self.thrust_n = float(thrust_newtons)
        self.adri_calc = ADRICalculator()

    def tsiolkovsky_fuel_cost(self, delta_v_m_s: float, current_mass_kg: float) -> float:
        """
        Calculate propellant consumed (kg) using Tsiolkovsky rocket equation:
        delta_m = m_0 * (1 - exp(-delta_v / (Isp * g0)))
        """
        ve = self.isp_s * STANDARD_GRAVITY
        dv = max(0.0, float(delta_v_m_s))
        fuel_mass = current_mass_kg * (1.0 - np.exp(-dv / ve))
        return float(fuel_mass)

    def compute_hohmann_altitude_burn(
        self,
        current_altitude_km: float,
        target_delta_alt_km: float = 3.5,
    ) -> Dict[str, float]:
        """
        Compute Delta-v (m/s) required for an orbit raising burn using Vis-Viva.
        """
        r1 = EARTH_RADIUS_KM + current_altitude_km
        r2 = r1 + target_delta_alt_km
        a_trans = (r1 + r2) / 2.0

        v1 = np.sqrt(EARTH_MU / r1)  # km/s
        v_trans1 = np.sqrt(EARTH_MU * (2.0 / r1 - 1.0 / a_trans))  # km/s

        dv1_km_s = abs(v_trans1 - v1)
        dv1_m_s = dv1_km_s * 1000.0

        fuel_used_kg = self.tsiolkovsky_fuel_cost(dv1_m_s, self.mass_kg)
        burn_duration_s = (self.mass_kg * dv1_m_s) / max(self.thrust_n, 0.05)

        return {
            "delta_v_m_s": dv1_m_s,
            "fuel_consumed_kg": fuel_used_kg,
            "burn_duration_seconds": burn_duration_s,
            "target_altitude_km": current_altitude_km + target_delta_alt_km,
        }

    def compute_phasing_burn(
        self,
        current_altitude_km: float,
        desired_separation_km: float = 12.0,
        orbits_to_encounter: float = 2.0,
    ) -> Dict[str, float]:
        """
        Compute along-track phasing burn Delta-v to introduce safe separation
        ahead of conjunction point.
        """
        r1 = EARTH_RADIUS_KM + current_altitude_km
        v_circ = np.sqrt(EARTH_MU / r1)  # km/s
        t_period_s = 2.0 * np.pi * np.sqrt((r1**3) / EARTH_MU)

        # Time shift needed = desired_separation_km / v_circ
        delta_t_needed = desired_separation_km / v_circ  # seconds

        # Change in orbital period per orbit
        delta_t_per_rev = delta_t_needed / max(orbits_to_encounter, 0.5)

        # Delta_a = (2/3) * (a / T) * delta_T
        delta_a_km = (2.0 / 3.0) * (r1 / t_period_s) * delta_t_per_rev

        # Vis-Viva relationship for small Delta-a: Delta-v = (v / (2 * a)) * Delta-a
        dv_km_s = (v_circ / (2.0 * r1)) * abs(delta_a_km)
        dv_m_s = dv_km_s * 1000.0

        fuel_used_kg = self.tsiolkovsky_fuel_cost(dv_m_s, self.mass_kg)
        burn_duration_s = (self.mass_kg * dv_m_s) / max(self.thrust_n, 0.05)

        return {
            "delta_v_m_s": dv_m_s,
            "fuel_consumed_kg": fuel_used_kg,
            "burn_duration_seconds": burn_duration_s,
            "resulting_separation_km": desired_separation_km,
        }

    def plan_autonomous_avoidance(
        self,
        current_altitude_km: float,
        inclination_deg: float,
        initial_miss_distance_km: float,
        storm_kp: float = 8.0,
        storm_f10_7: float = 240.0,
        along_track_disp_km: float = 14.2,
        area_m2: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Generate complete autonomous avoidance maneuver plan comparing
        proactive early micro-burn vs reactive emergency evasion.
        """
        # Initial Risk evaluation (pre-maneuver under storm)
        pre_risk = self.adri_calc.evaluate_risk(
            miss_distance_km=initial_miss_distance_km,
            altitude_km=current_altitude_km,
            inclination_deg=inclination_deg,
            area_m2=area_m2,
            mass_kg=self.mass_kg,
            kp=storm_kp,
            f10_7=storm_f10_7,
            along_track_disp_km=along_track_disp_km,
            remaining_propellant_kg=self.fuel_kg,
        )

        # Option A: Autonomous Proactive Micro-Burn (AEGIS-LEO Early Warning)
        # Executed early (e.g., 2 orbits prior), small along-track phasing burn
        proactive_plan = self.compute_phasing_burn(
            current_altitude_km=current_altitude_km,
            desired_separation_km=14.0,
            orbits_to_encounter=2.5,
        )
        proactive_dv = proactive_plan["delta_v_m_s"]
        proactive_fuel = proactive_plan["fuel_consumed_kg"]

        # Post-proactive risk evaluation (miss distance increased to safe margin)
        post_proactive_risk = self.adri_calc.evaluate_risk(
            miss_distance_km=initial_miss_distance_km + 14.0,
            altitude_km=current_altitude_km,
            inclination_deg=inclination_deg,
            area_m2=area_m2,
            mass_kg=self.mass_kg - proactive_fuel,
            kp=storm_kp,
            f10_7=storm_f10_7,
            along_track_disp_km=1.0,  # compensated
            remaining_propellant_kg=self.fuel_kg - proactive_fuel,
        )

        # Option B: Reactive Emergency Evasion (Standard Ground-in-the-loop delay)
        # Executed late (< 0.5 orbit prior), requires aggressive radial/transverse burn
        reactive_dv = max(2.8, proactive_dv * 3.4)
        reactive_fuel = self.tsiolkovsky_fuel_cost(reactive_dv, self.mass_kg)
        post_reactive_risk = self.adri_calc.evaluate_risk(
            miss_distance_km=initial_miss_distance_km + 8.5,
            altitude_km=current_altitude_km,
            inclination_deg=inclination_deg,
            area_m2=area_m2,
            mass_kg=self.mass_kg - reactive_fuel,
            kp=storm_kp,
            f10_7=storm_f10_7,
            along_track_disp_km=4.0,
            remaining_propellant_kg=self.fuel_kg - reactive_fuel,
        )

        # Savings comparison
        propellant_saved_kg = reactive_fuel - proactive_fuel
        fuel_savings_pct = (propellant_saved_kg / max(reactive_fuel, 1e-6)) * 100.0

        return {
            "status": "AUTONOMOUS_BURN_PLANNED",
            "pre_burn_adri": pre_risk["a_dri"],
            "pre_burn_risk_level": pre_risk["risk_level"],
            "recommended_strategy": ManeuverType.STORM_COMPENSATION_MICRO_BURN,
            "proactive_burn": {
                "strategy": "Autonomous Early Phasing Micro-Burn",
                "delta_v_m_s": round(proactive_dv, 3),
                "propellant_consumed_kg": round(proactive_fuel, 4),
                "burn_duration_seconds": round(proactive_plan["burn_duration_seconds"], 1),
                "resulting_separation_km": round(initial_miss_distance_km + 14.0, 2),
                "post_burn_adri": round(post_proactive_risk["a_dri"], 4),
                "post_burn_risk_level": post_proactive_risk["risk_level"],
            },
            "reactive_baseline": {
                "strategy": "Late Reactive Ground-Commanded Burn",
                "delta_v_m_s": round(reactive_dv, 3),
                "propellant_consumed_kg": round(reactive_fuel, 4),
                "post_burn_adri": round(post_reactive_risk["a_dri"], 4),
                "post_burn_risk_level": post_reactive_risk["risk_level"],
            },
            "efficiency_metrics": {
                "propellant_saved_kg": round(propellant_saved_kg, 4),
                "fuel_savings_percentage": round(fuel_savings_pct, 1),
                "risk_reduction_percentage": round(
                    ((pre_risk["a_dri"] - post_proactive_risk["a_dri"]) / max(pre_risk["a_dri"], 1e-6)) * 100.0, 1
                ),
            },
        }
