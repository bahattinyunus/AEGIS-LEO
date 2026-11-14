"""
Multi-Objective Pareto Frontier Maneuver Optimizer
Evaluates multi-attribute trade-offs: Delta-v cost vs Conjunction Clearance
vs Secondary Conjunction Risk vs Mission Downtime.
"""

from typing import List, Dict, Any
import numpy as np
from aegis_leo.constants import STANDARD_GRAVITY, EARTH_MU, EARTH_RADIUS_KM
from aegis_leo.risk.adri import ADRICalculator


class ParetoManeuverOptimizer:
    """
    Computes a Pareto-optimal frontier of candidate collision avoidance burns.
    """

    def __init__(
        self,
        mass_kg: float = 260.0,
        fuel_remaining_kg: float = 15.0,
        isp_s: float = 250.0,
    ):
        self.mass_kg = float(mass_kg)
        self.fuel_kg = float(fuel_remaining_kg)
        self.isp_s = float(isp_s)
        self.adri_calc = ADRICalculator()

    def generate_pareto_frontier(
        self,
        current_altitude_km: float,
        inclination_deg: float,
        initial_miss_km: float = 0.45,
        storm_kp: float = 8.0,
        storm_f10_7: float = 240.0,
        num_candidates: int = 12,
    ) -> Dict[str, Any]:
        """
        Generate candidate burn profiles across the Pareto front.
        """
        r1 = EARTH_RADIUS_KM + current_altitude_km
        v_circ = np.sqrt(EARTH_MU / r1)  # km/s

        candidates = []

        # Generate along-track phasing burns (Delta-v from 0.15 m/s to 3.0 m/s)
        dv_spectrum_m_s = np.linspace(0.15, 2.80, num_candidates)

        for i, dv in enumerate(dv_spectrum_m_s):
            # Tsiolkovsky propellant mass
            ve = self.isp_s * STANDARD_GRAVITY
            fuel_cost = self.mass_kg * (1.0 - np.exp(-dv / ve))

            # Resulting separation: Vis-Viva phasing displacement
            # separation approx proportional to dv * orbits
            orbits_to_tca = 2.5
            dv_km_s = dv / 1000.0
            disp_km = initial_miss_km + (dv_km_s / v_circ) * (3.0 * np.pi * orbits_to_tca * r1)

            # Secondary conjunction risk factor (higher burns increase chance of entering another orbit shell)
            secondary_risk_penalty = float(np.clip((dv - 1.2) * 0.08 if dv > 1.2 else 0.01, 0.01, 0.35))

            # Payload downtime (hours): larger burns take longer to settle & calibrate sensors
            downtime_hours = float(round(1.5 + (dv / 0.5) * 0.8, 1))

            # Post-burn A-DRI
            risk_eval = self.adri_calc.evaluate_risk(
                miss_distance_km=disp_km,
                altitude_km=current_altitude_km,
                inclination_deg=inclination_deg,
                kp=storm_kp,
                f10_7=storm_f10_7,
                along_track_disp_km=1.0,
                required_dv_m_s=dv,
                remaining_propellant_kg=self.fuel_kg - fuel_cost,
            )

            candidates.append({
                "candidate_id": f"BURN-OPT-{i+1:02d}",
                "delta_v_m_s": round(float(dv), 3),
                "propellant_kg": round(float(fuel_cost), 4),
                "resulting_separation_km": round(float(disp_km), 2),
                "post_burn_adri": round(risk_eval["a_dri"], 4),
                "risk_level": risk_eval["risk_level"],
                "secondary_conjunction_risk": round(secondary_risk_penalty, 3),
                "mission_downtime_hours": downtime_hours,
            })

        # Identify key trade-off solutions
        min_fuel = min(candidates, key=lambda c: c["delta_v_m_s"])
        max_clearance = max(candidates, key=lambda c: c["resulting_separation_km"])

        # Balanced solution: minimizes composite cost J = 0.5*(dv / max_dv) + 0.5*(1 - clearance / max_clearance)
        max_dv = max(c["delta_v_m_s"] for c in candidates)
        max_disp = max(c["resulting_separation_km"] for c in candidates)
        balanced = min(
            candidates,
            key=lambda c: 0.5 * (c["delta_v_m_s"] / max_dv) + 0.5 * (1.0 - c["resulting_separation_km"] / max_disp) + c["secondary_conjunction_risk"]
        )

        return {
            "satellite_mass_kg": self.mass_kg,
            "fuel_budget_kg": self.fuel_kg,
            "pareto_candidates": candidates,
            "optimal_recommendations": {
                "minimum_propellant_solution": min_fuel,
                "balanced_optimal_solution": balanced,
                "maximum_clearance_solution": max_clearance,
            },
        }
