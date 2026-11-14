"""
Constellation Fleet & Debris Cloud Conjunction Screening Engine
Performs batch multi-satellite all-on-all screening against cataloged debris populations.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from aegis_leo.constants import EARTH_RADIUS_KM, QUIET_KP, QUIET_F10_7
from aegis_leo.risk.adri import ADRICalculator, RiskLevel
from aegis_leo.physics.coordinates import kepler_to_cartesian
from aegis_leo.data.catalog import DEBRIS_CATALOG, SATELLITE_CATALOG


def generate_walker_delta_constellation(
    total_satellites: int = 24,
    planes: int = 4,
    altitude_km: float = 550.0,
    inclination_deg: float = 53.0,
    mass_kg: float = 260.0,
    area_m2: float = 2.2,
) -> List[Dict[str, Any]]:
    """
    Generate a Walker Delta constellation (T/P/F) of operational LEO satellites.
    """
    sats_per_plane = total_satellites // planes
    delta_raan = 360.0 / planes
    delta_anomaly = 360.0 / sats_per_plane

    fleet = []
    sat_idx = 1
    for p in range(planes):
        raan = p * delta_raan
        for s in range(sats_per_plane):
            nu = s * delta_anomaly + (p * (360.0 / total_satellites))
            nu = nu % 360.0

            fleet.append({
                "satellite_id": f"AEGIS-FLEET-{sat_idx:02d}",
                "name": f"AegisSat-{sat_idx:02d} (P{p+1})",
                "plane": p + 1,
                "altitude_km": altitude_km,
                "inclination_deg": inclination_deg,
                "raan_deg": round(raan, 2),
                "true_anomaly_deg": round(nu, 2),
                "mass_kg": mass_kg,
                "area_m2": area_m2,
                "remaining_fuel_kg": 15.0 - (sat_idx % 5) * 1.5,
            })
            sat_idx += 1

    return fleet


def generate_synthetic_debris_cloud(
    count: int = 150,
    center_alt_km: float = 550.0,
    alt_spread_km: float = 40.0,
    mean_inc_deg: float = 74.0,
) -> List[Dict[str, Any]]:
    """
    Generate realistic orbital debris fragments (Cosmos 2251 / Fengyun-1C type dispersion).
    """
    np.random.seed(42)  # Deterministic repeatability
    debris_list = []

    for i in range(1, count + 1):
        alt = np.random.normal(center_alt_km, alt_spread_km * 0.5)
        inc = np.random.normal(mean_inc_deg, 3.5)
        raan = np.random.uniform(0.0, 360.0)
        nu = np.random.uniform(0.0, 360.0)
        rcs = max(0.01, np.random.exponential(0.08))

        debris_list.append({
            "debris_id": f"DEB-FRAG-{i:03d}",
            "name": f"Fragment #{i:03d}",
            "altitude_km": round(float(alt), 2),
            "inclination_deg": round(float(inc), 2),
            "raan_deg": round(float(raan), 2),
            "true_anomaly_deg": round(float(nu), 2),
            "rcs_m2": round(float(rcs), 3),
        })

    return debris_list


class ConstellationConjunctionScreener:
    """
    Scalable constellation screening pipeline with spatial altitude filtering
    and real-time A-DRI risk scoring.
    """

    def __init__(self):
        self.adri_calc = ADRICalculator()

    def screen_constellation(
        self,
        fleet: List[Dict[str, Any]],
        debris_population: List[Dict[str, Any]],
        storm_kp: float = 8.0,
        storm_f10_7: float = 240.0,
        screening_threshold_km: float = 2.5,
    ) -> Dict[str, Any]:
        """
        Screen all constellation satellites against debris population.
        Identifies all conjunctions, evaluates A-DRI, and prioritizes critical threats.
        """
        conjunctions = []
        critical_count = 0
        high_count = 0
        elevated_count = 0
        nominal_count = 0

        # Along-track displacement under storm
        along_track_drift = (storm_kp - 2.0) * 2.1 if storm_kp >= 5.0 else 0.2

        for sat in fleet:
            sat_alt = sat["altitude_km"]
            sat_inc = sat["inclination_deg"]
            sat_raan = sat["raan_deg"]
            sat_nu = sat["true_anomaly_deg"]

            r_sat, v_sat = kepler_to_cartesian(
                a_km=EARTH_RADIUS_KM + sat_alt,
                ecc=0.0001,
                inc_deg=sat_inc,
                raan_deg=sat_raan,
                argp_deg=0.0,
                nu_deg=sat_nu,
            )

            # Spatial pre-filter: Altitude shell gate (+/- 15 km)
            candidate_debris = [
                d for d in debris_population if abs(d["altitude_km"] - sat_alt) <= 15.0
            ]

            sat_highest_adri = 0.0
            sat_worst_threat = None

            for deb in candidate_debris:
                r_deb, v_deb = kepler_to_cartesian(
                    a_km=EARTH_RADIUS_KM + deb["altitude_km"],
                    ecc=0.0001,
                    inc_deg=deb["inclination_deg"],
                    raan_deg=deb["raan_deg"],
                    argp_deg=0.0,
                    nu_deg=deb["true_anomaly_deg"],
                )

                miss_dist_km = float(np.linalg.norm(r_sat - r_deb))

                if miss_dist_km <= screening_threshold_km:
                    risk = self.adri_calc.evaluate_risk(
                        miss_distance_km=miss_dist_km,
                        altitude_km=sat_alt,
                        inclination_deg=sat_inc,
                        area_m2=sat["area_m2"],
                        mass_kg=sat["mass_kg"],
                        kp=storm_kp,
                        f10_7=storm_f10_7,
                        along_track_disp_km=along_track_drift,
                        remaining_propellant_kg=sat["remaining_fuel_kg"],
                    )

                    score = risk["a_dri"]
                    if score > sat_highest_adri:
                        sat_highest_adri = score
                        sat_worst_threat = {
                            "satellite_id": sat["satellite_id"],
                            "satellite_name": sat["name"],
                            "debris_id": deb["debris_id"],
                            "debris_name": deb["name"],
                            "miss_distance_km": round(miss_dist_km, 3),
                            "a_dri": round(score, 3),
                            "risk_level": risk["risk_level"],
                            "action_command": risk["action_command"],
                            "fuel_remaining_kg": sat["remaining_fuel_kg"],
                        }

            if sat_worst_threat:
                conjunctions.append(sat_worst_threat)
                if sat_worst_threat["risk_level"] == RiskLevel.CRITICAL.value:
                    critical_count += 1
                elif sat_worst_threat["risk_level"] == RiskLevel.HIGH.value:
                    high_count += 1
                else:
                    elevated_count += 1
            else:
                nominal_count += 1

        # Sort conjunctions from highest risk to lowest
        conjunctions.sort(key=lambda x: x["a_dri"], reverse=True)

        return {
            "total_fleet_size": len(fleet),
            "debris_population_size": len(debris_population),
            "conjunctions_detected": len(conjunctions),
            "summary": {
                "critical_satellites": critical_count,
                "high_risk_satellites": high_count,
                "elevated_satellites": elevated_count,
                "nominal_satellites": nominal_count,
            },
            "highest_risk_conjunctions": conjunctions[:10],
            "storm_kp": storm_kp,
            "screening_threshold_km": screening_threshold_km,
        }
