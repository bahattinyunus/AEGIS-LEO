"""
Monte Carlo Orbital Conjunction & Covariance Dispersion Simulator
Performs empirical sampling of 3D positional uncertainty ellipsoids to validate analytical Pc.
"""

from typing import Dict, Any, Tuple
import numpy as np
from aegis_leo.risk.collision import compute_collision_probability


class MonteCarloConjunctionSimulator:
    """
    Simulates thousands of conjunction passes with multivariate Gaussian covariance dispersion.
    """

    @staticmethod
    def run_simulation(
        nominal_miss_km: float = 0.45,
        combined_radius_m: float = 10.0,
        sigma_radial_km: float = 0.15,
        sigma_intrack_km: float = 0.60,
        sigma_crosstrack_km: float = 0.20,
        num_trials: int = 2000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo dispersion trials around nominal encounter geometry.
        """
        np.random.seed(seed)
        r_comb_km = combined_radius_m / 1000.0

        # Nominal relative position in RTN frame
        nominal_pos = np.array([0.0, nominal_miss_km, 0.0])

        # Combined 3D covariance matrix (diagonal approximation in RTN)
        cov_diag = np.array([sigma_radial_km**2, sigma_intrack_km**2, sigma_crosstrack_km**2])

        # Sample perturbations
        pert_r = np.random.normal(0.0, sigma_radial_km, num_trials)
        pert_t = np.random.normal(0.0, sigma_intrack_km, num_trials)
        pert_n = np.random.normal(0.0, sigma_crosstrack_km, num_trials)

        displaced_r = nominal_pos[0] + pert_r
        displaced_t = nominal_pos[1] + pert_t
        displaced_n = nominal_pos[2] + pert_n

        distances_km = np.sqrt(displaced_r**2 + displaced_t**2 + displaced_n**2)
        collisions = np.sum(distances_km <= r_comb_km)

        empirical_pc = float(collisions / num_trials)
        mean_miss_km = float(np.mean(distances_km))
        min_miss_km = float(np.min(distances_km))

        # Analytical comparison
        effective_sigma_2d = float(np.sqrt(sigma_radial_km * sigma_crosstrack_km))
        analytical_pc = compute_collision_probability(
            miss_distance_km=nominal_miss_km,
            combined_radius_m=combined_radius_m,
            sigma_pos_km=effective_sigma_2d,
        )

        return {
            "num_trials": num_trials,
            "collisions_detected": int(collisions),
            "empirical_pc": empirical_pc,
            "analytical_pc": analytical_pc,
            "nominal_miss_km": nominal_miss_km,
            "mean_simulated_miss_km": round(mean_miss_km, 4),
            "min_simulated_miss_km": round(min_miss_km, 4),
            "dispersion_sigmas": {
                "radial_km": sigma_radial_km,
                "intrack_km": sigma_intrack_km,
                "crosstrack_km": sigma_crosstrack_km,
            },
        }
