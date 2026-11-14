"""
AEGIS-LEO Physics Engine
========================
Astrodynamics, atmospheric drag models, coordinate transformations,
solar radiation pressure, and third-body gravitational perturbations.
"""

from aegis_leo.physics.coordinates import (
    kepler_to_cartesian,
    cartesian_to_kepler,
    eci_to_geodetic,
    relative_atmospheric_velocity,
)
from aegis_leo.physics.atmosphere import (
    compute_atmospheric_density,
    compute_drag_acceleration,
    compute_ballistic_coefficient,
)
from aegis_leo.physics.propagator import (
    OrbitalPropagator,
    compute_gravitational_and_j2_acceleration,
    rk4_step,
)
from aegis_leo.physics.perturbations import (
    compute_srp_acceleration,
    compute_third_body_acceleration,
    compute_solar_third_body_acceleration,
    compute_lunar_third_body_acceleration,
    compute_all_perturbations,
    total_perturbation_acceleration,
    compute_cylindrical_shadow,
    compute_dual_cone_shadow,
    compute_shadow_factor,
    sun_position_eci,
    moon_position_eci,
    datetime_to_jd,
    mjd_to_jd,
    jd_to_mjd,
    rk4_step_high_precision,
    HighPrecisionPerturbationEngine,
)

__all__ = [
    "kepler_to_cartesian",
    "cartesian_to_kepler",
    "eci_to_geodetic",
    "relative_atmospheric_velocity",
    "compute_atmospheric_density",
    "compute_drag_acceleration",
    "compute_ballistic_coefficient",
    "OrbitalPropagator",
    "compute_gravitational_and_j2_acceleration",
    "rk4_step",
    "compute_srp_acceleration",
    "compute_third_body_acceleration",
    "compute_solar_third_body_acceleration",
    "compute_lunar_third_body_acceleration",
    "compute_all_perturbations",
    "total_perturbation_acceleration",
    "compute_cylindrical_shadow",
    "compute_dual_cone_shadow",
    "compute_shadow_factor",
    "sun_position_eci",
    "moon_position_eci",
    "datetime_to_jd",
    "mjd_to_jd",
    "jd_to_mjd",
    "rk4_step_high_precision",
    "HighPrecisionPerturbationEngine",
]
