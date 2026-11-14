"""
AEGIS-LEO: Astrodynamics & Relative Motion Module
Provides Hill-Clohessy-Wiltshire (HCW) and Yamanaka-Ankersen (YA) analytical relative
orbit propagation, V-bar / R-bar approach corridor safety, and multi-impulse rendezvous planning.
"""

from aegis_leo.astrodynamics.cw_proximity import (
    ClohessyWiltshirePropagator,
    YamanakaAnkersenPropagator,
    ApproachSafetyValidator,
    ApproachSafetyReport,
    RendezvousPlan,
    CorridorViolation,
    plan_2_impulse_rendezvous,
    plan_4_impulse_rendezvous,
    compute_hcw_stm,
    compute_yamanaka_ankersen_stm,
    propagate_hcw,
    propagate_yamanaka_ankersen,
    check_approach_corridor,
    validate_approach_trajectory,
    compute_closed_orbit_velocity,
    compute_relative_orbit_geometry,
    mean_to_eccentric_anomaly,
    eccentric_to_true_anomaly,
    true_to_eccentric_anomaly,
    true_to_mean_anomaly,
)

__all__ = [
    "ClohessyWiltshirePropagator",
    "YamanakaAnkersenPropagator",
    "ApproachSafetyValidator",
    "ApproachSafetyReport",
    "RendezvousPlan",
    "CorridorViolation",
    "plan_2_impulse_rendezvous",
    "plan_4_impulse_rendezvous",
    "compute_hcw_stm",
    "compute_yamanaka_ankersen_stm",
    "propagate_hcw",
    "propagate_yamanaka_ankersen",
    "check_approach_corridor",
    "validate_approach_trajectory",
    "compute_closed_orbit_velocity",
    "compute_relative_orbit_geometry",
    "mean_to_eccentric_anomaly",
    "eccentric_to_true_anomaly",
    "true_to_eccentric_anomaly",
    "true_to_mean_anomaly",
]
