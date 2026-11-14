"""
AEGIS-LEO Active Sensor Emulation: Spaceborne Radar & Active LIDAR.

This module provides high-fidelity orbital sensor simulation including:
- Spaceborne Radar Range Equation & thermal SNR models
- Active LIDAR Range Equation & photon-counting shot-noise SNR models
- Swerling Target Fluctuation Models (Swerling 0 / V, I, II, III, IV)
- Range-Rate Doppler shifts, relativistic corrections, and ambiguity analysis
- Detection thresholding: Marcum Q, exact analytical Pd, CA-CFAR (1D/2D) and OS-CFAR
- Cramer-Rao Lower Bound (CRLB) measurement uncertainty modeling
- Spaceborne Radar & LIDAR mission observation engines
"""

from aegis_leo.sensors.laser_radar import (
    # Constants & Enums
    SPEED_OF_LIGHT,
    BOLTZMANN_CONSTANT,
    PLANCK_CONSTANT,
    STANDARD_TEMP_K,
    SwerlingModel,
    RadarBand,
    LidarWavelength,
    # Math & Detection Probabilities
    bessel_i0_scaled,
    marcum_q1,
    solve_gamma_threshold_pfa,
    swerling_pdf,
    sample_swerling_rcs,
    calculate_pd,
    # Radar Models
    radar_range_equation,
    compute_radar_snr,
    compute_max_radar_range,
    # LIDAR Models
    lidar_range_equation,
    compute_lidar_snr,
    compute_max_lidar_range,
    # Doppler & Ambiguities & CRLB
    compute_doppler_shift,
    compute_range_rate_from_doppler,
    compute_radar_ambiguities,
    compute_crlb_uncertainty,
    # CFAR & Thresholding
    detection_threshold_pfa,
    ca_cfar_1d,
    ca_cfar_2d,
    os_cfar_1d,
    # Sensor Configurations & Classes
    SpaceborneRadarConfig,
    RadarObservation,
    SpaceborneRadar,
    SpaceborneLidarConfig,
    LidarObservation,
    SpaceborneLidar,
    generate_range_doppler_map,
)

__all__ = [
    "SPEED_OF_LIGHT",
    "BOLTZMANN_CONSTANT",
    "PLANCK_CONSTANT",
    "STANDARD_TEMP_K",
    "SwerlingModel",
    "RadarBand",
    "LidarWavelength",
    "bessel_i0_scaled",
    "marcum_q1",
    "solve_gamma_threshold_pfa",
    "swerling_pdf",
    "sample_swerling_rcs",
    "calculate_pd",
    "radar_range_equation",
    "compute_radar_snr",
    "compute_max_radar_range",
    "lidar_range_equation",
    "compute_lidar_snr",
    "compute_max_lidar_range",
    "compute_doppler_shift",
    "compute_range_rate_from_doppler",
    "compute_radar_ambiguities",
    "compute_crlb_uncertainty",
    "detection_threshold_pfa",
    "ca_cfar_1d",
    "ca_cfar_2d",
    "os_cfar_1d",
    "SpaceborneRadarConfig",
    "RadarObservation",
    "SpaceborneRadar",
    "SpaceborneLidarConfig",
    "LidarObservation",
    "SpaceborneLidar",
    "generate_range_doppler_map",
]
