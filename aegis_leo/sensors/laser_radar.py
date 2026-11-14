"""
AEGIS-LEO: Active Spaceborne Radar and LIDAR Sensor Emulation Engine.

This module provides high-fidelity, physics-based modeling and emulation of active
spaceborne sensors (Radar and LIDAR) for proximity operations, conjunction screening,
and orbital debris tracking in LEO regimes:
1. Spaceborne Radar Range Equation & thermal noise SNR models.
2. Active LIDAR Range Equation & photon-counting shot-noise SNR models.
3. Radar Cross Section (RCS) Swerling target fluctuation models (Swerling 0/V, I, II, III, IV).
4. Range-rate Doppler frequency shifts (classical and relativistic) & radar ambiguity analysis.
5. Detection thresholding: Marcum Q-function, analytical detection probability (Pd),
   1D/2D Cell-Averaging CFAR (CA-CFAR), and Order-Statistic CFAR (OS-CFAR).
6. Cramer-Rao Lower Bound (CRLB) measurement uncertainty for range and range-rate.
7. SpaceborneRadar and SpaceborneLidar orbital observation classes with synthetic
   Range-Doppler map generation.

Pure Python and NumPy implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from aegis_leo.constants import SPEED_OF_LIGHT

# Physical & Fundamental Constants
BOLTZMANN_CONSTANT: float = 1.380649e-23   # J / K
PLANCK_CONSTANT: float = 6.62607015e-34    # J * s
STANDARD_TEMP_K: float = 290.0            # IEEE Standard Reference Temperature (K)


class SwerlingModel(Enum):
    """Radar Target Fluctuation Models according to Peter Swerling (1954)."""
    SWERLING_0 = 0  # Marcum non-fluctuating target (constant RCS)
    SWERLING_I = 1  # Slow fading (scan-to-scan), many independent Rayleigh scatterers (Chi-square 2 DOF)
    SWERLING_II = 2  # Fast fading (pulse-to-pulse), many independent Rayleigh scatterers (Chi-square 2 DOF)
    SWERLING_III = 3  # Slow fading (scan-to-scan), one dominant scatterer + Rayleigh (Chi-square 4 DOF)
    SWERLING_IV = 4  # Fast fading (pulse-to-pulse), one dominant scatterer + Rayleigh (Chi-square 4 DOF)


class RadarBand(Enum):
    """Standard IEEE Radar Frequency Bands (Nominal carrier frequencies in Hz)."""
    L_BAND = 1.5e9    # 1.5 GHz
    S_BAND = 3.0e9    # 3.0 GHz
    C_BAND = 5.5e9    # 5.5 GHz
    X_BAND = 10.0e9   # 10.0 GHz (standard for spaceborne tracking & imaging)
    KU_BAND = 15.0e9  # 15.0 GHz
    KA_BAND = 35.0e9  # 35.0 GHz (high resolution rendezvous / debris radar)
    W_BAND = 94.0e9   # 94.0 GHz (cloud & millimeter-wave micro-debris radar)


class LidarWavelength(Enum):
    """Common Spaceborne Active LIDAR Optical Wavelengths (meters)."""
    GREEN_532 = 532e-9    # Frequency-doubled Nd:YAG (532 nm)
    NIR_1064 = 1064e-9    # Nd:YAG fundamental (1064 nm)
    SWIR_1550 = 1550e-9   # Eye-safe Erbium fiber (1550 nm)


# ============================================================================
# SECTION 1: Mathematical Helper Functions (Bessel, Marcum-Q, Threshold Solvers)
# ============================================================================

def bessel_i0_scaled(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute exp(-z) * I_0(z) with high numerical precision across all z >= 0.
    Uses Abramowitz & Stegun polynomial and rational approximations (9.8.1 and 9.8.2).
    Accuracy is better than 2e-7 across the entire positive domain without overflow.
    """
    z_arr = np.asarray(np.maximum(0.0, z), dtype=float)
    res = np.zeros_like(z_arr)

    # Branch 1: z < 3.75
    mask = z_arr < 3.75
    if np.any(mask):
        t = z_arr[mask] / 3.75
        t2 = t * t
        i0 = 1.0 + t2 * (
            3.5156229 + t2 * (
                3.0899424 + t2 * (
                    1.2067492 + t2 * (
                        0.2659732 + t2 * (
                            0.0360768 + t2 * 0.0045813
                        )
                    )
                )
            )
        )
        res[mask] = np.exp(-z_arr[mask]) * i0

    # Branch 2: z >= 3.75
    mask_hi = ~mask
    if np.any(mask_hi):
        z_hi = z_arr[mask_hi]
        t_hi = 3.75 / z_hi
        poly = (
            0.39894228 + t_hi * (
                0.01328592 + t_hi * (
                    0.00225319 + t_hi * (
                        -0.00157565 + t_hi * (
                            0.00916281 + t_hi * (
                                -0.02057706 + t_hi * (
                                    0.02635537 + t_hi * (
                                        -0.01647633 + t_hi * 0.00392377
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        res[mask_hi] = (1.0 / np.sqrt(z_hi)) * poly

    if np.isscalar(z):
        return float(res.item())
    return res


def marcum_q1(a: float, b: float, n_pts: int = 64) -> float:
    """
    First-order Marcum Q-function Q_1(a, b).
    Q_1(a, b) = int_b^infty x * exp(-(x^2 + a^2)/2) * I_0(a * x) dx.

    Implemented using Gauss-Legendre quadrature on the exponentially factored integrand:
    x * exp(-(x - a)^2 / 2) * [exp(-a*x) * I_0(a*x)],
    completely avoiding numerical overflow/underflow, with analytical asymptotic bounds.
    """
    if b <= 0.0:
        return 1.0
    if a <= 0.0:
        return float(np.exp(-0.5 * b * b))

    # Asymptotic regimes:
    # Under H1, the envelope variable has mean ~ a and variance ~ 1.
    # If a is much larger than b, Pd -> 1; if b is much larger than a, Pd -> 0.
    if a - b > 8.0:
        return 1.0
    if b - a > 8.0:
        return 0.0

    lower = max(b, a - 8.0)
    upper = a + 8.0
    if upper <= lower:
        return 0.0

    xi, wi = np.polynomial.legendre.leggauss(n_pts)
    x = 0.5 * (upper - lower) * xi + 0.5 * (upper + lower)
    w = 0.5 * (upper - lower) * wi

    integrand = x * np.exp(-0.5 * (x - a) ** 2) * bessel_i0_scaled(a * x)
    val = float(np.sum(w * integrand))
    return float(np.clip(val, 0.0, 1.0))


def _log_poisson_tail(k_max: int, lam: float) -> float:
    """
    Compute log of sum_{k=0}^{k_max-1} (lam^k / k!) * exp(-lam)
    stably in log space for arbitrary k_max.
    """
    if lam <= 0.0:
        return 0.0
    terms = [-lam + k * math.log(lam) - math.lgamma(k + 1) for k in range(k_max)]
    max_t = max(terms)
    sum_exp = sum(math.exp(t - max_t) for t in terms)
    return max_t + math.log(max(1e-300, sum_exp))


def solve_gamma_threshold_pfa(pfa: float, num_pulses: int = 1) -> float:
    """
    Solve for normalized threshold V_T given false alarm probability Pfa
    and non-coherently integrated pulse count N under Chi-square 2N noise.
    For N=1: V_T = -ln(Pfa).
    For N>1: uses binary bracketed search on the log Poisson tail CDF.
    """
    if pfa <= 0.0 or pfa >= 1.0:
        raise ValueError(f"pfa must be in (0, 1), got {pfa}")
    if num_pulses <= 0:
        raise ValueError(f"num_pulses must be >= 1, got {num_pulses}")

    if num_pulses == 1:
        return -math.log(pfa)

    target_log_pfa = math.log(pfa)
    low = float(num_pulses)
    high = float(num_pulses + 35.0 * math.sqrt(num_pulses) + 60.0)

    for _ in range(60):
        mid = 0.5 * (low + high)
        val = _log_poisson_tail(num_pulses, mid)
        if val > target_log_pfa:
            low = mid
        else:
            high = mid

    return 0.5 * (low + high)


# ============================================================================
# SECTION 2: Spaceborne Radar Range Equation & SNR Models
# ============================================================================

def radar_range_equation(
    peak_power_w: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    wavelength_m: float,
    rcs_m2: Union[float, np.ndarray],
    range_m: Union[float, np.ndarray],
    system_loss_db: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Calculate received peak signal power P_r (Watts) via the Monostatic Radar Range Equation:
    P_r = (P_t * G_t * G_r * lambda^2 * sigma) / ((4 * pi)^3 * R^4 * L)

    Parameters:
        peak_power_w: Transmitter peak pulse power (Watts).
        tx_gain_dbi: Transmit antenna gain in dBi.
        rx_gain_dbi: Receive antenna gain in dBi.
        wavelength_m: Radar carrier wavelength (meters).
        rcs_m2: Radar cross section of target (m^2).
        range_m: Slant range to target (meters).
        system_loss_db: Total two-way radar system losses in dB (>= 0).

    Returns:
        Received power P_r in Watts (scalar or array matching range_m / rcs_m2).
    """
    if peak_power_w <= 0.0:
        raise ValueError("peak_power_w must be positive.")
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive.")
    if system_loss_db < 0.0:
        raise ValueError("system_loss_db must be non-negative.")

    gt_lin = 10.0 ** (tx_gain_dbi / 10.0)
    gr_lin = 10.0 ** (rx_gain_dbi / 10.0)
    loss_lin = 10.0 ** (system_loss_db / 10.0)

    range_arr = np.asarray(range_m, dtype=float)
    if np.any(range_arr <= 0.0):
        raise ValueError("range_m must be strictly positive.")

    rcs_arr = np.asarray(rcs_m2, dtype=float)
    if np.any(rcs_arr < 0.0):
        raise ValueError("rcs_m2 must be non-negative.")

    numerator = peak_power_w * gt_lin * gr_lin * (wavelength_m ** 2) * rcs_arr
    denominator = ((4.0 * math.pi) ** 3) * (range_arr ** 4) * loss_lin
    pr = numerator / denominator

    if np.isscalar(range_m) and np.isscalar(rcs_m2):
        return float(pr.item())
    return pr


def compute_radar_snr(
    peak_power_w: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    wavelength_m: Optional[float] = None,
    rcs_m2: Union[float, np.ndarray] = 1.0,
    range_m: Union[float, np.ndarray] = 10000.0,
    noise_temp_k: float = STANDARD_TEMP_K,
    noise_figure_db: float = 3.0,
    bandwidth_hz: float = 1e6,
    system_loss_db: float = 0.0,
    num_pulses: int = 1,
    coherent: bool = True,
    frequency_hz: Optional[float] = None,
    pulse_width_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate single-pulse and integrated Signal-to-Noise Ratio (SNR) for Spaceborne Radar.

    Parameters:
        peak_power_w: Transmitter peak pulse power (Watts).
        tx_gain_dbi: Transmit antenna gain (dBi).
        rx_gain_dbi: Receive antenna gain (dBi).
        wavelength_m: Carrier wavelength (meters).
        rcs_m2: Target radar cross section (m^2).
        range_m: Slant range (meters).
        noise_temp_k: Reference system noise temperature (Kelvin, default 290 K).
        noise_figure_db: Receiver noise figure in dB (default 3.0 dB).
        bandwidth_hz: Receiver matched filter noise bandwidth in Hz.
        system_loss_db: System loss in dB.
        num_pulses: Number of integrated radar pulses (CPI burst).
        coherent: If True, uses coherent integration (N * SNR);
                  If False, uses non-coherent (square-law / Peebles) integration.

    Returns:
        Dictionary with SNR metrics:
            received_power_w, noise_power_w, snr_linear, snr_db,
            integrated_snr_linear, integrated_snr_db, noise_temp_sys_k.
    """
    if wavelength_m is None:
        if frequency_hz is not None:
            wavelength_m = SPEED_OF_LIGHT / frequency_hz
        else:
            wavelength_m = SPEED_OF_LIGHT / 10.0e9
    pr = radar_range_equation(
        peak_power_w=peak_power_w,
        tx_gain_dbi=tx_gain_dbi,
        rx_gain_dbi=rx_gain_dbi,
        wavelength_m=wavelength_m,
        rcs_m2=rcs_m2,
        range_m=range_m,
        system_loss_db=system_loss_db,
    )

    f_lin = 10.0 ** (noise_figure_db / 10.0)
    t_sys = noise_temp_k * f_lin
    pn = BOLTZMANN_CONSTANT * t_sys * bandwidth_hz

    snr_lin = pr / pn
    snr_db = 10.0 * np.log10(np.maximum(1e-20, snr_lin))

    if num_pulses <= 1:
        snr_int_lin = snr_lin
    else:
        if coherent:
            snr_int_lin = num_pulses * snr_lin
        else:
            # Peebles / empirical non-coherent integration formula:
            # SNR_int = (N * SNR^2) / (1 + SNR)
            snr_int_lin = (num_pulses * (snr_lin ** 2)) / (1.0 + snr_lin)

    snr_int_db = 10.0 * np.log10(np.maximum(1e-20, snr_int_lin))

    return {
        "received_power_w": pr,
        "noise_power_w": pn,
        "snr_linear": snr_lin,
        "snr_db": snr_db,
        "integrated_snr_linear": snr_int_lin,
        "integrated_snr_db": snr_int_db,
        "noise_temp_sys_k": t_sys,
    }


def compute_max_radar_range(
    peak_power_w: float,
    tx_gain_dbi: float,
    rx_gain_dbi: float,
    wavelength_m: float,
    rcs_m2: float,
    min_snr_db: float,
    noise_temp_k: float = STANDARD_TEMP_K,
    noise_figure_db: float = 3.0,
    bandwidth_hz: float = 1e6,
    system_loss_db: float = 0.0,
) -> float:
    """
    Invert the radar range equation to compute maximum detection range R_max (meters)
    for a given minimum detectable SNR threshold (min_snr_db).
    """
    min_snr_lin = 10.0 ** (min_snr_db / 10.0)
    f_lin = 10.0 ** (noise_figure_db / 10.0)
    t_sys = noise_temp_k * f_lin
    pn = BOLTZMANN_CONSTANT * t_sys * bandwidth_hz

    gt_lin = 10.0 ** (tx_gain_dbi / 10.0)
    gr_lin = 10.0 ** (rx_gain_dbi / 10.0)
    loss_lin = 10.0 ** (system_loss_db / 10.0)

    num = peak_power_w * gt_lin * gr_lin * (wavelength_m ** 2) * rcs_m2
    denom = ((4.0 * math.pi) ** 3) * loss_lin * pn * min_snr_lin
    r_max = (num / denom) ** 0.25
    return float(r_max)


# ============================================================================
# SECTION 3: Active LIDAR Range Equation & Shot-Noise SNR Models
# ============================================================================

def lidar_range_equation(
    pulse_energy_j: float,
    pulse_width_s: float,
    optics_tx_eff: float,
    optics_rx_eff: float,
    aperture_diam_m: float,
    beam_divergence_rad: float,
    target_optical_cross_section_m2: Union[float, np.ndarray],
    range_m: Union[float, np.ndarray],
    target_type: str = "unresolved",
    diffuse_area_m2: Optional[float] = None,
    diffuse_reflectivity: Optional[float] = None,
    incidence_angle_rad: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Calculate received optical peak power P_rx (Watts) for spaceborne active LIDAR.

    Parameters:
        pulse_energy_j: Laser optical pulse energy (Joules).
        pulse_width_s: Full-width half-max pulse duration (seconds).
        optics_tx_eff: Transmitter optical throughput efficiency (0 to 1).
        optics_rx_eff: Receiver optical telescope transmission efficiency (0 to 1).
        aperture_diam_m: Receiver telescope clear aperture diameter (meters).
        beam_divergence_rad: Laser full-angle beam divergence (radians).
        target_optical_cross_section_m2: Target optical cross section sigma_opt (m^2).
        range_m: Range from LIDAR to target (meters).
        target_type: "unresolved" (debris smaller than beam footprint) or
                     "resolved_diffuse" (Lambertian plate intercepting beam).
        diffuse_area_m2: Area of target (m^2), used if target_type == "resolved_diffuse".
        diffuse_reflectivity: Surface reflectivity rho (0 to 1).
        incidence_angle_rad: Surface normal angle relative to line of sight (radians).

    Returns:
        Received peak power in Watts.
    """
    if pulse_energy_j <= 0.0 or pulse_width_s <= 0.0:
        raise ValueError("pulse_energy_j and pulse_width_s must be positive.")
    if aperture_diam_m <= 0.0 or beam_divergence_rad <= 0.0:
        raise ValueError("Aperture diameter and beam divergence must be positive.")

    peak_tx_power = pulse_energy_j / pulse_width_s
    aperture_area = 0.25 * math.pi * (aperture_diam_m ** 2)

    range_arr = np.asarray(range_m, dtype=float)
    if np.any(range_arr <= 0.0):
        raise ValueError("range_m must be strictly positive.")

    eta_opt = optics_tx_eff * optics_rx_eff

    if target_type == "resolved_diffuse":
        rho = diffuse_reflectivity if diffuse_reflectivity is not None else 0.2
        cos_theta = max(0.0, math.cos(incidence_angle_rad))
        # For a Lambertian resolved target:
        # P_rx = P_tx * eta_opt * (A_rx / (pi * R^2)) * rho * cos(theta)
        pr = peak_tx_power * eta_opt * (aperture_area / (math.pi * (range_arr ** 2))) * rho * cos_theta
    else:
        # Unresolved target with optical cross section sigma_opt:
        # Beam irradiance at target: I_t = (4 * P_tx * eta_tx) / (pi * R^2 * theta_div^2)
        # Reflected intensity: J = I_t * sigma_opt / (4 * pi)
        # Irradiance at telescope: I_rx = J / R^2
        # Power collected: P_rx = I_rx * A_rx * eta_rx
        # => P_rx = (P_tx * eta_tx * eta_rx * A_rx * sigma_opt) / (pi^2 * R^4 * theta_div^2)
        sigma_arr = np.asarray(target_optical_cross_section_m2, dtype=float)
        numerator = peak_tx_power * eta_opt * aperture_area * sigma_arr
        denominator = (math.pi ** 2) * (range_arr ** 4) * (beam_divergence_rad ** 2)
        pr = numerator / denominator

    if np.isscalar(range_m) and np.isscalar(target_optical_cross_section_m2):
        return float(pr.item())
    return pr


def compute_lidar_snr(
    pulse_energy_j: float,
    pulse_width_s: float,
    wavelength_m: float,
    optics_tx_eff: float,
    optics_rx_eff: float,
    aperture_diam_m: float,
    beam_divergence_rad: float,
    target_optical_cross_section_m2: Union[float, np.ndarray],
    range_m: Union[float, np.ndarray],
    quantum_efficiency: float = 0.40,
    dark_count_rate_hz: float = 500.0,
    background_power_w: float = 1e-12,
    excess_noise_factor: float = 2.0,
    num_pulses: int = 1,
) -> Dict[str, Any]:
    """
    Calculate active LIDAR photon counts and shot-noise-limited SNR.

    Parameters:
        pulse_energy_j: Optical pulse energy (Joules).
        pulse_width_s: Pulse duration (seconds).
        wavelength_m: Laser optical wavelength (meters).
        optics_tx_eff: Transmitter optical transmission.
        optics_rx_eff: Receiver optical transmission.
        aperture_diam_m: Telescope aperture diameter (meters).
        beam_divergence_rad: Beam full-angle divergence (radians).
        target_optical_cross_section_m2: Target optical cross section (m^2).
        range_m: Target range (meters).
        quantum_efficiency: Photodetector (APD / SiPM) quantum efficiency eta_q (0 to 1).
        dark_count_rate_hz: Detector dark count rate (counts/second).
        background_power_w: Solar/Earth albedo background optical power (Watts).
        excess_noise_factor: APD excess noise factor F (>= 1.0; 1.0 for ideal PIN/PMT).
        num_pulses: Number of laser pulses integrated.

    Returns:
        Dictionary with:
            received_peak_power_w: Optical peak power received (Watts).
            signal_photoelectrons: Mean signal photoelectrons per pulse N_s.
            background_photoelectrons: Background photoelectrons per pulse N_b.
            dark_count_photoelectrons: Dark count electrons per pulse N_d.
            noise_variance: Total noise electron variance sigma_n^2.
            single_pulse_snr_linear: Photoelectron SNR (N_s / sigma_n).
            single_pulse_snr_db: 10 * log10(SNR_power).
            integrated_snr_linear: Integrated SNR across N pulses.
            integrated_snr_db: Integrated SNR in dB.
    """
    if wavelength_m <= 0.0:
        raise ValueError("wavelength_m must be positive.")
    if quantum_efficiency <= 0.0 or quantum_efficiency > 1.0:
        raise ValueError("quantum_efficiency must be in (0, 1].")

    # Photon energy h * c / lambda
    photon_energy_j = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / wavelength_m

    p_rx = lidar_range_equation(
        pulse_energy_j=pulse_energy_j,
        pulse_width_s=pulse_width_s,
        optics_tx_eff=optics_tx_eff,
        optics_rx_eff=optics_rx_eff,
        aperture_diam_m=aperture_diam_m,
        beam_divergence_rad=beam_divergence_rad,
        target_optical_cross_section_m2=target_optical_cross_section_m2,
        range_m=range_m,
    )

    # Received pulse energy
    e_rx = p_rx * pulse_width_s

    # Photoelectron counts per pulse
    n_s = (quantum_efficiency * e_rx) / photon_energy_j
    n_b = (quantum_efficiency * background_power_w * pulse_width_s) / photon_energy_j
    n_d = dark_count_rate_hz * pulse_width_s

    # Total noise electron variance: APD shot noise + dark counts
    noise_var = excess_noise_factor * (n_s + n_b) + n_d
    noise_sigma = np.sqrt(np.maximum(1e-12, noise_var))

    snr_amplitude = n_s / noise_sigma
    snr_power_lin = snr_amplitude ** 2
    snr_db = 10.0 * np.log10(np.maximum(1e-20, snr_power_lin))

    if num_pulses <= 1:
        snr_int_lin = snr_power_lin
    else:
        snr_int_lin = num_pulses * snr_power_lin

    snr_int_db = 10.0 * np.log10(np.maximum(1e-20, snr_int_lin))

    return {
        "received_peak_power_w": p_rx,
        "signal_photoelectrons": n_s,
        "background_photoelectrons": n_b,
        "dark_count_photoelectrons": n_d,
        "noise_variance": noise_var,
        "single_pulse_snr_linear": snr_power_lin,
        "single_pulse_snr_db": snr_db,
        "integrated_snr_linear": snr_int_lin,
        "integrated_snr_db": snr_int_db,
    }


def compute_max_lidar_range(
    pulse_energy_j: float,
    pulse_width_s: float,
    wavelength_m: float,
    optics_tx_eff: float,
    optics_rx_eff: float,
    aperture_diam_m: float,
    beam_divergence_rad: float,
    target_optical_cross_section_m2: float,
    min_snr_db: float,
    quantum_efficiency: float = 0.40,
    dark_count_rate_hz: float = 500.0,
    background_power_w: float = 1e-12,
    excess_noise_factor: float = 2.0,
) -> float:
    """
    Compute maximum LIDAR range R_max (meters) for a target optical cross section
    achieving the minimum required SNR threshold.
    Solves via bisection.
    """
    min_snr_lin = 10.0 ** (min_snr_db / 10.0)

    # Initial search brackets: 1 meter to 1,000,000 meters (1,000 km)
    low = 1.0
    high = 1e6

    for _ in range(60):
        mid = 0.5 * (low + high)
        res = compute_lidar_snr(
            pulse_energy_j=pulse_energy_j,
            pulse_width_s=pulse_width_s,
            wavelength_m=wavelength_m,
            optics_tx_eff=optics_tx_eff,
            optics_rx_eff=optics_rx_eff,
            aperture_diam_m=aperture_diam_m,
            beam_divergence_rad=beam_divergence_rad,
            target_optical_cross_section_m2=target_optical_cross_section_m2,
            range_m=mid,
            quantum_efficiency=quantum_efficiency,
            dark_count_rate_hz=dark_count_rate_hz,
            background_power_w=background_power_w,
            excess_noise_factor=excess_noise_factor,
            num_pulses=1,
        )
        if res["single_pulse_snr_linear"] > min_snr_lin:
            low = mid
        else:
            high = mid

    return float(0.5 * (low + high))


# ============================================================================
# SECTION 4: RCS Swerling Target Fluctuation Models (Swerling 0 - IV)
# ============================================================================

def swerling_pdf(
    rcs: Union[float, np.ndarray],
    sigma_mean: float,
    model: SwerlingModel,
) -> Union[float, np.ndarray]:
    """
    Evaluate the theoretical Probability Density Function (PDF) p(sigma) for
    target Radar Cross Section (RCS) under Peter Swerling's models.

    Parameters:
        rcs: Realized RCS value(s) in m^2 (>= 0).
        sigma_mean: Mean target RCS sigma_0 in m^2 (> 0).
        model: SwerlingModel enum (SWERLING_0, I, II, III, IV).

    Returns:
        p(sigma) probability density.
    """
    if sigma_mean <= 0.0:
        raise ValueError("sigma_mean must be strictly positive.")

    rcs_arr = np.asarray(rcs, dtype=float)
    pdf = np.zeros_like(rcs_arr)
    valid = rcs_arr >= 0.0

    if model in (SwerlingModel.SWERLING_I, SwerlingModel.SWERLING_II):
        # Chi-square with 2 degrees of freedom: exponential distribution
        # p(sigma) = (1 / sigma_0) * exp(-sigma / sigma_0)
        pdf[valid] = (1.0 / sigma_mean) * np.exp(-rcs_arr[valid] / sigma_mean)

    elif model in (SwerlingModel.SWERLING_III, SwerlingModel.SWERLING_IV):
        # Chi-square with 4 degrees of freedom:
        # p(sigma) = (4 * sigma / sigma_0^2) * exp(-2 * sigma / sigma_0)
        s = rcs_arr[valid]
        pdf[valid] = ((4.0 * s) / (sigma_mean ** 2)) * np.exp(-2.0 * s / sigma_mean)

    elif model == SwerlingModel.SWERLING_0:
        # Non-fluctuating Marcum target: delta distribution at sigma_mean
        # Approximate with narrow Gaussian or inf
        match = np.isclose(rcs_arr, sigma_mean, atol=1e-6)
        pdf[match] = np.inf

    if np.isscalar(rcs):
        return float(pdf.item())
    return pdf


def sample_swerling_rcs(
    sigma_mean: float,
    model: SwerlingModel,
    num_samples: int = 1,
    num_pulses: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate stochastic Monte Carlo samples of target RCS (m^2) according to
    the Swerling fluctuation model and pulse-to-pulse correlation:

    - Swerling 0: Deterministic constant sigma = sigma_mean.
    - Swerling I: Slow fading (scan-to-scan). Constant across all pulses in a dwell,
                  exponentially distributed across scans.
    - Swerling II: Fast fading (pulse-to-pulse). Independent exponential RCS for each pulse.
    - Swerling III: Slow fading (scan-to-scan). Constant across pulses in a dwell,
                    Gamma(shape=2, scale=sigma_mean/2) distributed across scans.
    - Swerling IV: Fast fading (pulse-to-pulse). Independent Gamma(shape=2, scale=sigma_mean/2)
                   for each pulse.

    Returns:
        Array of shape (num_samples, num_pulses).
    """
    if rng is None:
        rng = np.random.default_rng()

    if sigma_mean <= 0.0:
        raise ValueError("sigma_mean must be positive.")

    if model == SwerlingModel.SWERLING_0:
        return np.full((num_samples, num_pulses), sigma_mean, dtype=float)

    elif model == SwerlingModel.SWERLING_I:
        # 1 realization per scan, shared across all pulses
        samples = rng.exponential(scale=sigma_mean, size=(num_samples, 1))
        return np.repeat(samples, num_pulses, axis=1)

    elif model == SwerlingModel.SWERLING_II:
        # Independent realization per pulse
        return rng.exponential(scale=sigma_mean, size=(num_samples, num_pulses))

    elif model == SwerlingModel.SWERLING_III:
        # 1 realization per scan from Gamma(2, sigma_0 / 2)
        samples = rng.gamma(shape=2.0, scale=sigma_mean / 2.0, size=(num_samples, 1))
        return np.repeat(samples, num_pulses, axis=1)

    elif model == SwerlingModel.SWERLING_IV:
        # Independent realization per pulse from Gamma(2, sigma_0 / 2)
        return rng.gamma(shape=2.0, scale=sigma_mean / 2.0, size=(num_samples, num_pulses))

    raise ValueError(f"Unknown SwerlingModel: {model}")


def calculate_pd(
    snr_linear: float,
    pfa: float,
    model: SwerlingModel,
    num_pulses: int = 1,
) -> float:
    """
    Calculate theoretical Detection Probability (Pd) for Swerling targets in
    Gaussian white noise given linear SNR and False Alarm Probability (Pfa).

    Exact analytical formulas:
    - Swerling 0: Q_1(sqrt(2 * SNR), sqrt(-2 * ln(Pfa))) (Marcum Q)
    - Swerling I: Pd = (Pfa)^(1 / (1 + SNR))
    - Swerling II: Post-detection integrated Erlang tail integral
    - Swerling III: Pd = [1 + 2*Vt*SNR / (2 + SNR)^2] * (Pfa)^(2 / (2 + SNR)) for N=1
    - Swerling IV: Fast-fading Gamma sum for N pulses
    """
    if pfa <= 0.0 or pfa >= 1.0:
        raise ValueError("pfa must be in (0, 1)")
    if snr_linear < 0.0:
        raise ValueError("snr_linear must be non-negative")

    vt = solve_gamma_threshold_pfa(pfa, num_pulses=1)

    if model == SwerlingModel.SWERLING_0:
        a = math.sqrt(2.0 * snr_linear)
        b = math.sqrt(2.0 * vt)
        return marcum_q1(a, b)

    elif model == SwerlingModel.SWERLING_I:
        # Single-pulse or slow-fading coherent dwell
        # Pd = exp(-Vt / (1 + SNR)) = (Pfa)^(1 / (1 + SNR))
        return float(math.pow(pfa, 1.0 / (1.0 + snr_linear)))

    elif model == SwerlingModel.SWERLING_II:
        if num_pulses == 1:
            return float(math.pow(pfa, 1.0 / (1.0 + snr_linear)))
        vt_n = solve_gamma_threshold_pfa(pfa, num_pulses=num_pulses)
        scaled_vt = vt_n / (1.0 + snr_linear)
        log_pd = _log_poisson_tail(num_pulses, scaled_vt)
        return float(np.clip(math.exp(log_pd), 0.0, 1.0))

    elif model == SwerlingModel.SWERLING_III:
        # Dominant scatterer, slow fading
        # Pd = [1 + (2 * Vt * SNR) / ((2 + SNR)^2)] * (Pfa)^(2 / (2 + SNR))
        term = 1.0 + (2.0 * vt * snr_linear) / ((2.0 + snr_linear) ** 2)
        exponent = 2.0 / (2.0 + snr_linear)
        return float(np.clip(term * math.pow(pfa, exponent), 0.0, 1.0))

    elif model == SwerlingModel.SWERLING_IV:
        if num_pulses == 1:
            term = 1.0 + (2.0 * vt * snr_linear) / ((2.0 + snr_linear) ** 2)
            exponent = 2.0 / (2.0 + snr_linear)
            return float(np.clip(term * math.pow(pfa, exponent), 0.0, 1.0))
        # Fast fading dominant scatterer: effective SNR scaled by pulse count
        eff_snr = snr_linear * math.sqrt(num_pulses)
        term = 1.0 + (2.0 * vt * eff_snr) / ((2.0 + eff_snr) ** 2)
        exponent = 2.0 / (2.0 + eff_snr)
        return float(np.clip(term * math.pow(pfa, exponent), 0.0, 1.0))

    raise ValueError(f"Unknown SwerlingModel: {model}")


# ============================================================================
# SECTION 5: Range-Rate Doppler Shifts, Ambiguities & CRLB
# ============================================================================

def compute_doppler_shift(
    carrier_freq_hz: float,
    range_rate_m_s: Union[float, np.ndarray],
    relativistic: bool = False,
) -> Union[float, np.ndarray]:
    """
    Compute Doppler frequency shift f_D (Hz) from relative range-rate (m/s).
    Convention:
        - range_rate < 0 (approaching/closing): positive Doppler shift (+f_D)
        - range_rate > 0 (receding/opening): negative Doppler shift (-f_D)

    Parameters:
        carrier_freq_hz: Radar or laser carrier frequency (Hz).
        range_rate_m_s: Relative line-of-sight velocity dR/dt (m/s).
        relativistic: If True, uses exact relativistic Doppler formula.

    Returns:
        Doppler frequency shift f_D in Hz.
    """
    if carrier_freq_hz <= 0.0:
        raise ValueError("carrier_freq_hz must be positive.")

    rr_arr = np.asarray(range_rate_m_s, dtype=float)

    if not relativistic:
        # Classical two-way radar Doppler: f_D = -2 * (dR/dt) / lambda = -2 * (dR/dt) * f_0 / c
        fd = -2.0 * rr_arr * carrier_freq_hz / SPEED_OF_LIGHT
    else:
        # Relativistic Doppler for two-way reflection:
        # beta = (dR/dt) / c
        beta = rr_arr / SPEED_OF_LIGHT
        # Closing: f_rx = f_0 * (1 - beta) / (1 + beta)
        ratio = (1.0 - beta) / (1.0 + beta)
        ratio = np.maximum(1e-12, ratio)
        fd = carrier_freq_hz * (ratio - 1.0)

    if np.isscalar(range_rate_m_s):
        return float(fd.item())
    return fd


def compute_range_rate_from_doppler(
    carrier_freq_hz: float,
    doppler_shift_hz: Union[float, np.ndarray],
    relativistic: bool = False,
) -> Union[float, np.ndarray]:
    """
    Invert Doppler shift f_D (Hz) to obtain relative range-rate dR/dt (m/s).
    """
    if carrier_freq_hz <= 0.0:
        raise ValueError("carrier_freq_hz must be positive.")

    fd_arr = np.asarray(doppler_shift_hz, dtype=float)

    if not relativistic:
        rr = -0.5 * fd_arr * SPEED_OF_LIGHT / carrier_freq_hz
    else:
        ratio = 1.0 + (fd_arr / carrier_freq_hz)
        # ratio = (1 - beta) / (1 + beta) => beta = (1 - ratio) / (1 + ratio)
        beta = (1.0 - ratio) / (1.0 + ratio)
        rr = beta * SPEED_OF_LIGHT

    if np.isscalar(doppler_shift_hz):
        return float(rr.item())
    return rr


def compute_radar_ambiguities(
    prf_hz: float,
    carrier_freq_hz: float,
) -> Dict[str, float]:
    """
    Compute radar unambiguous range, unambiguous velocity, and the Doppler Dilemma product.

    Parameters:
        prf_hz: Pulse Repetition Frequency (Hz).
        carrier_freq_hz: Radar carrier frequency (Hz).

    Returns:
        Dictionary with:
            pri_s: Pulse repetition interval (s).
            unambiguous_range_m: R_ua = c / (2 * PRF).
            unambiguous_range_km: R_ua in km.
            unambiguous_velocity_m_s: v_ua = lambda * PRF / 4.
            wavelength_m: Carrier wavelength lambda (m).
            doppler_dilemma_product: R_ua * v_ua = (c * lambda) / 8.
    """
    if prf_hz <= 0.0:
        raise ValueError("prf_hz must be positive.")
    if carrier_freq_hz <= 0.0:
        raise ValueError("carrier_freq_hz must be positive.")

    pri_s = 1.0 / prf_hz
    wavelength_m = SPEED_OF_LIGHT / carrier_freq_hz

    r_ua_m = SPEED_OF_LIGHT / (2.0 * prf_hz)
    v_ua_m_s = (wavelength_m * prf_hz) / 4.0
    product = r_ua_m * v_ua_m_s

    return {
        "pri_s": pri_s,
        "unambiguous_range_m": r_ua_m,
        "unambiguous_range_km": r_ua_m / 1000.0,
        "unambiguous_velocity_m_s": v_ua_m_s,
        "wavelength_m": wavelength_m,
        "doppler_dilemma_product": product,
    }


def compute_crlb_uncertainty(
    snr_linear: float,
    chirp_bandwidth_hz: float,
    obs_time_s: float,
    carrier_freq_hz: float,
) -> Dict[str, float]:
    """
    Compute Cramer-Rao Lower Bounds (CRLB) on Range and Range-Rate measurement variances.

    Parameters:
        snr_linear: Linear Signal-to-Noise Ratio (SNR >= 0.1).
        chirp_bandwidth_hz: Linear FM chirp modulation bandwidth (Hz).
        obs_time_s: Coherent integration observation dwell time (seconds).
        carrier_freq_hz: Radar carrier frequency (Hz).

    Returns:
        Dictionary with:
            sigma_range_m: 1-sigma range measurement standard deviation (m).
            sigma_velocity_m_s: 1-sigma range-rate standard deviation (m/s).
            sigma_doppler_hz: 1-sigma Doppler frequency error (Hz).
            range_resolution_m: delta_R = c / (2 * B).
            velocity_resolution_m_s: delta_v = lambda / (2 * T_obs).
    """
    if snr_linear <= 0.0:
        raise ValueError("snr_linear must be positive.")
    if chirp_bandwidth_hz <= 0.0:
        raise ValueError("chirp_bandwidth_hz must be positive.")
    if obs_time_s <= 0.0:
        raise ValueError("obs_time_s must be positive.")

    wavelength_m = SPEED_OF_LIGHT / carrier_freq_hz

    # Range CRLB for rectangular / LFM chirp:
    # sigma_R = (sqrt(3) * c) / (2 * pi * B * sqrt(2 * SNR))
    denom_r = 2.0 * math.pi * chirp_bandwidth_hz * math.sqrt(2.0 * snr_linear)
    sigma_r = (math.sqrt(3.0) * SPEED_OF_LIGHT) / denom_r

    # Velocity CRLB for coherent dwell T_obs:
    # sigma_v = (sqrt(3) * lambda) / (pi * T_obs * sqrt(2 * SNR))
    denom_v = math.pi * obs_time_s * math.sqrt(2.0 * snr_linear)
    sigma_v = (math.sqrt(3.0) * wavelength_m) / denom_v

    sigma_fd = 2.0 * sigma_v / wavelength_m

    res_r = SPEED_OF_LIGHT / (2.0 * chirp_bandwidth_hz)
    res_v = wavelength_m / (2.0 * obs_time_s)

    return {
        "sigma_range_m": sigma_r,
        "sigma_velocity_m_s": sigma_v,
        "sigma_doppler_hz": sigma_fd,
        "range_resolution_m": res_r,
        "velocity_resolution_m_s": res_v,
    }


# ============================================================================
# SECTION 6: Detection Thresholding & CFAR Algorithms
# ============================================================================

def detection_threshold_pfa(
    pfa: float,
    noise_power_w: float = 1.0,
) -> float:
    """
    Compute fixed square-law detection threshold V_T for a given False Alarm Probability.
    V_T = -noise_power * ln(Pfa).
    """
    if pfa <= 0.0 or pfa >= 1.0:
        raise ValueError("pfa must be in (0, 1)")
    if noise_power_w <= 0.0:
        raise ValueError("noise_power_w must be positive.")
    return float(-noise_power_w * math.log(pfa))


def ca_cfar_1d(
    signal_power: np.ndarray,
    num_guard_cells: int = 2,
    num_ref_cells: int = 16,
    pfa: float = 1e-4,
) -> Dict[str, np.ndarray]:
    """
    1D Cell-Averaging Constant False Alarm Rate (CA-CFAR) detector.

    Parameters:
        signal_power: 1D array of detected power values (e.g., range profile or Doppler spectrum).
        num_guard_cells: Number of guard cells on each side of the cell under test (CUT).
        num_ref_cells: Number of training/reference cells on each side of the guard cells.
        pfa: Desired false alarm probability.

    Returns:
        Dictionary with:
            threshold: Adaptive threshold values across the signal array.
            detections: Boolean array where True indicates detection (CUT > threshold).
            alpha: Multiplicative threshold factor.
    """
    sig = np.asarray(signal_power, dtype=float)
    n_total = len(sig)
    num_train = 2 * num_ref_cells

    # Multiplicative factor: alpha = N * (Pfa^(-1/N) - 1)
    alpha = num_train * (math.pow(pfa, -1.0 / num_train) - 1.0)

    threshold = np.zeros(n_total, dtype=float)
    detections = np.zeros(n_total, dtype=bool)

    # Window span
    window_half = num_guard_cells + num_ref_cells

    for i in range(n_total):
        # Left reference slice
        left_start = max(0, i - window_half)
        left_end = max(0, i - num_guard_cells)
        # Right reference slice
        right_start = min(n_total, i + num_guard_cells + 1)
        right_end = min(n_total, i + window_half + 1)

        left_cells = sig[left_start:left_end]
        right_cells = sig[right_start:right_end]
        ref = np.concatenate([left_cells, right_cells])

        if len(ref) > 0:
            noise_est = np.mean(ref)
        else:
            noise_est = sig[i]

        t_val = alpha * noise_est
        threshold[i] = t_val
        detections[i] = sig[i] > t_val

    return {
        "threshold": threshold,
        "detections": detections,
        "alpha": alpha,
    }


def ca_cfar_2d(
    matrix_power: np.ndarray,
    guard_cells: Tuple[int, int] = (2, 2),
    ref_cells: Tuple[int, int] = (4, 4),
    pfa: float = 1e-4,
) -> Dict[str, Any]:
    """
    2D Cell-Averaging CFAR detector for Range-Doppler matrices.

    Parameters:
        matrix_power: 2D array of power values (shape [N_range, N_doppler]).
        guard_cells: (guard_range, guard_doppler) half-widths.
        ref_cells: (ref_range, ref_doppler) half-widths of reference band.
        pfa: Desired false alarm probability.

    Returns:
        Dictionary with:
            threshold_map: 2D array of adaptive thresholds.
            detections_map: 2D boolean mask of detections.
            alpha: Multiplicative threshold factor.
    """
    mat = np.asarray(matrix_power, dtype=float)
    rows, cols = mat.shape

    gr, gd = guard_cells
    rr, rd = ref_cells

    total_win_rows = 2 * (gr + rr) + 1
    total_win_cols = 2 * (gd + rd) + 1
    guard_win_rows = 2 * gr + 1
    guard_win_cols = 2 * gd + 1

    n_train = (total_win_rows * total_win_cols) - (guard_win_rows * guard_win_cols)
    alpha = n_train * (math.pow(pfa, -1.0 / n_train) - 1.0)

    threshold_map = np.zeros_like(mat)
    detections_map = np.zeros_like(mat, dtype=bool)

    for r in range(rows):
        r_min = max(0, r - (gr + rr))
        r_max = min(rows, r + (gr + rr) + 1)
        g_r_min = max(0, r - gr)
        g_r_max = min(rows, r + gr + 1)

        for c in range(cols):
            c_min = max(0, c - (gd + rd))
            c_max = min(cols, c + (gd + rd) + 1)
            g_c_min = max(0, c - gd)
            g_c_max = min(cols, c + gd + 1)

            outer_sum = np.sum(mat[r_min:r_max, c_min:c_max])
            outer_count = (r_max - r_min) * (c_max - c_min)

            inner_sum = np.sum(mat[g_r_min:g_r_max, g_c_min:g_c_max])
            inner_count = (g_r_max - g_r_min) * (g_c_max - g_c_min)

            train_sum = outer_sum - inner_sum
            train_count = outer_count - inner_count

            if train_count > 0:
                noise_est = train_sum / train_count
            else:
                noise_est = mat[r, c]

            thresh = alpha * noise_est
            threshold_map[r, c] = thresh
            detections_map[r, c] = mat[r, c] > thresh

    return {
        "threshold_map": threshold_map,
        "detections_map": detections_map,
        "alpha": alpha,
    }


def os_cfar_1d(
    signal_power: np.ndarray,
    num_guard_cells: int = 2,
    num_ref_cells: int = 16,
    pfa: float = 1e-4,
    k_rank: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    1D Order-Statistic CFAR (OS-CFAR) detector.
    Robust against multiple target interference in dense orbital debris clusters.

    Parameters:
        signal_power: 1D array of power values.
        num_guard_cells: Guard cells on each side.
        num_ref_cells: Reference cells on each side (total reference N = 2 * num_ref_cells).
        pfa: Desired false alarm probability.
        k_rank: Order statistic rank k (1 <= k <= N). Default is round(0.75 * N).

    Returns:
        Dictionary with:
            threshold: Adaptive OS-CFAR threshold array.
            detections: Boolean detection mask.
            alpha_os: Scaling multiplier.
            k_rank: Rank used.
    """
    sig = np.asarray(signal_power, dtype=float)
    n_total = len(sig)
    num_train = 2 * num_ref_cells

    if k_rank is None:
        k_rank = int(round(0.75 * num_train))
    k_rank = max(1, min(num_train, k_rank))

    # Solve for alpha_os: Pfa = prod_{i=0}^{k-1} (N - i) / (N - i + alpha)
    target_log_pfa = math.log(pfa)
    low = 0.0
    high = 1000.0
    for _ in range(35):
        mid = 0.5 * (low + high)
        val = sum(math.log(num_train - i) - math.log(num_train - i + mid) for i in range(k_rank))
        if val > target_log_pfa:
            low = mid
        else:
            high = mid
    alpha_os = 0.5 * (low + high)

    threshold = np.zeros(n_total, dtype=float)
    detections = np.zeros(n_total, dtype=bool)
    window_half = num_guard_cells + num_ref_cells

    for i in range(n_total):
        left_start = max(0, i - window_half)
        left_end = max(0, i - num_guard_cells)
        right_start = min(n_total, i + num_guard_cells + 1)
        right_end = min(n_total, i + window_half + 1)

        ref = np.concatenate([sig[left_start:left_end], sig[right_start:right_end]])
        if len(ref) >= k_rank:
            sorted_ref = np.sort(ref)
            k_val = sorted_ref[k_rank - 1]
        elif len(ref) > 0:
            k_val = np.median(ref)
        else:
            k_val = sig[i]

        t_val = alpha_os * k_val
        threshold[i] = t_val
        detections[i] = sig[i] > t_val

    return {
        "threshold": threshold,
        "detections": detections,
        "alpha_os": alpha_os,
        "k_rank": k_rank,
    }


# ============================================================================
# SECTION 7: Sensor Configurations & Observation Models
# ============================================================================

@dataclass
class SpaceborneRadarConfig:
    """Design parameters for a spaceborne active tracking radar."""
    frequency_hz: float = 10.0e9             # 10 GHz (X-band)
    peak_power_w: float = 1500.0             # 1.5 kW peak power
    tx_gain_dbi: float = 42.0                # Transmit gain
    rx_gain_dbi: float = 42.0                # Receive gain
    chirp_bandwidth_hz: float = 100.0e6      # 100 MHz LFM chirp (1.5 m range res)
    pulse_width_s: float = 20.0e-6           # 20 microseconds
    prf_hz: float = 2000.0                   # 2 kHz PRF
    noise_figure_db: float = 3.0             # 3.0 dB LNA noise figure
    system_loss_db: float = 4.0              # 4.0 dB two-way system loss
    noise_temp_k: float = STANDARD_TEMP_K    # 290 K
    pfa: float = 1e-6                        # 1e-6 detection false alarm rate
    num_integrated_pulses: int = 16          # Coherent burst integration
    coherent_integration: bool = True

    @property
    def wavelength_m(self) -> float:
        return SPEED_OF_LIGHT / self.frequency_hz


@dataclass
class RadarObservation:
    """Emulated spaceborne radar observation packet."""
    detected: bool
    true_range_m: float
    measured_range_m: Optional[float]
    true_range_rate_m_s: float
    measured_range_rate_m_s: Optional[float]
    doppler_shift_hz: float
    received_power_w: float
    single_pulse_snr_db: float
    integrated_snr_db: float
    detection_probability: float
    realized_rcs_m2: float
    sigma_range_m: float
    sigma_velocity_m_s: float
    range_error_m: Optional[float]
    velocity_error_m_s: Optional[float]


class SpaceborneRadar:
    """
    Active Spaceborne Tracking Radar emulation engine for orbital conjunctions.
    """

    def __init__(
        self,
        config: Optional[SpaceborneRadarConfig] = None,
        pt_watts: Optional[float] = None,
        freq_hz: Optional[float] = None,
        antenna_gain_dbi: Optional[float] = None,
        **kwargs,
    ):
        if config is None:
            config = SpaceborneRadarConfig(
                frequency_hz=freq_hz if freq_hz is not None else 10.0e9,
                peak_power_w=pt_watts if pt_watts is not None else 1500.0,
                tx_gain_dbi=antenna_gain_dbi if antenna_gain_dbi is not None else 42.0,
                rx_gain_dbi=antenna_gain_dbi if antenna_gain_dbi is not None else 42.0,
            )
        self.config = config

    def compute_snr_db(self, range_km: float, rcs_m2: float) -> float:
        """Compute integrated SNR in dB for given target distance in km and RCS in m^2."""
        res = compute_radar_snr(
            peak_power_w=self.config.peak_power_w,
            tx_gain_dbi=self.config.tx_gain_dbi,
            rx_gain_dbi=self.config.rx_gain_dbi,
            wavelength_m=self.config.wavelength_m,
            rcs_m2=rcs_m2,
            range_m=range_km * 1000.0,
            bandwidth_hz=self.config.chirp_bandwidth_hz,
            noise_temp_k=self.config.noise_temp_k,
            noise_figure_db=self.config.noise_figure_db,
            system_loss_db=self.config.system_loss_db,
            num_pulses=self.config.num_integrated_pulses,
            coherent=self.config.coherent_integration,
        )
        return float(res.get("integrated_snr_db", res.get("snr_db", 0.0)))

    def calculate_pd(
        self,
        snr_linear: float,
        pfa: float = 1e-6,
        model: Union[str, SwerlingModel] = "SWERLING_1",
    ) -> float:
        """Calculate probability of detection given linear SNR, Pfa, and Swerling model."""
        if isinstance(model, str):
            mapping = {
                "SWERLING_0": SwerlingModel.SWERLING_0,
                "SWERLING_1": SwerlingModel.SWERLING_I,
                "SWERLING_2": SwerlingModel.SWERLING_II,
                "SWERLING_3": SwerlingModel.SWERLING_III,
                "SWERLING_4": SwerlingModel.SWERLING_IV,
            }
            model = mapping.get(model.upper(), SwerlingModel.SWERLING_I)
        return float(calculate_pd(snr_linear=snr_linear, pfa=pfa, model=model, num_pulses=self.config.num_integrated_pulses))

    def observe(
        self,
        target_pos_km: Union[List[float], np.ndarray],
        target_vel_km: Union[List[float], np.ndarray],
        radar_pos_km: Union[List[float], np.ndarray],
        radar_vel_km: Union[List[float], np.ndarray],
        mean_rcs_m2: float = 1.0,
        swerling_model: SwerlingModel = SwerlingModel.SWERLING_I,
        rng: Optional[np.random.Generator] = None,
    ) -> RadarObservation:
        """
        Simulate radar tracking observation of an orbital target.

        Parameters:
            target_pos_km: Target ECI position [x, y, z] in km.
            target_vel_km: Target ECI velocity [vx, vy, vz] in km/s.
            radar_pos_km: Radar host satellite ECI position [x, y, z] in km.
            radar_vel_km: Radar host satellite ECI velocity [vx, vy, vz] in km/s.
            mean_rcs_m2: Mean target Radar Cross Section in m^2.
            swerling_model: Swerling target fluctuation model.
            rng: Random number generator.

        Returns:
            RadarObservation dataclass.
        """
        if rng is None:
            rng = np.random.default_rng()

        r_tgt = np.asarray(target_pos_km, dtype=float) * 1000.0  # to meters
        v_tgt = np.asarray(target_vel_km, dtype=float) * 1000.0  # to m/s
        r_rad = np.asarray(radar_pos_km, dtype=float) * 1000.0
        v_rad = np.asarray(radar_vel_km, dtype=float) * 1000.0

        rel_pos = r_tgt - r_rad
        true_range_m = float(np.linalg.norm(rel_pos))
        if true_range_m < 1.0:
            true_range_m = 1.0

        los_unit = rel_pos / true_range_m
        rel_vel = v_tgt - v_rad
        true_range_rate_m_s = float(np.dot(rel_vel, los_unit))

        # Sample target RCS under Swerling model
        rcs_samples = sample_swerling_rcs(
            sigma_mean=mean_rcs_m2,
            model=swerling_model,
            num_samples=1,
            num_pulses=self.config.num_integrated_pulses,
            rng=rng,
        )
        realized_rcs = float(np.mean(rcs_samples))

        # Compute SNR
        snr_info = compute_radar_snr(
            peak_power_w=self.config.peak_power_w,
            tx_gain_dbi=self.config.tx_gain_dbi,
            rx_gain_dbi=self.config.rx_gain_dbi,
            wavelength_m=self.config.wavelength_m,
            rcs_m2=realized_rcs,
            range_m=true_range_m,
            noise_temp_k=self.config.noise_temp_k,
            noise_figure_db=self.config.noise_figure_db,
            bandwidth_hz=self.config.chirp_bandwidth_hz,
            system_loss_db=self.config.system_loss_db,
            num_pulses=self.config.num_integrated_pulses,
            coherent=self.config.coherent_integration,
        )

        int_snr_lin = snr_info["integrated_snr_linear"]
        pd = calculate_pd(
            snr_linear=int_snr_lin,
            pfa=self.config.pfa,
            model=swerling_model,
            num_pulses=self.config.num_integrated_pulses,
        )

        # Detection decision: Bernoulli trial based on Pd
        detected = bool(rng.uniform(0.0, 1.0) < pd)

        # Doppler
        doppler_hz = compute_doppler_shift(
            carrier_freq_hz=self.config.frequency_hz,
            range_rate_m_s=true_range_rate_m_s,
        )

        # CRLB uncertainties
        obs_time = self.config.num_integrated_pulses / self.config.prf_hz
        crlb = compute_crlb_uncertainty(
            snr_linear=max(0.1, int_snr_lin),
            chirp_bandwidth_hz=self.config.chirp_bandwidth_hz,
            obs_time_s=obs_time,
            carrier_freq_hz=self.config.frequency_hz,
        )

        sigma_r = crlb["sigma_range_m"]
        sigma_v = crlb["sigma_velocity_m_s"]

        if detected:
            range_err = float(rng.normal(0.0, sigma_r))
            vel_err = float(rng.normal(0.0, sigma_v))
            measured_range = true_range_m + range_err
            measured_range_rate = true_range_rate_m_s + vel_err
        else:
            range_err = None
            vel_err = None
            measured_range = None
            measured_range_rate = None

        return RadarObservation(
            detected=detected,
            true_range_m=true_range_m,
            measured_range_m=measured_range,
            true_range_rate_m_s=true_range_rate_m_s,
            measured_range_rate_m_s=measured_range_rate,
            doppler_shift_hz=doppler_hz,
            received_power_w=snr_info["received_power_w"],
            single_pulse_snr_db=snr_info["snr_db"],
            integrated_snr_db=snr_info["integrated_snr_db"],
            detection_probability=pd,
            realized_rcs_m2=realized_rcs,
            sigma_range_m=sigma_r,
            sigma_velocity_m_s=sigma_v,
            range_error_m=range_err,
            velocity_error_m_s=vel_err,
        )


@dataclass
class SpaceborneLidarConfig:
    """Design parameters for a spaceborne active pulsed LIDAR."""
    wavelength_m: float = 1064.0e-9           # 1064 nm Nd:YAG
    pulse_energy_j: float = 0.05              # 50 mJ per pulse
    pulse_width_s: float = 5.0e-9             # 5 ns pulse duration
    prf_hz: float = 1000.0                    # 1 kHz repetition rate
    beam_divergence_rad: float = 0.5e-3       # 0.5 mrad divergence
    aperture_diam_m: float = 0.20             # 20 cm receiver telescope
    optics_tx_eff: float = 0.85               # 85% transmitter optics efficiency
    optics_rx_eff: float = 0.80               # 80% receiver optics efficiency
    quantum_efficiency: float = 0.40          # 40% APD quantum efficiency
    dark_count_rate_hz: float = 500.0         # 500 Hz dark count
    background_power_w: float = 1.0e-12       # 1 pW solar background
    excess_noise_factor: float = 2.0          # APD excess noise factor
    pfa: float = 1e-6
    num_integrated_pulses: int = 10


@dataclass
class LidarObservation:
    """Emulated spaceborne LIDAR observation packet."""
    detected: bool
    true_range_m: float
    measured_range_m: Optional[float]
    true_range_rate_m_s: float
    measured_range_rate_m_s: Optional[float]
    signal_photoelectrons: float
    background_photoelectrons: float
    optical_snr_db: float
    detection_probability: float
    sigma_range_m: float
    sigma_velocity_m_s: float
    range_error_m: Optional[float]
    velocity_error_m_s: Optional[float]


class SpaceborneLidar:
    """
    Active Spaceborne Pulsed LIDAR emulation engine for orbital rendezvous and close-range debris.
    """

    def __init__(self, config: Optional[SpaceborneLidarConfig] = None):
        self.config = config or SpaceborneLidarConfig()

    def observe(
        self,
        target_pos_km: Union[List[float], np.ndarray],
        target_vel_km: Union[List[float], np.ndarray],
        lidar_pos_km: Union[List[float], np.ndarray],
        lidar_vel_km: Union[List[float], np.ndarray],
        optical_cross_section_m2: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> LidarObservation:
        """
        Simulate LIDAR observation of a target.
        """
        if rng is None:
            rng = np.random.default_rng()

        r_tgt = np.asarray(target_pos_km, dtype=float) * 1000.0
        v_tgt = np.asarray(target_vel_km, dtype=float) * 1000.0
        r_lid = np.asarray(lidar_pos_km, dtype=float) * 1000.0
        v_lid = np.asarray(lidar_vel_km, dtype=float) * 1000.0

        rel_pos = r_tgt - r_lid
        true_range_m = float(np.linalg.norm(rel_pos))
        if true_range_m < 0.1:
            true_range_m = 0.1

        los_unit = rel_pos / true_range_m
        rel_vel = v_tgt - v_lid
        true_range_rate_m_s = float(np.dot(rel_vel, los_unit))

        snr_info = compute_lidar_snr(
            pulse_energy_j=self.config.pulse_energy_j,
            pulse_width_s=self.config.pulse_width_s,
            wavelength_m=self.config.wavelength_m,
            optics_tx_eff=self.config.optics_tx_eff,
            optics_rx_eff=self.config.optics_rx_eff,
            aperture_diam_m=self.config.aperture_diam_m,
            beam_divergence_rad=self.config.beam_divergence_rad,
            target_optical_cross_section_m2=optical_cross_section_m2,
            range_m=true_range_m,
            quantum_efficiency=self.config.quantum_efficiency,
            dark_count_rate_hz=self.config.dark_count_rate_hz,
            background_power_w=self.config.background_power_w,
            excess_noise_factor=self.config.excess_noise_factor,
            num_pulses=self.config.num_integrated_pulses,
        )

        snr_lin = snr_info["integrated_snr_linear"]
        pd = calculate_pd(
            snr_linear=snr_lin,
            pfa=self.config.pfa,
            model=SwerlingModel.SWERLING_0,
            num_pulses=self.config.num_integrated_pulses,
        )

        detected = bool(rng.uniform(0.0, 1.0) < pd)

        # LIDAR range precision: sigma_R = (c * tau_pulse) / (2 * sqrt(2 * SNR))
        snr_eff = max(0.1, snr_lin)
        sigma_r = (SPEED_OF_LIGHT * self.config.pulse_width_s) / (2.0 * math.sqrt(2.0 * snr_eff))

        # Velocity precision via pulse-pair or Doppler
        obs_time = self.config.num_integrated_pulses / self.config.prf_hz
        sigma_v = (math.sqrt(3.0) * self.config.wavelength_m) / (math.pi * obs_time * math.sqrt(2.0 * snr_eff))

        if detected:
            range_err = float(rng.normal(0.0, sigma_r))
            vel_err = float(rng.normal(0.0, sigma_v))
            measured_range = true_range_m + range_err
            measured_range_rate = true_range_rate_m_s + vel_err
        else:
            range_err = None
            vel_err = None
            measured_range = None
            measured_range_rate = None

        return LidarObservation(
            detected=detected,
            true_range_m=true_range_m,
            measured_range_m=measured_range,
            true_range_rate_m_s=true_range_rate_m_s,
            measured_range_rate_m_s=measured_range_rate,
            signal_photoelectrons=snr_info["signal_photoelectrons"],
            background_photoelectrons=snr_info["background_photoelectrons"],
            optical_snr_db=snr_info["integrated_snr_db"],
            detection_probability=pd,
            sigma_range_m=sigma_r,
            sigma_velocity_m_s=sigma_v,
            range_error_m=range_err,
            velocity_error_m_s=vel_err,
        )


# ============================================================================
# SECTION 8: Synthetic Range-Doppler Map Generation
# ============================================================================

def generate_range_doppler_map(
    targets: List[Dict[str, float]],
    radar_config: Optional[SpaceborneRadarConfig] = None,
    range_bins_m: Optional[np.ndarray] = None,
    velocity_bins_m_s: Optional[np.ndarray] = None,
    noise_floor_w: float = 1e-13,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize a 2D Range-Doppler radar power matrix with complex thermal noise and multiple orbital target returns.
    """
    if radar_config is None:
        radar_config = SpaceborneRadarConfig()
    if rng is None:
        rng = np.random.default_rng()

    return_km = False
    norm_targets = []
    max_r_km = 30.0
    for tgt in targets:
        r_m = tgt.get("range_m", tgt.get("range_km", 15.0) * (1.0 if "range_m" in tgt else 1000.0))
        v_m_s = tgt.get("velocity_m_s", tgt.get("relative_speed_km_s", 0.0) * (1.0 if "velocity_m_s" in tgt else 1000.0))
        rcs = tgt.get("rcs_m2", 1.0)
        norm_targets.append({"range_m": r_m, "velocity_m_s": v_m_s, "rcs_m2": rcs})
        max_r_km = max(max_r_km, (r_m / 1000.0) * 1.5)

    if range_bins_m is None:
        range_bins_km = np.linspace(0.0, max(30.0, max_r_km), 64)
        range_bins_m = range_bins_km * 1000.0
        return_km = True
    else:
        range_bins_km = range_bins_m / 1000.0

    if velocity_bins_m_s is None:
        velocity_bins_km_s = np.linspace(-5.0, 5.0, 64)
        velocity_bins_m_s = velocity_bins_km_s * 1000.0
    else:
        velocity_bins_km_s = velocity_bins_m_s / 1000.0

    nr = len(range_bins_m)
    nv = len(velocity_bins_m_s)

    # Resolution scales for point spread function (PSF)
    res_r = SPEED_OF_LIGHT / (2.0 * radar_config.chirp_bandwidth_hz)
    obs_time = radar_config.num_integrated_pulses / radar_config.prf_hz
    res_v = radar_config.wavelength_m / (2.0 * obs_time)

    # Complex Gaussian noise field: real + 1j * imag
    sigma_noise = math.sqrt(0.5 * noise_floor_w)
    noise_real = rng.normal(0.0, sigma_noise, size=(nr, nv))
    noise_imag = rng.normal(0.0, sigma_noise, size=(nr, nv))
    signal_complex = noise_real + 1j * noise_imag

    r_grid, v_grid = np.meshgrid(range_bins_m, velocity_bins_m_s, indexing="ij")

    for tgt in norm_targets:
        r_t = tgt["range_m"]
        v_t = tgt["velocity_m_s"]
        rcs_t = tgt["rcs_m2"]

        p_rx = radar_range_equation(
            peak_power_w=radar_config.peak_power_w,
            tx_gain_dbi=radar_config.tx_gain_dbi,
            rx_gain_dbi=radar_config.rx_gain_dbi,
            wavelength_m=radar_config.wavelength_m,
            rcs_m2=rcs_t,
            range_m=r_t,
            system_loss_db=radar_config.system_loss_db,
        )

        amplitude = math.sqrt(p_rx * radar_config.num_integrated_pulses)
        phase = rng.uniform(0.0, 2.0 * math.pi)

        # 2D Sinc-like Point Spread Function (PSF)
        dr = (r_grid - r_t) / res_r
        dv = (v_grid - v_t) / res_v

        # Sinc approximation: sin(pi*x) / (pi*x)
        psf = np.sinc(dr) * np.sinc(dv)
        signal_complex += amplitude * psf * np.exp(1j * phase)

    power_map = np.abs(signal_complex) ** 2
    if return_km:
        return power_map, range_bins_km, velocity_bins_km_s
    return power_map, range_bins_m, velocity_bins_m_s
