"""
Unit and Integration Tests for AEGIS-LEO Spaceborne Radar and LIDAR Sensor Emulation.

Covers:
- Physical constants & radar/lidar band enums
- Scaled modified Bessel I0 & Marcum Q-function
- Monostatic Radar Range Equation & thermal SNR models (single & integrated)
- Inverted maximum radar range solver
- Active LIDAR Range Equation (unresolved & resolved diffuse) & photon shot-noise SNR
- Maximum LIDAR range solver
- RCS Swerling target fluctuation models (Swerling 0 / V, I, II, III, IV)
- Detection probability (Pd) analytical calculations & boundary limits
- Range-rate Doppler shift (classical and relativistic) & round-trip inversion
- Radar unambiguous range, unambiguous velocity, and Doppler dilemma product
- Cramer-Rao Lower Bounds (CRLB) for range and velocity measurement errors
- False alarm thresholding, 1D/2D CA-CFAR, and 1D OS-CFAR detectors
- SpaceborneRadar and SpaceborneLidar orbital observation engines
- Synthetic 2D Range-Doppler map generation
"""

import math
import numpy as np
import pytest

from aegis_leo.sensors import (
    SPEED_OF_LIGHT,
    BOLTZMANN_CONSTANT,
    PLANCK_CONSTANT,
    STANDARD_TEMP_K,
    SwerlingModel,
    RadarBand,
    LidarWavelength,
    bessel_i0_scaled,
    marcum_q1,
    solve_gamma_threshold_pfa,
    swerling_pdf,
    sample_swerling_rcs,
    calculate_pd,
    radar_range_equation,
    compute_radar_snr,
    compute_max_radar_range,
    lidar_range_equation,
    compute_lidar_snr,
    compute_max_lidar_range,
    compute_doppler_shift,
    compute_range_rate_from_doppler,
    compute_radar_ambiguities,
    compute_crlb_uncertainty,
    detection_threshold_pfa,
    ca_cfar_1d,
    ca_cfar_2d,
    os_cfar_1d,
    SpaceborneRadarConfig,
    RadarObservation,
    SpaceborneRadar,
    SpaceborneLidarConfig,
    LidarObservation,
    SpaceborneLidar,
    generate_range_doppler_map,
)


# ============================================================================
# 1. Constants & Enums
# ============================================================================

def test_constants_and_enums():
    assert SPEED_OF_LIGHT == pytest.approx(299792458.0)
    assert BOLTZMANN_CONSTANT == pytest.approx(1.380649e-23)
    assert PLANCK_CONSTANT == pytest.approx(6.62607015e-34)
    assert STANDARD_TEMP_K == 290.0

    assert SwerlingModel.SWERLING_0.value == 0
    assert SwerlingModel.SWERLING_I.value == 1
    assert SwerlingModel.SWERLING_II.value == 2
    assert SwerlingModel.SWERLING_III.value == 3
    assert SwerlingModel.SWERLING_IV.value == 4

    assert RadarBand.X_BAND.value == 10.0e9
    assert RadarBand.KA_BAND.value == 35.0e9
    assert LidarWavelength.NIR_1064.value == 1064e-9


# ============================================================================
# 2. Mathematical Functions: Bessel I0 & Marcum-Q
# ============================================================================

def test_bessel_i0_scaled():
    # At z = 0, exp(0) * I_0(0) = 1.0 * 1.0 = 1.0
    assert bessel_i0_scaled(0.0) == pytest.approx(1.0, rel=1e-6)

    # Strictly positive and monotonically decreasing for z >= 0
    z_vals = np.array([0.0, 1.0, 5.0, 20.0, 50.0])
    vals = bessel_i0_scaled(z_vals)
    assert np.all(vals > 0.0)
    assert np.all(np.diff(vals) < 0.0)

    # Large z asymptote: exp(-z)*I0(z) ~ 1 / sqrt(2 * pi * z)
    z_large = 200.0
    expected_large = 1.0 / math.sqrt(2.0 * math.pi * z_large)
    assert bessel_i0_scaled(z_large) == pytest.approx(expected_large, rel=1e-3)


def test_marcum_q1_edge_cases():
    # When a = 0: Q_1(0, b) = exp(-0.5 * b^2)
    for b in [0.5, 1.0, 2.0, 3.0]:
        assert marcum_q1(0.0, b) == pytest.approx(math.exp(-0.5 * b * b), rel=1e-5)

    # When b = 0: Q_1(a, 0) = 1.0
    for a in [0.0, 1.0, 5.0]:
        assert marcum_q1(a, 0.0) == pytest.approx(1.0, abs=1e-9)

    # Asymptotic regimes:
    # When a - b > 8: Q_1(a, b) -> 1.0
    assert marcum_q1(20.0, 5.0) == pytest.approx(1.0, abs=1e-7)
    # When b - a > 8: Q_1(a, b) -> 0.0
    assert marcum_q1(2.0, 15.0) == pytest.approx(0.0, abs=1e-7)


def test_marcum_q1_monotonicity():
    # Fixed a, increasing b must strictly decrease Q_1(a, b)
    a = 3.0
    b_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    q_vals = [marcum_q1(a, b) for b in b_vals]
    for i in range(len(q_vals) - 1):
        assert q_vals[i] > q_vals[i + 1]


def test_solve_gamma_threshold_pfa():
    # N=1: V_T = -ln(Pfa)
    pfa = 1e-6
    vt_1 = solve_gamma_threshold_pfa(pfa, num_pulses=1)
    assert vt_1 == pytest.approx(-math.log(pfa), rel=1e-9)

    # N > 1: Threshold increases as pulse count increases
    vt_4 = solve_gamma_threshold_pfa(pfa, num_pulses=4)
    vt_16 = solve_gamma_threshold_pfa(pfa, num_pulses=16)
    assert vt_4 > vt_1
    assert vt_16 > vt_4

    # Invalid input checks
    with pytest.raises(ValueError):
        solve_gamma_threshold_pfa(0.0)
    with pytest.raises(ValueError):
        solve_gamma_threshold_pfa(1.5)
    with pytest.raises(ValueError):
        solve_gamma_threshold_pfa(1e-4, num_pulses=0)


# ============================================================================
# 3. Radar Range Equation & SNR Models
# ============================================================================

def test_radar_range_equation_physics():
    # Parameters
    pt = 1000.0       # 1 kW
    gt = 40.0         # 40 dBi
    gr = 40.0         # 40 dBi
    wavelength = 0.03 # 3 cm (X-band 10 GHz)
    rcs = 1.0         # 1 m^2
    r1 = 10000.0      # 10 km
    r2 = 20000.0      # 20 km (doubled)

    p1 = radar_range_equation(pt, gt, gr, wavelength, rcs, r1, system_loss_db=0.0)
    p2 = radar_range_equation(pt, gt, gr, wavelength, rcs, r2, system_loss_db=0.0)

    # 1 / R^4 drop-off: doubling range must decrease power by 16x
    assert p1 / p2 == pytest.approx(16.0, rel=1e-6)

    # Linear scaling with Pt
    p_half_pt = radar_range_equation(pt / 2.0, gt, gr, wavelength, rcs, r1)
    assert p1 / p_half_pt == pytest.approx(2.0, rel=1e-6)

    # Linear scaling with RCS
    p_double_rcs = radar_range_equation(pt, gt, gr, wavelength, rcs * 2.0, r1)
    assert p_double_rcs / p1 == pytest.approx(2.0, rel=1e-6)

    # Gain: +10 dB gain increases received power by 10x
    p_gt_plus10 = radar_range_equation(pt, gt + 10.0, gr, wavelength, rcs, r1)
    assert p_gt_plus10 / p1 == pytest.approx(10.0, rel=1e-6)


def test_radar_range_equation_vectorization_and_errors():
    pt = 1000.0
    gt = gr = 30.0
    wavelength = 0.05
    ranges = np.array([5000.0, 10000.0, 20000.0])
    rcs = 2.0

    powers = radar_range_equation(pt, gt, gr, wavelength, rcs, ranges)
    assert isinstance(powers, np.ndarray)
    assert len(powers) == 3
    assert powers[0] > powers[1] > powers[2]

    # Error conditions
    with pytest.raises(ValueError):
        radar_range_equation(-100.0, gt, gr, wavelength, rcs, 1000.0)
    with pytest.raises(ValueError):
        radar_range_equation(pt, gt, gr, wavelength, rcs, 0.0)
    with pytest.raises(ValueError):
        radar_range_equation(pt, gt, gr, wavelength, -1.0, 1000.0)


def test_compute_radar_snr_coherent_and_incoherent():
    pt = 2000.0
    gt = gr = 42.0
    wavelength = 0.03
    rcs = 0.5
    r = 15000.0
    bw = 5e6
    n_pulses = 16

    # Coherent integration: SNR_int = N * SNR_single (+10 log10(N) dB)
    snr_coh = compute_radar_snr(
        pt, gt, gr, wavelength, rcs, r,
        bandwidth_hz=bw, num_pulses=n_pulses, coherent=True
    )
    expected_gain_db = 10.0 * math.log10(n_pulses)
    assert snr_coh["integrated_snr_db"] - snr_coh["snr_db"] == pytest.approx(expected_gain_db, abs=1e-6)

    # Non-coherent integration: SNR_int < N * SNR_single due to integration loss
    snr_incoh = compute_radar_snr(
        pt, gt, gr, wavelength, rcs, r,
        bandwidth_hz=bw, num_pulses=n_pulses, coherent=False
    )
    assert snr_incoh["integrated_snr_linear"] < snr_coh["integrated_snr_linear"]


def test_compute_max_radar_range_inversion():
    pt = 1500.0
    gt = gr = 40.0
    wavelength = 0.03
    rcs = 1.0
    target_snr_db = 15.0
    bw = 2e6

    r_max = compute_max_radar_range(
        peak_power_w=pt,
        tx_gain_dbi=gt,
        rx_gain_dbi=gr,
        wavelength_m=wavelength,
        rcs_m2=rcs,
        min_snr_db=target_snr_db,
        bandwidth_hz=bw,
    )
    assert r_max > 0.0

    # Verification: evaluate SNR at r_max, should match target_snr_db exactly
    eval_snr = compute_radar_snr(
        peak_power_w=pt,
        tx_gain_dbi=gt,
        rx_gain_dbi=gr,
        wavelength_m=wavelength,
        rcs_m2=rcs,
        range_m=r_max,
        bandwidth_hz=bw,
        num_pulses=1,
    )
    assert eval_snr["snr_db"] == pytest.approx(target_snr_db, abs=1e-4)


# ============================================================================
# 4. Active LIDAR Range Equation & Shot-Noise SNR Models
# ============================================================================

def test_lidar_range_equation_unresolved_and_diffuse():
    ep = 0.02          # 20 mJ
    tau_p = 5.0e-9     # 5 ns
    eta_tx = 0.85
    eta_rx = 0.80
    aperture_diam = 0.25 # 25 cm
    beam_div = 1.0e-3    # 1 mrad
    sigma_opt = 0.05     # 0.05 m^2

    r1 = 1000.0
    r2 = 2000.0

    # Unresolved: 1 / R^4 scaling
    p1 = lidar_range_equation(
        ep, tau_p, eta_tx, eta_rx, aperture_diam, beam_div, sigma_opt, r1, target_type="unresolved"
    )
    p2 = lidar_range_equation(
        ep, tau_p, eta_tx, eta_rx, aperture_diam, beam_div, sigma_opt, r2, target_type="unresolved"
    )
    assert p1 / p2 == pytest.approx(16.0, rel=1e-5)

    # Resolved diffuse: 1 / R^2 scaling
    p_diff_1 = lidar_range_equation(
        ep, tau_p, eta_tx, eta_rx, aperture_diam, beam_div, sigma_opt, r1,
        target_type="resolved_diffuse", diffuse_reflectivity=0.2
    )
    p_diff_2 = lidar_range_equation(
        ep, tau_p, eta_tx, eta_rx, aperture_diam, beam_div, sigma_opt, r2,
        target_type="resolved_diffuse", diffuse_reflectivity=0.2
    )
    assert p_diff_1 / p_diff_2 == pytest.approx(4.0, rel=1e-5)


def test_compute_lidar_snr_and_max_range():
    ep = 0.05
    tau_p = 5.0e-9
    wavelength = 1064e-9
    eta_tx = 0.85
    eta_rx = 0.80
    aperture = 0.20
    beam_div = 0.5e-3
    sigma_opt = 0.1
    r = 2000.0

    snr_res = compute_lidar_snr(
        pulse_energy_j=ep,
        pulse_width_s=tau_p,
        wavelength_m=wavelength,
        optics_tx_eff=eta_tx,
        optics_rx_eff=eta_rx,
        aperture_diam_m=aperture,
        beam_divergence_rad=beam_div,
        target_optical_cross_section_m2=sigma_opt,
        range_m=r,
        num_pulses=4,
    )

    assert snr_res["signal_photoelectrons"] > 0.0
    assert snr_res["background_photoelectrons"] >= 0.0
    assert snr_res["dark_count_photoelectrons"] >= 0.0
    assert snr_res["single_pulse_snr_linear"] > 0.0
    # 4 pulses integration gives 4x linear power SNR
    assert snr_res["integrated_snr_linear"] == pytest.approx(4.0 * snr_res["single_pulse_snr_linear"], rel=1e-6)

    # Invert for max range
    target_snr_db = 10.0
    r_max = compute_max_lidar_range(
        pulse_energy_j=ep,
        pulse_width_s=tau_p,
        wavelength_m=wavelength,
        optics_tx_eff=eta_tx,
        optics_rx_eff=eta_rx,
        aperture_diam_m=aperture,
        beam_divergence_rad=beam_div,
        target_optical_cross_section_m2=sigma_opt,
        min_snr_db=target_snr_db,
    )
    assert r_max > 0.0

    # Verify at r_max
    eval_snr = compute_lidar_snr(
        pulse_energy_j=ep,
        pulse_width_s=tau_p,
        wavelength_m=wavelength,
        optics_tx_eff=eta_tx,
        optics_rx_eff=eta_rx,
        aperture_diam_m=aperture,
        beam_divergence_rad=beam_div,
        target_optical_cross_section_m2=sigma_opt,
        range_m=r_max,
        num_pulses=1,
    )
    assert eval_snr["single_pulse_snr_db"] == pytest.approx(target_snr_db, abs=0.1)


# ============================================================================
# 5. RCS Swerling Target Fluctuation Models (Swerling 0 - IV)
# ============================================================================

def test_swerling_pdf_properties():
    sigma_0 = 2.0

    # Swerling I/II: Exponential PDF p(0) = 1 / sigma_0
    pdf_0 = swerling_pdf(0.0, sigma_0, SwerlingModel.SWERLING_I)
    assert pdf_0 == pytest.approx(1.0 / sigma_0, rel=1e-6)

    # Swerling III/IV: Chi-square 4 DOF p(0) = 0, mode at sigma_0 / 2
    pdf_iii_0 = swerling_pdf(0.0, sigma_0, SwerlingModel.SWERLING_III)
    assert pdf_iii_0 == pytest.approx(0.0, abs=1e-9)

    mode = sigma_0 / 2.0
    pdf_mode = swerling_pdf(mode, sigma_0, SwerlingModel.SWERLING_III)
    # Check that mode has higher density than mode +- 0.5
    assert pdf_mode > swerling_pdf(mode - 0.5, sigma_0, SwerlingModel.SWERLING_III)
    assert pdf_mode > swerling_pdf(mode + 0.5, sigma_0, SwerlingModel.SWERLING_III)


def test_sample_swerling_rcs_fading_and_mean():
    rng = np.random.default_rng(42)
    sigma_mean = 1.5
    n_samples = 50000
    n_pulses = 8

    # Swerling 0: Deterministic
    s0 = sample_swerling_rcs(sigma_mean, SwerlingModel.SWERLING_0, num_samples=10, num_pulses=n_pulses)
    assert np.all(s0 == sigma_mean)

    # Swerling I: Slow fading (constant across pulses in a dwell)
    s1 = sample_swerling_rcs(sigma_mean, SwerlingModel.SWERLING_I, num_samples=n_samples, num_pulses=n_pulses, rng=rng)
    assert np.all(s1[:, 0:1] == s1)  # All columns identical in each row
    assert np.mean(s1) == pytest.approx(sigma_mean, rel=0.03)

    # Swerling II: Fast fading (independent across pulses)
    s2 = sample_swerling_rcs(sigma_mean, SwerlingModel.SWERLING_II, num_samples=n_samples, num_pulses=n_pulses, rng=rng)
    assert not np.all(s2[:, 0:1] == s2)  # Columns differ
    assert np.mean(s2) == pytest.approx(sigma_mean, rel=0.03)

    # Swerling III: Slow fading
    s3 = sample_swerling_rcs(sigma_mean, SwerlingModel.SWERLING_III, num_samples=n_samples, num_pulses=n_pulses, rng=rng)
    assert np.all(s3[:, 0:1] == s3)
    assert np.mean(s3) == pytest.approx(sigma_mean, rel=0.03)

    # Swerling IV: Fast fading
    s4 = sample_swerling_rcs(sigma_mean, SwerlingModel.SWERLING_IV, num_samples=n_samples, num_pulses=n_pulses, rng=rng)
    assert not np.all(s4[:, 0:1] == s4)
    assert np.mean(s4) == pytest.approx(sigma_mean, rel=0.03)


def test_calculate_pd_swerling_models():
    pfa = 1e-6

    # Test Swerling I: Exact closed-form Pd = Pfa^(1 / (1 + SNR))
    for snr_db in [0.0, 10.0, 20.0]:
        snr_lin = 10.0 ** (snr_db / 10.0)
        expected_pd1 = math.pow(pfa, 1.0 / (1.0 + snr_lin))
        calc_pd1 = calculate_pd(snr_lin, pfa, SwerlingModel.SWERLING_I)
        assert calc_pd1 == pytest.approx(expected_pd1, rel=1e-5)

    # At N=1, Swerling II equals Swerling I
    for snr_lin in [1.0, 10.0, 50.0]:
        pd1 = calculate_pd(snr_lin, pfa, SwerlingModel.SWERLING_I, num_pulses=1)
        pd2 = calculate_pd(snr_lin, pfa, SwerlingModel.SWERLING_II, num_pulses=1)
        assert pd1 == pytest.approx(pd2, rel=1e-5)

    # For N > 1, Swerling II achieves higher Pd due to fluctuation diversity
    pd2_n4 = calculate_pd(10.0, pfa, SwerlingModel.SWERLING_II, num_pulses=4)
    pd2_n1 = calculate_pd(10.0, pfa, SwerlingModel.SWERLING_II, num_pulses=1)
    assert pd2_n4 > pd2_n1

    # Monotonicity with SNR
    snrs = [0.1, 1.0, 10.0, 100.0]
    for model in SwerlingModel:
        pds = [calculate_pd(s, pfa, model) for s in snrs]
        for i in range(len(pds) - 1):
            assert pds[i] < pds[i + 1]

    # Asymptotic limits:
    # SNR -> 0: Pd -> Pfa
    for model in SwerlingModel:
        assert calculate_pd(1e-6, pfa, model) == pytest.approx(pfa, rel=0.05)
    # SNR -> inf: Pd -> 1.0
    for model in SwerlingModel:
        assert calculate_pd(1e7, pfa, model) == pytest.approx(1.0, abs=1e-4)


# ============================================================================
# 6. Range-Rate Doppler Shifts & Ambiguities
# ============================================================================

def test_doppler_shift_and_round_trip_inversion():
    carrier_freq = 10.0e9 # 10 GHz X-band
    wavelength = SPEED_OF_LIGHT / carrier_freq # 0.03 m

    # Approaching target (closing): range_rate = -500 m/s
    # f_D = -2 * dR/dt / lambda = -2 * (-500) / 0.03 = +33333.33 Hz
    rr_closing = -500.0
    fd_closing = compute_doppler_shift(carrier_freq, rr_closing)
    assert fd_closing > 0.0
    assert fd_closing == pytest.approx(33355.608, rel=1e-3)

    # Receding target (opening): range_rate = +1200 m/s => negative Doppler
    rr_opening = 1200.0
    fd_opening = compute_doppler_shift(carrier_freq, rr_opening)
    assert fd_opening < 0.0

    # Exact round-trip inversion
    for rr in [-15000.0, -250.0, 0.0, 750.0, 8000.0]:
        fd = compute_doppler_shift(carrier_freq, rr)
        rr_recovered = compute_range_rate_from_doppler(carrier_freq, fd)
        assert rr_recovered == pytest.approx(rr, abs=1e-9)

    # Relativistic Doppler round-trip
    for rr in [-15000.0, 7500.0]: # orbital speeds up to 15 km/s
        fd_rel = compute_doppler_shift(carrier_freq, rr, relativistic=True)
        rr_rel_rec = compute_range_rate_from_doppler(carrier_freq, fd_rel, relativistic=True)
        assert rr_rel_rec == pytest.approx(rr, rel=1e-6)


def test_radar_ambiguities_and_crlb():
    prf = 4000.0 # 4 kHz
    fc = 10.0e9  # 10 GHz (lambda = 0.03 m)

    amb = compute_radar_ambiguities(prf_hz=prf, carrier_freq_hz=fc)
    # R_ua = c / (2 * PRF) = 3e8 / 8000 = 37.5 km
    assert amb["unambiguous_range_km"] == pytest.approx(37.474, rel=1e-3)
    # v_ua = lambda * PRF / 4 = 0.03 * 4000 / 4 = 30 m/s
    assert amb["unambiguous_velocity_m_s"] == pytest.approx(29.979, rel=1e-3)
    # Doppler dilemma product = c * lambda / 8
    assert amb["doppler_dilemma_product"] == pytest.approx(amb["unambiguous_range_m"] * amb["unambiguous_velocity_m_s"], rel=1e-6)

    # CRLB Uncertainty
    crlb = compute_crlb_uncertainty(
        snr_linear=100.0, # 20 dB SNR
        chirp_bandwidth_hz=100.0e6, # 100 MHz
        obs_time_s=0.05, # 50 ms
        carrier_freq_hz=fc,
    )
    # Range error sigma_R is on the order of a few centimeters
    assert crlb["sigma_range_m"] < 0.20
    # Velocity error sigma_v is on the order of cm/s
    assert crlb["sigma_velocity_m_s"] < 0.10
    # Range resolution delta_R = c / (2 * B) = 3e8 / 2e8 = 1.5 m
    assert crlb["range_resolution_m"] == pytest.approx(1.4989, rel=1e-3)


# ============================================================================
# 7. Detection Thresholding & CFAR Detectors
# ============================================================================

def test_detection_threshold_pfa():
    pfa = 1e-4
    pn = 2.0e-14
    vt = detection_threshold_pfa(pfa, noise_power_w=pn)
    assert vt == pytest.approx(-pn * math.log(pfa), rel=1e-9)


def test_ca_cfar_1d_target_detection():
    rng = np.random.default_rng(123)
    n_cells = 128
    noise_floor = 1.0

    # Exponential noise background
    signal = rng.exponential(scale=noise_floor, size=n_cells)
    target_idx = 64
    signal[target_idx] += 35.0 # Strong injected target

    cfar = ca_cfar_1d(signal, num_guard_cells=2, num_ref_cells=16, pfa=1e-4)

    # Target cell must be detected
    assert cfar["detections"][target_idx] is True or cfar["detections"][target_idx] == True
    # Most noise cells should not be detected
    false_alarms = np.sum(cfar["detections"]) - 1
    assert false_alarms <= 2


def test_ca_cfar_2d_range_doppler():
    rng = np.random.default_rng(456)
    rows, cols = 40, 40
    mat = rng.exponential(scale=1.0, size=(rows, cols))

    # Inject target at (20, 20)
    mat[20, 20] += 50.0

    cfar2d = ca_cfar_2d(mat, guard_cells=(2, 2), ref_cells=(4, 4), pfa=1e-4)
    assert cfar2d["detections_map"][20, 20] is True or cfar2d["detections_map"][20, 20] == True
    assert np.sum(cfar2d["detections_map"]) <= 3


def test_os_cfar_1d_multiple_targets():
    rng = np.random.default_rng(789)
    n_cells = 100
    signal = rng.exponential(scale=1.0, size=n_cells)

    # Inject two closely spaced targets at bin 45 and 48
    signal[45] += 40.0
    signal[48] += 40.0

    os_res = os_cfar_1d(signal, num_guard_cells=1, num_ref_cells=12, pfa=1e-3, k_rank=18)
    assert os_res["detections"][45] == True
    assert os_res["detections"][48] == True


# ============================================================================
# 8. High-Level Spaceborne Radar and LIDAR Observation Engines
# ============================================================================

def test_spaceborne_radar_observation():
    config = SpaceborneRadarConfig(
        frequency_hz=10.0e9,
        peak_power_w=2000.0,
        tx_gain_dbi=42.0,
        rx_gain_dbi=42.0,
        num_integrated_pulses=16,
    )
    radar = SpaceborneRadar(config)

    # Host satellite at 7000 km altitude along X
    radar_pos = [7000.0, 0.0, 0.0]
    radar_vel = [0.0, 7.546, 0.0]

    # Target debris at 7010 km along X, slightly closing along Y
    target_pos = [7010.0, 2.0, 0.0]
    target_vel = [0.0, 7.500, 0.0]

    obs = radar.observe(
        target_pos_km=target_pos,
        target_vel_km=target_vel,
        radar_pos_km=radar_pos,
        radar_vel_km=radar_vel,
        mean_rcs_m2=2.0,
        swerling_model=SwerlingModel.SWERLING_I,
        rng=np.random.default_rng(100),
    )

    # True range: sqrt(10^2 + 2^2) = sqrt(104) = 10.198 km = 10198 m
    assert obs.true_range_m == pytest.approx(10198.039, rel=1e-3)
    assert obs.true_range_rate_m_s != 0.0
    assert obs.integrated_snr_db > 10.0
    assert obs.detection_probability > 0.5
    assert obs.detected is True

    # Measured range within 3-sigma of CRLB
    assert obs.measured_range_m is not None
    assert abs(obs.measured_range_m - obs.true_range_m) < 3.0 * obs.sigma_range_m
    assert abs(obs.measured_range_rate_m_s - obs.true_range_rate_m_s) < 3.0 * obs.sigma_velocity_m_s


def test_spaceborne_lidar_observation():
    config = SpaceborneLidarConfig(
        pulse_energy_j=0.05,
        aperture_diam_m=0.20,
        num_integrated_pulses=10,
    )
    lidar = SpaceborneLidar(config)

    # Target at 500 meters range
    target_pos = [7000.0, 0.500, 0.0]
    target_vel = [0.0, 7.540, 0.0]
    lidar_pos = [7000.0, 0.0, 0.0]
    lidar_vel = [0.0, 7.546, 0.0]

    obs = lidar.observe(
        target_pos_km=target_pos,
        target_vel_km=target_vel,
        lidar_pos_km=lidar_pos,
        lidar_vel_km=lidar_vel,
        optical_cross_section_m2=0.05,
        rng=np.random.default_rng(200),
    )

    assert obs.true_range_m == pytest.approx(500.0, rel=1e-4)
    assert obs.signal_photoelectrons > 10.0
    assert obs.optical_snr_db > 20.0
    assert obs.detected is True
    assert obs.measured_range_m is not None
    assert abs(obs.measured_range_m - obs.true_range_m) < 3.0 * obs.sigma_range_m


def test_generate_range_doppler_map():
    config = SpaceborneRadarConfig(
        frequency_hz=10.0e9,
        chirp_bandwidth_hz=50.0e6,
        prf_hz=2000.0,
        num_integrated_pulses=16,
    )

    range_bins = np.linspace(5000.0, 15000.0, 64)
    velocity_bins = np.linspace(-100.0, 100.0, 64)

    targets = [
        {"range_m": 8000.0, "velocity_m_s": -20.0, "rcs_m2": 5.0},
        {"range_m": 12000.0, "velocity_m_s": 35.0, "rcs_m2": 10.0},
    ]

    p_map, r_axis, v_axis = generate_range_doppler_map(
        targets=targets,
        radar_config=config,
        range_bins_m=range_bins,
        velocity_bins_m_s=velocity_bins,
        rng=np.random.default_rng(300),
    )

    assert p_map.shape == (64, 64)
    assert np.all(p_map >= 0.0)

    # Check that peaks exist near target coordinates
    idx_r1 = np.argmin(np.abs(range_bins - 8000.0))
    idx_v1 = np.argmin(np.abs(velocity_bins - (-20.0)))
    # Local power at target should be significantly above background mean
    mean_power = np.mean(p_map)
    assert p_map[idx_r1, idx_v1] > mean_power * 2.0
