"""
Unit Tests for Adaptive Drag Make-Up (ADMU) Controller Engine
Tests L1 Adaptive Control, Aerospace PID with Anti-Windup (Clamping & Back-Calculation),
Atmospheric Waves, Geomagnetic Storm Density Surges, and Closed-Loop VLEO Flight.
"""

import math
import pytest
import numpy as np

from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    STANDARD_GRAVITY,
    QUIET_KP,
    QUIET_F10_7,
    STORM_G5_KP,
)
from aegis_leo.ai.drag_makeup import (
    ADMUControlMode,
    AntiWindupMethod,
    FilterType,
    VLEOSpacecraft,
    VLEOThruster,
    DisturbanceConfig,
    ADMUStepTelemetry,
    ADMUPerformanceReport,
    L1AdaptiveController,
    PIDWithAntiWindup,
    HybridL1PIDController,
    AtmosphericDisturbanceSynthesizer,
    ThrusterActuatorEmulator,
    ADMUSimulator,
    compare_admu_controllers,
    tune_l1_drag_controller,
)


# ==============================================================================
# 1. SPACECRAFT & THRUSTER HARDWARE PRESETS
# ==============================================================================

def test_vleo_spacecraft_presets_and_ballistic_coefficient():
    """Verify spacecraft aerodynamic properties and presets."""
    goce = VLEOSpacecraft.goce_like()
    assert goce.mass_kg == 1050.0
    assert goce.cross_section_area_m2 == 1.0
    assert goce.drag_coefficient == 2.1
    # BC = m / (Cd * A) = 1050 / (2.1 * 1.0) = 500.0 kg/m^2
    assert pytest.approx(goce.ballistic_coefficient, rel=1e-3) == 500.0
    assert pytest.approx(goce.area_to_mass_ratio, rel=1e-3) == 1.0 / 1050.0

    skimsat = VLEOSpacecraft.skimsat_vleo()
    assert skimsat.mass_kg == 180.0
    assert 170.0 < skimsat.ballistic_coefficient < 190.0

    slats = VLEOSpacecraft.tsubame_slats()
    assert slats.nominal_altitude_km == 200.0
    assert slats.mass_kg == 400.0

    starlink = VLEOSpacecraft.starlink_vleo()
    assert starlink.nominal_altitude_km == 280.0


def test_vleo_thruster_hardware_and_propellant_physics():
    """Verify electric thruster specs, propellant depletion, and air-breathing ABEP mode."""
    qit = VLEOThruster.goce_qit()
    assert qit.thrust_min_n == 0.001
    assert qit.thrust_max_n == 0.021
    assert qit.isp_s == 3100.0
    # Mass flow rate at 20 mN: m_dot = 0.020 / (3100 * 9.80665)
    expected_mdot = 0.020 / (3100.0 * STANDARD_GRAVITY)
    assert pytest.approx(qit.calculate_propellant_rate_kg_s(0.020), rel=1e-4) == expected_mdot

    # Power consumption: idle + thrust * power_per_thrust
    power_at_20mn = qit.calculate_power_consumption_w(0.020)
    assert power_at_20mn == pytest.approx(80.0 + 0.020 * 22000.0, rel=1e-3)

    # Air-Breathing Electric Propulsion (ABEP) consumes zero onboard tank propellant
    abep = VLEOThruster.abep_ram_ep()
    assert abep.is_air_breathing is True
    assert abep.calculate_propellant_rate_kg_s(0.050) == 0.0


# ==============================================================================
# 2. ATMOSPHERIC DISTURBANCE SYNTHESIZER
# ==============================================================================

def test_atmospheric_disturbance_synthesizer_waves_and_storm():
    """Verify atmospheric gravity waves and CME storm surge generation."""
    config = DisturbanceConfig(
        enable_waves=True,
        gravity_waves=[(0.20, 600.0, 0.0)],  # 20% wave, 10 min period
        enable_storm=True,
        storm_start_time_s=1000.0,
        storm_rise_time_s=200.0,
        storm_peak_multiplier=4.0,           # 4x storm surge
        storm_decay_time_s=1000.0,
        enable_diurnal=False,
    )
    synth = AtmosphericDisturbanceSynthesizer(config)

    # 1. Base density decreases with altitude
    rho_200, _ = synth.compute_density(200.0, time_s=0.0)
    rho_250, _ = synth.compute_density(250.0, time_s=0.0)
    rho_300, _ = synth.compute_density(300.0, time_s=0.0)
    assert rho_200 > rho_250 > rho_300 > 0.0

    # 2. Gravity Wave oscillation test
    # At t=0: sin(0) = 0 -> multiplier = 1.0
    # At t=150 (quarter period): sin(pi/2) = 1 -> multiplier = 1.20
    # At t=450 (three-quarters period): sin(3*pi/2) = -1 -> multiplier = 0.80
    _, diag_0 = synth.compute_density(250.0, time_s=0.0)
    _, diag_150 = synth.compute_density(250.0, time_s=150.0)
    _, diag_450 = synth.compute_density(250.0, time_s=450.0)
    assert pytest.approx(diag_0["wave_multiplier"], abs=0.03) == 1.0
    assert pytest.approx(diag_150["wave_multiplier"], abs=0.03) == 1.20
    assert pytest.approx(diag_450["wave_multiplier"], abs=0.03) == 0.80

    # 3. Geomagnetic Storm Surge test
    # Before storm onset (t=500 s < 1000 s): storm multiplier is 1.0
    _, diag_pre = synth.compute_density(250.0, time_s=500.0)
    assert diag_pre["storm_multiplier"] == 1.0

    # At peak of storm (t=1200 s): multiplier surges up towards 4.0
    _, diag_peak = synth.compute_density(250.0, time_s=1200.0)
    assert diag_peak["storm_multiplier"] > 3.0


# ==============================================================================
# 3. PID WITH ANTI-WINDUP (CLAMPING & BACK-CALCULATION)
# ==============================================================================

def test_pid_anti_windup_clamping_prevents_integrator_runaway():
    """Verify conditional integration (clamping) freezes the integrator during saturation."""
    # Saturated PID: output_max = 1.0
    pid = PIDWithAntiWindup(
        kp=1.0,
        ki=2.0,
        kd=0.1,
        anti_windup_method=AntiWindupMethod.CLAMPING,
        output_min=0.0,
        output_max=1.0,
    )

    # Large positive error for 10 seconds: would accumulate 2.0 * 10 * 10 = 200 without anti-windup
    for _ in range(20):
        out, diag = pid.step(dt=0.5, error=10.0)
        assert out == 1.0
        assert diag["is_saturated"] is True

    # Integrator should NOT be wound up to 100+
    assert pid.integral_state < 2.0

    # Instant error sign reversal: error becomes negative (-2.0)
    # With clamping, controller recovers immediately without waiting for windup unwinding
    out_rev, _ = pid.step(dt=0.5, error=-2.0)
    assert out_rev < 1.0


def test_pid_anti_windup_back_calculation():
    """Verify back-calculation tracking discharges accumulated integrator error."""
    pid = PIDWithAntiWindup(
        kp=1.0,
        ki=1.5,
        kd=0.05,
        anti_windup_method=AntiWindupMethod.BACK_CALCULATION,
        back_calc_tracking_gain=1.5,
        output_min=0.0,
        output_max=1.0,
    )

    # Drive into saturation
    for _ in range(15):
        pid.step(dt=0.2, error=5.0)

    # Integrator remains bounded by tracking feedback
    assert pid.integral_state < 3.0

    # Test error reversal recovery
    out, _ = pid.step(dt=0.2, error=-2.0)
    assert out == 0.0  # Output clamps to min immediately


def test_pid_derivative_filtering_and_derivative_on_measurement():
    """Verify filtered derivative on measurement prevents setpoint derivative kick."""
    pid = PIDWithAntiWindup(
        kp=1.0,
        ki=0.0,
        kd=5.0,
        derivative_on_measurement=True,
        derivative_filter_tau_s=0.10,
        output_min=-100.0,
        output_max=100.0,
    )

    # Constant measurement, step change in setpoint/error
    # Derivative on measurement should remain 0, preventing derivative kick
    out1, diag1 = pid.step(dt=0.1, error=10.0, measurement=100.0)
    out2, diag2 = pid.step(dt=0.1, error=20.0, measurement=100.0)  # error jumped, measurement didn't
    assert pytest.approx(diag2["d_term"], abs=1e-5) == 0.0


# ==============================================================================
# 4. L1 ADAPTIVE CONTROLLER ARCHITECTURE
# ==============================================================================

def test_l1_adaptive_controller_initialization_and_validation():
    """Verify parameter bounds, Hurwitz reference pole check, and filter setup."""
    # Reference pole must be negative
    with pytest.raises(ValueError):
        L1AdaptiveController(am=0.5)

    with pytest.raises(ValueError):
        L1AdaptiveController(gamma=-10.0)

    with pytest.raises(ValueError):
        L1AdaptiveController(cutoff_freq_hz=-1.0)

    l1 = L1AdaptiveController(
        am=-0.5,
        gamma=1000.0,
        cutoff_freq_hz=0.30,
        filter_type=FilterType.SECOND_ORDER_BUTTERWORTH,
    )
    assert l1.p_lyap > 0.0
    assert l1.omega_c == pytest.approx(2.0 * math.pi * 0.30, rel=1e-3)


def test_l1_adaptive_disturbance_rejection_step():
    """Verify fast adaptation law accurately estimates unmodeled drag step disturbance."""
    l1 = L1AdaptiveController(
        am=-0.6,
        gamma=1200.0,
        cutoff_freq_hz=0.40,
        sigma_min=-0.05,
        sigma_max=0.05,
        k_base=0.10,
    )

    # Injected constant unmodeled drag disturbance: d = -0.006 m/s^2
    true_disturbance = -0.006
    e = 0.0
    u_applied = 0.0

    for _ in range(80):
        u_cmd, diag = l1.step(
            dt=0.05,
            tracking_error=e,
            applied_accel_cmd=u_applied,
            feedforward_accel=0.0,
        )
        u_applied = u_cmd
        # Plant update: e_dot = u + d
        e += (u_applied + true_disturbance) * 0.05

    # L1 adaptive estimate should closely track true disturbance
    assert pytest.approx(diag["sigma_hat"], rel=0.15) == true_disturbance
    # Adaptive cancellation term u_ad = - filtered_sigma should be positive
    assert diag["u_ad"] > 0.0
    # Final tracking error should be driven close to zero (< 0.0005 m/s)
    assert abs(e) < 5e-4


def test_l1_projection_operator():
    """Verify projection operator strictly bounds parameter estimate within [sigma_min, sigma_max]."""
    l1 = L1AdaptiveController(
        sigma_min=-0.010,
        sigma_max=0.010,
    )
    # Moving outwards past boundary -> projected to 0
    assert l1._projection(0.010, 5.0, -0.010, 0.010) == 0.0
    assert l1._projection(-0.010, -5.0, -0.010, 0.010) == 0.0
    # Moving inwards from boundary -> allowed through
    assert l1._projection(0.010, -5.0, -0.010, 0.010) == -5.0
    assert l1._projection(-0.010, 5.0, -0.010, 0.010) == 5.0


def test_l1_butterworth_lowpass_filter():
    """Verify 2nd-order Butterworth low-pass filter suppresses high-frequency noise."""
    l1 = L1AdaptiveController(
        cutoff_freq_hz=0.10,  # 0.1 Hz cutoff (period = 10s)
        filter_type=FilterType.SECOND_ORDER_BUTTERWORTH,
    )

    # High frequency oscillation: 10 Hz sinusoid (100x above cutoff)
    dt = 0.01
    filtered_outputs = []
    for k in range(200):
        t = k * dt
        hf_input = math.sin(2.0 * math.pi * 10.0 * t)
        out = l1.update_filter(hf_input, dt)
        filtered_outputs.append(out)

    # High frequency signal amplitude should be heavily attenuated (> 80% reduction)
    peak_out = max(abs(x) for x in filtered_outputs[50:])
    assert peak_out < 0.20


# ==============================================================================
# 5. ACTUATOR & HARDWARE EMULATION
# ==============================================================================

def test_thruster_actuator_slew_rate_and_quantization():
    """Verify thruster actuator respects slew rate limits and digital throttle steps."""
    thruster = VLEOThruster(
        name="Test Thruster",
        thrust_min_n=0.005,
        thrust_max_n=0.050,
        slew_rate_limit_n_s=0.010,  # 10 mN/s max slew
        isp_s=2000.0,
        throttle_step_n=0.001,       # 1 mN discrete steps
        time_constant_s=0.01,
    )
    actuator = ThrusterActuatorEmulator(thruster)

    # Instant step command from 0 to 50 mN with dt=1.0 s
    # Slew limit is 10 mN/s, so after 1.0 s thrust cannot exceed ~10 mN
    actual_t, _ = actuator.step(dt=1.0, commanded_thrust_n=0.050)
    assert actual_t <= 0.015

    # Check zero thrust when command is 0
    actuator.reset()
    actual_0, is_sat = actuator.step(dt=1.0, commanded_thrust_n=0.0)
    assert actual_0 == 0.0
    assert is_sat is False


# ==============================================================================
# 6. CLOSED-LOOP VLEO SIMULATION & BENCHMARKS
# ==============================================================================

def test_uncontrolled_ballistic_orbital_decay():
    """Verify uncontrolled spacecraft experiences orbital decay under atmospheric drag."""
    sc = VLEOSpacecraft.skimsat_vleo()
    th = VLEOThruster.starlink_krypton()
    dc = DisturbanceConfig(enable_waves=False, enable_storm=False, enable_diurnal=False)

    sim = ADMUSimulator(sc, th, dc, ADMUControlMode.UNCONTROLLED)
    report, _ = sim.run_simulation(duration_s=600.0, dt=1.0)

    # Ballistic flight has 0% drag compensation and positive altitude loss
    assert report.drag_compensation_efficiency_pct == 0.0
    assert report.total_thrust_impulse_n_s == 0.0
    assert report.altitude_loss_m > 0.0
    assert report.final_altitude_km < report.initial_altitude_km


def test_feedforward_only_drag_compensation():
    """Verify feedforward-only control provides baseline compensation but drifts under disturbances."""
    sc = VLEOSpacecraft.skimsat_vleo()
    th = VLEOThruster.starlink_krypton()
    dc = DisturbanceConfig(enable_waves=True, enable_storm=False)

    sim = ADMUSimulator(sc, th, dc, ADMUControlMode.FEEDFORWARD_ONLY)
    report, _ = sim.run_simulation(duration_s=600.0, dt=1.0)

    # Partial drag compensation (quiet density baseline vs true density with waves)
    assert 60.0 < report.drag_compensation_efficiency_pct < 120.0
    assert report.total_thrust_impulse_n_s > 0.0


def test_pid_anti_windup_closed_loop_tracking():
    """Verify PID with anti-windup maintains velocity and near-zero altitude deviation."""
    sc = VLEOSpacecraft.skimsat_vleo()
    th = VLEOThruster.starlink_krypton()
    dc = DisturbanceConfig(enable_waves=True, enable_storm=False)

    sim = ADMUSimulator(sc, th, dc, ADMUControlMode.PID_ANTI_WINDUP)
    report, _ = sim.run_simulation(duration_s=600.0, dt=1.0)

    assert report.rms_velocity_error_m_s < 0.005
    assert abs(report.altitude_loss_m) < 0.50
    assert 98.0 < report.drag_compensation_efficiency_pct < 102.0


def test_l1_adaptive_wave_disturbance_tracking():
    """Verify L1 adaptive control achieves >99% drag compensation and minimal chattering."""
    sc = VLEOSpacecraft.skimsat_vleo()
    th = VLEOThruster.starlink_krypton()
    dc = DisturbanceConfig(
        enable_waves=True,
        gravity_waves=[(0.25, 400.0, 0.0)],  # Severe 25% gravity wave
        enable_storm=False,
    )

    sim = ADMUSimulator(sc, th, dc, ADMUControlMode.L1_ADAPTIVE)
    report, telem = sim.run_simulation(duration_s=800.0, dt=0.5)

    assert report.rms_velocity_error_m_s < 0.001
    assert abs(report.altitude_loss_m) < 0.10
    assert 99.0 < report.drag_compensation_efficiency_pct < 101.5
    # Chattering index should be small (smooth continuous thrust)
    assert report.chattering_index_n_s < 0.001
    assert len(telem) > 1000


def test_hybrid_l1_pid_dual_loop_performance():
    """Verify Hybrid L1 + PID dual loop controller combines fast disturbance rejection with tracking."""
    sc = VLEOSpacecraft.skimsat_vleo()
    th = VLEOThruster.starlink_krypton()
    dc = DisturbanceConfig(enable_waves=True, enable_storm=False)

    sim = ADMUSimulator(sc, th, dc, ADMUControlMode.HYBRID_L1_PID)
    report, _ = sim.run_simulation(duration_s=600.0, dt=1.0)

    assert report.rms_velocity_error_m_s < 0.002
    assert abs(report.altitude_loss_m) < 0.20
    assert 98.5 < report.drag_compensation_efficiency_pct < 101.5


def test_geomagnetic_storm_surge_survival():
    """Verify satellite survives severe CME storm surge (3.5x density spike) with active drag make-up."""
    sc = VLEOSpacecraft.skimsat_vleo()
    # High-capacity thruster capable of handling peak storm drag
    th = VLEOThruster.high_capacity_vleo()
    dc = DisturbanceConfig(
        enable_waves=False,
        enable_storm=True,
        storm_start_time_s=300.0,
        storm_rise_time_s=150.0,
        storm_peak_multiplier=3.5,
        storm_decay_time_s=600.0,
    )

    sim = ADMUSimulator(sc, th, dc, ADMUControlMode.L1_ADAPTIVE)
    report, telem = sim.run_simulation(duration_s=1200.0, dt=1.0)

    # Spacecraft must survive the storm
    assert report.survived_storm is True
    assert report.min_altitude_km > 200.0
    assert report.rms_velocity_error_m_s < 0.005
    # Verify density surge occurred in telemetry
    peak_density = max(t.true_density_kg_m3 for t in telem)
    base_density = telem[0].true_density_kg_m3
    assert peak_density > 2.5 * base_density


def test_compare_admu_controllers_benchmark_suite():
    """Verify multi-controller comparison suite produces valid comparative reports."""
    results = compare_admu_controllers(duration_s=400.0, dt=1.0)
    assert ADMUControlMode.UNCONTROLLED.value in results
    assert ADMUControlMode.FEEDFORWARD_ONLY.value in results
    assert ADMUControlMode.PID_ANTI_WINDUP.value in results
    assert ADMUControlMode.L1_ADAPTIVE.value in results
    assert ADMUControlMode.HYBRID_L1_PID.value in results

    uncontrolled_loss = results[ADMUControlMode.UNCONTROLLED.value].altitude_loss_m
    l1_loss = abs(results[ADMUControlMode.L1_ADAPTIVE.value].altitude_loss_m)
    # L1 adaptive altitude loss must be dramatically smaller than uncontrolled decay
    assert uncontrolled_loss > l1_loss * 5.0

    # Serialization test
    d = results[ADMUControlMode.L1_ADAPTIVE.value].to_dict()
    assert isinstance(d, dict)
    assert d["controller_type"] == ADMUControlMode.L1_ADAPTIVE.value


def test_tune_l1_drag_controller_helper():
    """Verify analytical L1 controller tuning helper."""
    sc = VLEOSpacecraft.goce_like()
    tuned_l1 = tune_l1_drag_controller(sc, actuator_bandwidth_hz=0.6)
    assert tuned_l1.cutoff_freq_hz == pytest.approx(0.3, rel=1e-3)
    assert tuned_l1.am == -0.4
    assert tuned_l1.gamma == 1000.0
    assert tuned_l1.filter_type == FilterType.SECOND_ORDER_BUTTERWORTH
