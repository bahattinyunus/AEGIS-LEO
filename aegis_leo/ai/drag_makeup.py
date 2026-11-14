"""
AEGIS-LEO: Adaptive Drag Make-Up (ADMU) Controller Engine
Very Low Earth Orbit (VLEO: 180-300 km) & LEO Atmospheric Compensation System

Implements:
1. L1 Adaptive Control Architecture:
   - Fast state predictor / estimator for matched drag disturbance estimation
   - Projection-based high-gain adaptation law ensuring parameter boundedness
   - Low-pass control filter (1st and 2nd-order Butterworth) decoupling fast adaptation
     from control bandwidth to guarantee robustness and eliminate actuator chattering.
2. Aerospace PID Controller with Anti-Windup:
   - Conditional integration (clamping) and back-calculation tracking anti-windup
   - Filtered derivative on measurement to eliminate setpoint derivative kick
   - Dynamic thruster output saturation and slew rate limits.
3. Hybrid Dual-Loop Architecture:
   - Fast inner-loop L1 adaptive disturbance rejection coupled with outer-loop PID
     orbital altitude / velocity stationkeeping.
4. Dynamic Atmospheric Disturbance & Weather Synthesizer:
   - Calibrated VLEO base density tables (180 - 300 km)
   - Multi-harmonic atmospheric gravity waves (periods 5 to 60 min, 5% to 40% amplitude)
   - Geomagnetic storm surges (CME / solar flares, Kp 5 to 9, 2x to 8x density rise)
   - Diurnal solar thermal bulge and stochastic wind/density turbulence.
5. High-Precision 2-DOF / Polar Orbit Dynamics Simulator:
   - Coupled radial / along-track non-linear equations of motion with Earth gravity
   - Electric thruster hardware emulation (GOCE QIT, Hall thruster, ABEP air-breathing)
   - Real-time drag compensation efficiency, chattering index, and storm survival verification.

Adheres strictly to physics models and uses ONLY NumPy and the Python standard library.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import math
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import numpy as np

from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    EARTH_RADIUS_M,
    EARTH_MU,
    EARTH_MU_SI,
    STANDARD_GRAVITY,
    QUIET_KP,
    STORM_G5_KP,
    QUIET_F10_7,
    EXTREME_F10_7,
)
from aegis_leo.physics.atmosphere import compute_atmospheric_density


# ==============================================================================
# ENUMS & CONSTANTS
# ==============================================================================

class ADMUControlMode(str, Enum):
    """Control architecture operational modes for drag compensation."""
    L1_ADAPTIVE = "L1_ADAPTIVE"
    PID_ANTI_WINDUP = "PID_ANTI_WINDUP"
    HYBRID_L1_PID = "HYBRID_L1_PID"
    FEEDFORWARD_ONLY = "FEEDFORWARD_ONLY"
    UNCONTROLLED = "UNCONTROLLED"


class AntiWindupMethod(str, Enum):
    """Anti-windup reset / compensation mechanisms for PID integral channel."""
    CLAMPING = "CLAMPING"                  # Conditional integration: freeze when saturated
    BACK_CALCULATION = "BACK_CALCULATION"  # Tracking feedback: dI/dt = Ki*e + Kb*(u_sat - u_unsat)
    INTEGRAL_SATURATION = "INTEGRAL_SATURATION" # Hard clamp on integrated error state


class FilterType(str, Enum):
    """Filter architecture for L1 adaptive control low-pass filter."""
    FIRST_ORDER = "FIRST_ORDER"
    SECOND_ORDER_BUTTERWORTH = "SECOND_ORDER_BUTTERWORTH"


class DisturbanceType(str, Enum):
    """Atmospheric disturbance phenomena in VLEO."""
    GRAVITY_WAVE = "GRAVITY_WAVE"
    STORM_SURGE = "STORM_SURGE"
    DIURNAL_BULGE = "DIURNAL_BULGE"
    STOCHASTIC_TURBULENCE = "STOCHASTIC_TURBULENCE"
    COMBINED = "COMBINED"


# ==============================================================================
# DATA STRUCTURES & HARDWARE PRESETS
# ==============================================================================

@dataclass
class VLEOSpacecraft:
    """
    Spacecraft mass, aerodynamic geometry, and nominal orbital configuration.
    """
    name: str
    mass_kg: float
    cross_section_area_m2: float
    drag_coefficient: float
    nominal_altitude_km: float
    inclination_deg: float = 96.5  # Sun-synchronous orbit standard
    cd_uncertainty_fraction: float = 0.15  # 15% physical unmodeled drag uncertainty

    @property
    def ballistic_coefficient(self) -> float:
        """Ballistic coefficient BC = m / (Cd * A) [kg/m^2]."""
        denom = self.drag_coefficient * self.cross_section_area_m2
        return self.mass_kg / denom if denom > 0 else float("inf")

    @property
    def area_to_mass_ratio(self) -> float:
        """Area to mass ratio A / m [m^2 / kg]."""
        return self.cross_section_area_m2 / self.mass_kg if self.mass_kg > 0 else 0.0

    @classmethod
    def goce_like(cls) -> "VLEOSpacecraft":
        """
        ESA GOCE-like ultra-streamlined arrow configuration (250 km VLEO).
        Fin-stabilized slender octagonal body minimizing cross-sectional ram area.
        """
        return cls(
            name="GOCE-Like Aerodynamic Explorer",
            mass_kg=1050.0,
            cross_section_area_m2=1.00,
            drag_coefficient=2.1,
            nominal_altitude_km=250.0,
            inclination_deg=96.7,
        )

    @classmethod
    def skimsat_vleo(cls) -> "VLEOSpacecraft":
        """
        Compact SmallSat / CubeSat constellation platform for 200-220 km VLEO.
        """
        return cls(
            name="SkimSat VLEO SmallSat",
            mass_kg=180.0,
            cross_section_area_m2=0.45,
            drag_coefficient=2.2,
            nominal_altitude_km=220.0,
            inclination_deg=97.0,
        )

    @classmethod
    def tsubame_slats(cls) -> "VLEOSpacecraft":
        """
        JAXA SLATS (Super Low Altitude Test Satellite / Tsubame) configuration.
        Flown successfully between 167 km and 271 km with continuous ion propulsion.
        """
        return cls(
            name="JAXA SLATS Tsubame Benchmark",
            mass_kg=400.0,
            cross_section_area_m2=0.90,
            drag_coefficient=2.2,
            nominal_altitude_km=200.0,
            inclination_deg=98.0,
        )

    @classmethod
    def starlink_vleo(cls) -> "VLEOSpacecraft":
        """
        Commercial flat-panel VLEO constellation concept (~280 km).
        """
        return cls(
            name="Commercial VLEO FlatSat",
            mass_kg=300.0,
            cross_section_area_m2=0.65,
            drag_coefficient=2.25,
            nominal_altitude_km=280.0,
            inclination_deg=53.0,
        )


@dataclass
class VLEOThruster:
    """
    Electric or Air-Breathing Electric Propulsion (ABEP) hardware parameters.
    """
    name: str
    thrust_min_n: float            # Minimum controllable thrust (N)
    thrust_max_n: float            # Maximum peak thrust (N)
    slew_rate_limit_n_s: float     # Maximum rate of thrust variation dT/dt (N/s)
    isp_s: float                   # Specific impulse (s)
    power_per_thrust_w_n: float = 25000.0  # ~25 W/mN standard for ion/Hall thrusters
    idle_power_w: float = 120.0    # Housekeeping / cathode keeper power (W)
    throttle_step_n: Optional[float] = None # Digital throttle quantization step (None = continuous)
    time_constant_s: float = 0.05  # Actuator 1st-order response lag (s)
    is_air_breathing: bool = False # If True (ABEP), operates on ambient atmosphere (0 tank propellant)

    def calculate_propellant_rate_kg_s(self, thrust_n: float) -> float:
        """Calculate fuel consumption rate: m_dot = T / (Isp * g0) [kg/s]."""
        if self.is_air_breathing or self.isp_s <= 0 or thrust_n <= 0:
            return 0.0
        return float(thrust_n / (self.isp_s * STANDARD_GRAVITY))

    def calculate_power_consumption_w(self, thrust_n: float) -> float:
        """Calculate total electrical power drawn from PPU / solar panels."""
        if thrust_n <= 0:
            return self.idle_power_w
        return float(self.idle_power_w + thrust_n * self.power_per_thrust_w_n)

    @classmethod
    def goce_qit(cls) -> "VLEOThruster":
        """
        QinetiQ T5 / QIT-25 gridded ion thruster flown on ESA GOCE.
        Continuous fine throttling from 1.0 mN to 20.6 mN with 3100 s Isp.
        """
        return cls(
            name="GOCE QIT-25 Ion Thruster",
            thrust_min_n=0.001,      # 1 mN
            thrust_max_n=0.021,      # 21 mN
            slew_rate_limit_n_s=0.005, # 5 mN/s
            isp_s=3100.0,
            power_per_thrust_w_n=22000.0,
            idle_power_w=80.0,
            throttle_step_n=None,
            time_constant_s=0.08,
            is_air_breathing=False,
        )

    @classmethod
    def starlink_krypton(cls) -> "VLEOThruster":
        """
        High-thrust Hall effect thruster (80 mN, Krypton propellant, 1800 s Isp).
        """
        return cls(
            name="Starlink Krypton Hall Thruster",
            thrust_min_n=0.005,      # 5 mN
            thrust_max_n=0.080,      # 80 mN
            slew_rate_limit_n_s=0.020, # 20 mN/s
            isp_s=1800.0,
            power_per_thrust_w_n=17000.0,
            idle_power_w=150.0,
            throttle_step_n=0.0005,  # 0.5 mN discrete step
            time_constant_s=0.04,
            is_air_breathing=False,
        )

    @classmethod
    def abep_ram_ep(cls) -> "VLEOThruster":
        """
        Air-Breathing Electric Propulsion (ABEP / RAM-EP) concept.
        Ingests atmospheric O/N2 at orbital speed (7.8 km/s), ionizes and expels it.
        Consumes zero tank propellant, enabling multi-year operational lifetime at 180-220 km.
        """
        return cls(
            name="ABEP Atmosphere-Breathing RAM Thruster",
            thrust_min_n=0.002,      # 2 mN
            thrust_max_n=0.060,      # 60 mN
            slew_rate_limit_n_s=0.015,
            isp_s=1400.0,
            power_per_thrust_w_n=30000.0,
            idle_power_w=200.0,
            throttle_step_n=None,
            time_constant_s=0.10,
            is_air_breathing=True,
        )

    @classmethod
    def high_capacity_vleo(cls) -> "VLEOThruster":
        """
        High-power throttleable Hall cluster (up to 200 mN) designed for severe storm surges.
        """
        return cls(
            name="High-Capacity VLEO Storm Shield Thruster",
            thrust_min_n=0.002,      # 2 mN
            thrust_max_n=0.250,      # 250 mN
            slew_rate_limit_n_s=0.050, # 50 mN/s
            isp_s=2000.0,
            power_per_thrust_w_n=18000.0,
            idle_power_w=250.0,
            throttle_step_n=None,
            time_constant_s=0.03,
            is_air_breathing=False,
        )


@dataclass
class DisturbanceConfig:
    """
    Environmental disturbances acting on the spacecraft in VLEO.
    """
    enable_waves: bool = True
    # List of (amplitude_fraction, period_seconds, phase_radians)
    gravity_waves: List[Tuple[float, float, float]] = field(default_factory=lambda: [
        (0.15, 600.0, 0.0),    # 10 min acoustic-gravity wave (15% density variation)
        (0.10, 1800.0, 1.2),   # 30 min mesoscale wave (10% variation)
        (0.05, 300.0, 2.5),    # 5 min micro-wave perturbation (5% variation)
    ])
    enable_storm: bool = False
    storm_start_time_s: float = 1200.0       # Onset time of CME / geomagnetic storm
    storm_rise_time_s: float = 400.0         # Sigmoidal rise duration (s)
    storm_peak_multiplier: float = 3.5       # Density surge factor at peak (e.g. 3.5x = 350%)
    storm_decay_time_s: float = 3600.0       # Relaxation half-life (s)
    enable_diurnal: bool = True
    diurnal_amplitude: float = 0.25          # 25% day-to-night thermospheric bulge
    turbulence_sigma: float = 0.02           # Stochastic Gaussian turbulence (2% 1-sigma)
    accelerometer_noise_sigma: float = 1e-6  # Accelerometer measurement noise (m/s^2)
    accelerometer_bias: float = 5e-6         # Accelerometer sensor bias drift (m/s^2)
    spacecraft_area_fluctuation: float = 0.05 # 5% area modulation from attitude limit-cycling


@dataclass
class ADMUStepTelemetry:
    """Telemetry recorded at each integration / control cycle step."""
    time_s: float
    altitude_km: float
    orbital_speed_m_s: float
    relative_speed_m_s: float
    true_density_kg_m3: float
    true_drag_force_n: float
    true_drag_accel_m_s2: float
    commanded_thrust_n: float
    actual_thrust_n: float
    tracking_error_m_s: float
    estimated_drag_accel_m_s2: float
    is_saturated: bool
    cumulative_propellant_kg: float
    electrical_power_w: float
    l1_sigma_hat: Optional[float] = None
    l1_filter_out: Optional[float] = None
    pid_p: Optional[float] = None
    pid_i: Optional[float] = None
    pid_d: Optional[float] = None


@dataclass
class ADMUPerformanceReport:
    """Summary metrics evaluating controller tracking, fuel consumption, and storm resilience."""
    controller_type: ADMUControlMode
    duration_s: float
    initial_altitude_km: float
    final_altitude_km: float
    min_altitude_km: float
    altitude_loss_m: float
    rms_velocity_error_m_s: float
    max_velocity_error_m_s: float
    mean_velocity_error_m_s: float
    drag_compensation_efficiency_pct: float  # (Total thrust impulse / Total drag impulse) * 100
    total_drag_impulse_n_s: float
    total_thrust_impulse_n_s: float
    total_propellant_consumed_kg: float
    average_power_w: float
    peak_power_w: float
    chattering_index_n_s: float               # Integrated rate of thrust change: sum(|dT/dt|*dt)/duration
    saturation_fraction_pct: float           # % of time thruster was clamped at T_max or T_min
    survived_storm: bool                     # True if altitude stayed within allowable decay envelope
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert performance report to dictionary."""
        d = asdict(self)
        d["controller_type"] = self.controller_type.value
        return d


# ==============================================================================
# CONTROLLER 1: L1 ADAPTIVE CONTROLLER
# ==============================================================================

class L1AdaptiveController:
    """
    L1 Adaptive Controller for Spacecraft Drag Make-Up in VLEO.

    Key Theoretical Architecture (Hovakimyan & Cao):
    1. Reference model / Predictor:
       e_dot_hat(t) = Am * e_hat(t) + (u(t) + sigma_hat(t)) + L * (e(t) - e_hat(t))
       where e(t) = v(t) - v_ref is velocity tracking error, and Am < 0 is Hurwitz.
    2. Fast Adaptation Law:
       sigma_dot_hat(t) = Proj( sigma_hat(t), -Gamma * P * e_tilde(t) )
       where e_tilde = e_hat - e, P is the Lyapunov solution (2*P*Am = -Q),
       and Proj() is a smooth projection operator bounding parameter estimates.
    3. Low-Pass Control Filter:
       u_L(s) = -C(s) * sigma_hat(s)
       where C(s) is a 1st or 2nd-order Butterworth filter.
       This decouples the adaptation rate Gamma (which can be arbitrarily high, e.g. 1000+)
       from actuator control bandwidth, guaranteeing non-chattering robust control!
    """

    def __init__(
        self,
        am: float = -0.4,                    # Reference model pole (Hurwitz: am < 0)
        gamma: float = 800.0,                # High adaptation gain
        cutoff_freq_hz: float = 0.25,        # Filter bandwidth (Hz)
        filter_type: FilterType = FilterType.SECOND_ORDER_BUTTERWORTH,
        damping_ratio: float = 0.7071,       # Butterworth damping factor zeta = 1 / sqrt(2)
        observer_gain: float = 0.5,          # Observer feedback gain L >= 0
        sigma_min: float = -0.010,           # Lower disturbance acceleration bound (m/s^2)
        sigma_max: float = 0.010,            # Upper disturbance acceleration bound (m/s^2)
        k_base: float = 0.08,                # Nominal baseline feedback gain (proportional to e)
        q_lyap: float = 1.0,                 # Lyapunov weight Q > 0
    ):
        if am >= 0:
            raise ValueError(f"Reference pole am must be strictly negative (Hurwitz), got {am}")
        if gamma <= 0:
            raise ValueError(f"Adaptation gain gamma must be positive, got {gamma}")
        if cutoff_freq_hz <= 0:
            raise ValueError(f"Cutoff frequency must be positive, got {cutoff_freq_hz}")

        self.am = float(am)
        self.gamma = float(gamma)
        self.cutoff_freq_hz = float(cutoff_freq_hz)
        self.omega_c = 2.0 * math.pi * self.cutoff_freq_hz
        self.filter_type = filter_type
        self.damping_ratio = float(damping_ratio)
        self.observer_gain = float(observer_gain)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.k_base = float(k_base)
        self.q_lyap = float(q_lyap)

        # Lyapunov solution: 2 * P * (am - observer_gain) = -Q
        # Effective pole = am - observer_gain < 0
        effective_pole = self.am - self.observer_gain
        self.p_lyap = self.q_lyap / (2.0 * abs(effective_pole))

        # State variables
        self.e_hat: float = 0.0          # Predictor state
        self.sigma_hat: float = 0.0      # Adaptive disturbance estimate (m/s^2)
        self.filtered_sigma: float = 0.0 # Low-pass filtered disturbance estimate (m/s^2)

        # Filter internal states
        self.xf1: float = 0.0            # Filter state 1
        self.xf2: float = 0.0            # Filter state 2 (for 2nd-order Butterworth)

        # Diagnostics history
        self.last_e_tilde: float = 0.0
        self.last_u_cmd: float = 0.0

    def reset(self, initial_error: float = 0.0) -> None:
        """Reset internal predictor, adaptation, and filter states."""
        self.e_hat = float(initial_error)
        self.sigma_hat = 0.0
        self.filtered_sigma = 0.0
        self.xf1 = 0.0
        self.xf2 = 0.0
        self.last_e_tilde = 0.0
        self.last_u_cmd = 0.0

    def _projection(self, theta: float, y: float, theta_min: float, theta_max: float, eps: float = 1e-4) -> float:
        """
        Projection operator ensuring theta remains bounded within [theta_min, theta_max]:
        Proj(theta, y) = 0 if at boundary and moving outwards, else y.
        """
        if theta >= theta_max and y > 0:
            return 0.0
        if theta <= theta_min and y < 0:
            return 0.0
        # Smooth projection boundary layer transition
        if theta > theta_max - eps and y > 0:
            factor = max(0.0, (theta_max - theta) / eps)
            return factor * y
        if theta < theta_min + eps and y < 0:
            factor = max(0.0, (theta - theta_min) / eps)
            return factor * y
        return y

    def update_filter(self, u_in: float, dt: float) -> float:
        """
        Integrate continuous low-pass filter over time step dt.
        Returns filtered output.
        """
        if dt <= 0:
            return self.filtered_sigma

        # Ensure numerical stability with internal sub-stepping for stiff/fast filters
        max_sub = 0.01
        n_sub = max(1, int(math.ceil(dt / max_sub)))
        h = dt / n_sub

        for _ in range(n_sub):
            if self.filter_type == FilterType.FIRST_ORDER:
                # 1st order filter: C(s) = omega_c / (s + omega_c)
                # dx/dt = -omega_c * x + omega_c * u_in
                dx = -self.omega_c * self.xf1 + self.omega_c * u_in
                self.xf1 += dx * h
                self.filtered_sigma = self.xf1
            else:
                # 2nd order Butterworth: C(s) = omega_c^2 / (s^2 + 2*zeta*omega_c*s + omega_c^2)
                # State space:
                # dx1/dt = x2
                # dx2/dt = -omega_c^2 * x1 - 2*zeta*omega_c * x2 + omega_c^2 * u_in
                w2 = self.omega_c ** 2
                two_zeta_w = 2.0 * self.damping_ratio * self.omega_c

                dx1 = self.xf2
                dx2 = -w2 * self.xf1 - two_zeta_w * self.xf2 + w2 * u_in

                self.xf1 += dx1 * h
                self.xf2 += dx2 * h
                self.filtered_sigma = self.xf1

        return self.filtered_sigma

    def step(
        self,
        dt: float,
        tracking_error: float,
        applied_accel_cmd: float,
        feedforward_accel: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Execute one discrete control cycle of L1 adaptive controller.

        Args:
            dt: Sampling interval (s).
            tracking_error: Current velocity error e = v_actual - v_target (m/s).
            applied_accel_cmd: Actual acceleration executed by thrusters in previous step (m/s^2).
            feedforward_accel: Optional feedforward acceleration (e.g. nominal predicted drag).

        Returns:
            Tuple of (commanded_control_accel, diagnostics_dict).
        """
        if dt <= 0:
            return 0.0, {}

        # Internal sub-stepping ensures unconditional numerical stability
        # for high adaptation gains (Gamma = 500-2000) under arbitrary simulation step sizes
        max_sub_dt = 0.01  # 10 ms maximum internal step
        n_sub = max(1, int(math.ceil(dt / max_sub_dt)))
        sub_dt = dt / n_sub

        for _ in range(n_sub):
            # 1. Prediction error: e_tilde = e_hat - e
            e_tilde = self.e_hat - tracking_error
            self.last_e_tilde = e_tilde

            # 2. Adaptation law with projection
            # Adaptation gradient = -Gamma * P * e_tilde
            adapt_deriv_raw = -self.gamma * self.p_lyap * e_tilde
            sigma_dot = self._projection(
                self.sigma_hat,
                adapt_deriv_raw,
                self.sigma_min,
                self.sigma_max,
            )
            self.sigma_hat += sigma_dot * sub_dt
            self.sigma_hat = float(np.clip(self.sigma_hat, self.sigma_min, self.sigma_max))

            # 3. Predictor state integration
            e_hat_dot = (
                self.am * self.e_hat
                + applied_accel_cmd
                + self.sigma_hat
                - self.observer_gain * e_tilde
            )
            self.e_hat += e_hat_dot * sub_dt

            # 4. Low-pass filter evaluation
            filtered_sigma = self.update_filter(self.sigma_hat, sub_dt)

        # 5. Control command computation:
        # u_ad = - filtered_sigma (adaptively cancels the estimated disturbance)
        # u_base = - K_base * tracking_error (stabilizes baseline error dynamics)
        # u_ff = feedforward_accel
        u_ad = -self.filtered_sigma
        u_base = -self.k_base * tracking_error
        u_cmd = feedforward_accel + u_base + u_ad

        self.last_u_cmd = u_cmd

        diag = {
            "e_tilde": float(self.last_e_tilde),
            "e_hat": float(self.e_hat),
            "sigma_hat": float(self.sigma_hat),
            "filtered_sigma": float(self.filtered_sigma),
            "u_ad": float(u_ad),
            "u_base": float(u_base),
            "u_cmd": float(u_cmd),
        }
        return u_cmd, diag


# ==============================================================================
# CONTROLLER 2: PID WITH ANTI-WINDUP
# ==============================================================================

class PIDWithAntiWindup:
    """
    Flight-Grade PID Controller with Anti-Windup for Drag Make-Up.

    Features:
    - Conditional integration (clamping) anti-windup: Freezes integrator during saturation.
    - Back-calculation anti-windup: Dynamically discharges integrator via tracking feedback:
      dI/dt = Ki * e + Kb * (u_sat - u_unsat).
    - Filtered derivative on measurement to prevent derivative kick upon setpoint jumps.
    - Slew-rate limiting and hard saturation bounds.
    """

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        anti_windup_method: AntiWindupMethod = AntiWindupMethod.CLAMPING,
        back_calc_tracking_gain: Optional[float] = None,
        derivative_filter_tau_s: float = 0.08,
        derivative_on_measurement: bool = True,
        integral_min: float = -1e5,
        integral_max: float = 1e5,
        output_min: float = 0.0,
        output_max: float = 0.10,  # Max acceleration (m/s^2)
        slew_rate_limit: Optional[float] = None, # Max dU/dt (m/s^3)
    ):
        if kp < 0 or ki < 0 or kd < 0:
            raise ValueError("PID gains (kp, ki, kd) must be non-negative.")

        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.anti_windup_method = anti_windup_method
        # Default back-calculation tracking gain Kb = sqrt(Ki / Kd) or Kp / Ki if available
        if back_calc_tracking_gain is not None:
            self.kb = float(back_calc_tracking_gain)
        else:
            self.kb = 1.0 / math.sqrt(self.ki * self.kd) if (self.ki > 0 and self.kd > 0) else 1.0

        self.tau_d = max(1e-4, float(derivative_filter_tau_s))
        self.derivative_on_measurement = bool(derivative_on_measurement)
        self.integral_min = float(integral_min)
        self.integral_max = float(integral_max)
        self.output_min = float(output_min)
        self.output_max = float(output_max)
        self.slew_rate_limit = float(slew_rate_limit) if slew_rate_limit is not None else None

        # State registers
        self.integral_state: float = 0.0
        self.filtered_deriv: float = 0.0
        self.prev_measurement: Optional[float] = None
        self.prev_error: Optional[float] = None
        self.prev_output: float = 0.0
        self.is_saturated: bool = False

    def reset(self) -> None:
        """Reset internal accumulator, derivative filter, and memory registers."""
        self.integral_state = 0.0
        self.filtered_deriv = 0.0
        self.prev_measurement = None
        self.prev_error = None
        self.prev_output = 0.0
        self.is_saturated = False

    def step(
        self,
        dt: float,
        error: float,
        measurement: Optional[float] = None,
        feedforward: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Execute one discrete step of the Anti-Windup PID.

        Args:
            dt: Sampling time step (s).
            error: Target - Actual tracking error.
            measurement: Actual measured variable (used for derivative-on-measurement).
            feedforward: Nominal feedforward effort.

        Returns:
            Tuple of (saturated_control_output, diagnostics_dict).
        """
        if dt <= 0:
            return self.prev_output, {}

        # 1. Proportional term
        p_term = self.kp * error

        # 2. Derivative term with 1st-order low-pass filter
        if self.derivative_on_measurement and measurement is not None:
            if self.prev_measurement is None:
                d_input = 0.0
            else:
                # -dy/dt (negative because error = setpoint - measurement)
                d_input = -(measurement - self.prev_measurement) / dt
            self.prev_measurement = measurement
        else:
            if self.prev_error is None:
                d_input = 0.0
            else:
                d_input = (error - self.prev_error) / dt
            self.prev_error = error

        # Filter: tau_d * dD/dt + D = Kd * d_input
        deriv_alpha = dt / (self.tau_d + dt)
        self.filtered_deriv = (1.0 - deriv_alpha) * self.filtered_deriv + deriv_alpha * (self.kd * d_input)
        d_term = self.filtered_deriv

        # 3. Unsat output estimate (using current integral state)
        u_unsat = p_term + self.integral_state + d_term + feedforward

        # 4. Apply slew-rate limiter if configured
        u_rate_limited = u_unsat
        if self.slew_rate_limit is not None and self.prev_output is not None:
            max_delta = self.slew_rate_limit * dt
            u_rate_limited = float(np.clip(u_unsat, self.prev_output - max_delta, self.prev_output + max_delta))

        # 5. Actuator saturation clamp
        u_sat = float(np.clip(u_rate_limited, self.output_min, self.output_max))
        self.is_saturated = abs(u_sat - u_unsat) > 1e-7

        # 6. Anti-Windup Integral Update
        if self.anti_windup_method == AntiWindupMethod.CLAMPING:
            # Clamping / conditional integration:
            # Freeze integration if output is saturated and the error has the same sign as the saturation
            if self.is_saturated:
                if (u_unsat > self.output_max and error > 0) or (u_unsat < self.output_min and error < 0):
                    i_increment = 0.0  # Freeze integrator
                else:
                    i_increment = self.ki * error * dt
            else:
                i_increment = self.ki * error * dt

            self.integral_state += i_increment

        elif self.anti_windup_method == AntiWindupMethod.BACK_CALCULATION:
            # Back-calculation: dI/dt = Ki * error + Kb * (u_sat - u_unsat)
            # When saturated, the tracking difference discharges the accumulator
            saturation_diff = u_sat - u_unsat
            i_dot = self.ki * error + self.kb * saturation_diff
            self.integral_state += i_dot * dt

        elif self.anti_windup_method == AntiWindupMethod.INTEGRAL_SATURATION:
            self.integral_state += self.ki * error * dt

        # Hard clamp on accumulator to prevent numeric drift
        self.integral_state = float(np.clip(self.integral_state, self.integral_min, self.integral_max))

        self.prev_output = u_sat

        diag = {
            "p_term": float(p_term),
            "i_term": float(self.integral_state),
            "d_term": float(d_term),
            "u_unsat": float(u_unsat),
            "u_sat": float(u_sat),
            "is_saturated": bool(self.is_saturated),
        }
        return u_sat, diag


# ==============================================================================
# CONTROLLER 3: HYBRID L1 + PID DUAL-LOOP CONTROLLER
# ==============================================================================

class HybridL1PIDController:
    """
    Dual-Loop Architecture:
    - Inner Loop: L1 Adaptive Controller cancels high-frequency atmospheric disturbances,
      gravity waves, and solar storm density surges via filtered adaptive disturbance rejection.
    - Outer Loop: Anti-Windup PID regulates low-frequency orbital velocity and semi-major axis drifts.
    """

    def __init__(
        self,
        l1_controller: L1AdaptiveController,
        pid_controller: PIDWithAntiWindup,
    ):
        self.l1 = l1_controller
        self.pid = pid_controller

    def reset(self) -> None:
        self.l1.reset()
        self.pid.reset()

    def step(
        self,
        dt: float,
        error: float,
        applied_thrust_n: float,
        spacecraft_mass_kg: float,
        feedforward_n: float = 0.0,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Execute combined hybrid control step.

        Args:
            dt: Sampling interval (s).
            error: Tracking error e = v_actual - v_target (m/s).
            applied_thrust_n: Actual thrust output in previous cycle (N).
            spacecraft_mass_kg: Current spacecraft mass (kg).
            feedforward_n: Nominal feedforward thrust (N).

        Returns:
            Tuple of (commanded_thrust_n, diagnostic_dict).
        """
        applied_accel = applied_thrust_n / spacecraft_mass_kg if spacecraft_mass_kg > 0 else 0.0
        ff_accel = feedforward_n / spacecraft_mass_kg if spacecraft_mass_kg > 0 else 0.0

        # Outer loop: PID computes target correction acceleration on tracking error
        # PID receives (v_target - v_actual) = -error
        pid_accel, pid_diag = self.pid.step(
            dt=dt,
            error=-error,
            measurement=None,
            feedforward=0.0,
        )

        # Inner loop: L1 adaptive controller observes disturbance and filters it
        _, l1_diag = self.l1.step(
            dt=dt,
            tracking_error=error,
            applied_accel_cmd=applied_accel,
            feedforward_accel=0.0,
        )

        # Disturbance adaptive cancellation acceleration
        u_ad = l1_diag.get("u_ad", 0.0)

        # Combined command: Feedforward + PID tracking + L1 adaptive cancellation
        total_cmd_accel = ff_accel + pid_accel + u_ad
        commanded_thrust_n = max(0.0, total_cmd_accel * spacecraft_mass_kg)

        combined_diag = {
            "pid": pid_diag,
            "l1": l1_diag,
            "pid_accel": float(pid_accel),
            "u_ad": float(u_ad),
            "commanded_thrust_n": float(commanded_thrust_n),
        }
        return commanded_thrust_n, combined_diag


# ==============================================================================
# ATMOSPHERIC DISTURBANCE SYNTHESIZER
# ==============================================================================

class AtmosphericDisturbanceSynthesizer:
    """
    Generates realistic, physically-calibrated atmospheric density disturbances in VLEO:
    - Background NRLMSISE / Jacchia density at altitude (180 - 300 km)
    - Atmospheric gravity waves (superposition of sinusoidal waves with periods 5-60 min)
    - Coronal Mass Ejection (CME) geomagnetic storm surges (2x - 8x density leap)
    - Solar diurnal thermal bulge (day/night density cycle)
    - Stochastic thermospheric turbulence
    """

    def __init__(self, config: DisturbanceConfig):
        self.config = config

    def compute_density(
        self,
        altitude_km: float,
        time_s: float,
        orbit_true_anomaly_rad: float = 0.0,
        kp_base: float = QUIET_KP,
        f10_7_base: float = QUIET_F10_7,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate instantaneous thermospheric density rho(h, t) in kg/m^3 with all wave/storm components.
        """
        # 1. Base thermospheric density from model
        base_rho = compute_atmospheric_density(altitude_km, kp=kp_base, f10_7=f10_7_base)

        # 2. Atmospheric Gravity Waves: delta_rho = sum( A_k * sin(2*pi*t/T_k + phi_k) )
        wave_multiplier = 1.0
        if self.config.enable_waves:
            wave_delta = 0.0
            for amp, period, phase in self.config.gravity_waves:
                if period > 0:
                    wave_delta += amp * math.sin(2.0 * math.pi * time_s / period + phase)
            wave_multiplier = max(0.1, 1.0 + wave_delta)

        # 3. Geomagnetic Storm Density Surge: CME arrival
        storm_multiplier = 1.0
        if self.config.enable_storm and time_s >= self.config.storm_start_time_s:
            t_rel = time_s - self.config.storm_start_time_s
            # Sigmoidal onset: rises from 1.0 to peak_multiplier
            rise_factor = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, (t_rel - self.config.storm_rise_time_s / 2.0) / (self.config.storm_rise_time_s / 5.0)))))
            # Exponential decay after peak
            t_after_peak = max(0.0, t_rel - self.config.storm_rise_time_s)
            decay_factor = math.exp(-t_after_peak / max(1.0, self.config.storm_decay_time_s))

            surge_magnitude = (self.config.storm_peak_multiplier - 1.0) * rise_factor * decay_factor
            storm_multiplier = 1.0 + surge_magnitude

        # 4. Solar Diurnal Bulge: 1 + A_diurnal * cos(theta - phi_sun)
        diurnal_multiplier = 1.0
        if self.config.enable_diurnal:
            # Orbital diurnal variation (peak at ~14:00 local solar time, ~120 deg offset)
            diurnal_multiplier = 1.0 + self.config.diurnal_amplitude * math.cos(orbit_true_anomaly_rad - math.radians(60.0))

        # 5. Stochastic turbulence (deterministic pseudo-random variation based on sine hashes)
        # Using pure deterministic math for strict repeatability in unit tests
        turb_seed = math.sin(time_s * 1.732 + 3.1415) * math.cos(time_s * 0.457)
        turbulence_multiplier = 1.0 + self.config.turbulence_sigma * turb_seed

        # Total combined atmospheric density
        total_density = base_rho * wave_multiplier * storm_multiplier * diurnal_multiplier * turbulence_multiplier
        total_density = max(1e-15, float(total_density))

        diag = {
            "base_density_kg_m3": float(base_rho),
            "wave_multiplier": float(wave_multiplier),
            "storm_multiplier": float(storm_multiplier),
            "diurnal_multiplier": float(diurnal_multiplier),
            "total_density_kg_m3": float(total_density),
        }
        return total_density, diag


# ==============================================================================
# ACTUATOR & THRUSTER EMULATOR
# ==============================================================================

class ThrusterActuatorEmulator:
    """
    Physical actuator emulation:
    - Slew-rate limiting (|dT/dt| <= slew_limit)
    - Thrust saturation ([T_min, T_max])
    - Throttle quantization (discrete step discretization)
    - Actuator 1st-order response lag: tau * dT_act/dt + T_act = T_quant
    - Real-time mass depletion and power consumption tracking.
    """

    def __init__(self, hardware: VLEOThruster):
        self.hardware = hardware
        self.actual_thrust_n: float = 0.0
        self.target_thrust_n: float = 0.0
        self.cumulative_propellant_kg: float = 0.0

    def reset(self) -> None:
        self.actual_thrust_n = 0.0
        self.target_thrust_n = 0.0
        self.cumulative_propellant_kg = 0.0

    def step(self, dt: float, commanded_thrust_n: float) -> Tuple[float, bool]:
        """
        Advance actuator state by dt under commanded thrust.

        Returns:
            Tuple of (actual_thrust_executed_n, is_saturated).
        """
        if dt <= 0:
            return self.actual_thrust_n, False

        # 1. Clamp to allowable physical hardware bounds (or zero if thruster is off)
        if commanded_thrust_n <= 0.0:
            t_bounded = 0.0
            is_saturated = False
        else:
            t_bounded = float(np.clip(commanded_thrust_n, self.hardware.thrust_min_n, self.hardware.thrust_max_n))
            is_saturated = abs(t_bounded - commanded_thrust_n) > 1e-7

        # 2. Slew rate limit
        max_delta = self.hardware.slew_rate_limit_n_s * dt
        delta = float(np.clip(t_bounded - self.target_thrust_n, -max_delta, max_delta))
        self.target_thrust_n += delta

        # 3. Throttle quantization (if hardware has discrete steps)
        if self.target_thrust_n <= 0.0 or t_bounded <= 0.0:
            t_quantized = 0.0
            self.target_thrust_n = 0.0
        else:
            t_quantized = self.target_thrust_n
            if self.hardware.throttle_step_n is not None and self.hardware.throttle_step_n > 0:
                step = self.hardware.throttle_step_n
                t_quantized = round(t_quantized / step) * step
                t_quantized = float(np.clip(t_quantized, self.hardware.thrust_min_n, self.hardware.thrust_max_n))

        # 4. Actuator 1st-order response lag
        if t_bounded <= 0.0 and self.target_thrust_n <= 0.0:
            self.actual_thrust_n = 0.0
        else:
            tau = max(1e-4, self.hardware.time_constant_s)
            alpha = dt / (tau + dt)
            self.actual_thrust_n = (1.0 - alpha) * self.actual_thrust_n + alpha * t_quantized

        # 5. Mass depletion
        m_dot = self.hardware.calculate_propellant_rate_kg_s(self.actual_thrust_n)
        self.cumulative_propellant_kg += m_dot * dt

        return self.actual_thrust_n, is_saturated


# ==============================================================================
# FULL CLOSED-LOOP ADMU SIMULATION ENGINE
# ==============================================================================

class ADMUSimulator:
    """
    High-Fidelity VLEO 2-DOF Orbital Motion & Drag Compensation Simulation Engine.

    Equations of Motion in Polar Orbital Plane (r, theta):
    - r_dot = v_r
    - theta_dot = v_t / r
    - v_r_dot = v_t^2 / r - mu / r^2 + a_r_thrust
    - v_t_dot = - (v_r * v_t) / r + (T_thrust / m) - a_drag
    where:
    - a_drag = 0.5 * rho(h, t) * (Cd * A / m) * v_rel^2
    - v_rel = v_t - omega_earth * r * cos(inclination)
    """

    def __init__(
        self,
        spacecraft: VLEOSpacecraft,
        thruster: VLEOThruster,
        disturbance_config: DisturbanceConfig,
        control_mode: ADMUControlMode = ADMUControlMode.L1_ADAPTIVE,
        l1_controller: Optional[L1AdaptiveController] = None,
        pid_controller: Optional[PIDWithAntiWindup] = None,
    ):
        self.spacecraft = spacecraft
        self.thruster = thruster
        self.disturbance_config = disturbance_config
        self.control_mode = control_mode

        # Initialize disturbance synthesizer
        self.dist_synthesizer = AtmosphericDisturbanceSynthesizer(disturbance_config)

        # Initialize actuator
        self.actuator = ThrusterActuatorEmulator(thruster)

        # Initialize controllers
        max_accel = self.thruster.thrust_max_n / self.spacecraft.mass_kg
        if l1_controller is not None:
            self.l1 = l1_controller
        else:
            self.l1 = L1AdaptiveController(
                am=-0.5,
                gamma=1000.0,
                cutoff_freq_hz=0.20,
                sigma_min=-max_accel * 1.5,
                sigma_max=max_accel * 1.5,
                k_base=0.10,
            )

        if pid_controller is not None:
            self.pid = pid_controller
        else:
            # Tuned PID gains for VLEO velocity tracking
            self.pid = PIDWithAntiWindup(
                kp=0.15,
                ki=0.008,
                kd=0.50,
                anti_windup_method=AntiWindupMethod.CLAMPING,
                output_min=-max_accel,
                output_max=max_accel,
            )

        self.hybrid = HybridL1PIDController(self.l1, self.pid)

    def run_simulation(
        self,
        duration_s: float = 3600.0,
        dt: float = 0.5,
    ) -> Tuple[ADMUPerformanceReport, List[ADMUStepTelemetry]]:
        """
        Execute full closed-loop drag make-up simulation over duration_s.

        Args:
            duration_s: Total mission simulation time in seconds.
            dt: Numerical integration time step (s). Recommended: 0.1 to 1.0 s.

        Returns:
            Tuple of (ADMUPerformanceReport, telemetry_history).
        """
        # Initial orbit conditions (circular orbit at nominal altitude)
        r0 = (EARTH_RADIUS_KM + self.spacecraft.nominal_altitude_km) * 1000.0  # meters
        v_target = math.sqrt(EARTH_MU_SI / r0)                               # circular orbital speed (m/s)

        r = r0
        theta = 0.0
        v_r = 0.0
        v_t = v_target

        # Reset controllers & actuator
        self.l1.reset()
        self.pid.reset()
        self.hybrid.reset()
        self.actuator.reset()

        telemetry_log: List[ADMUStepTelemetry] = []
        n_steps = int(math.ceil(duration_s / dt))

        total_thrust_impulse = 0.0
        total_drag_impulse = 0.0
        chattering_sum = 0.0
        saturation_steps = 0
        prev_thrust = 0.0

        earth_omega = 7.292115e-5  # rad/s
        inc_rad = math.radians(self.spacecraft.inclination_deg)
        cos_inc = math.cos(inc_rad)

        for step_idx in range(n_steps):
            time_now = step_idx * dt
            alt_km = (r - EARTH_RADIUS_M) / 1000.0

            # 1. Compute true atmospheric density
            true_rho, dist_diag = self.dist_synthesizer.compute_density(
                altitude_km=alt_km,
                time_s=time_now,
                orbit_true_anomaly_rad=theta,
            )

            # 2. Relative atmospheric velocity
            v_earth_surface = earth_omega * r * cos_inc
            v_rel = v_t - v_earth_surface

            # Dynamic spacecraft area fluctuation (e.g. slight attitude motion)
            area_mod = 1.0 + self.disturbance_config.spacecraft_area_fluctuation * math.sin(time_now * 0.1)
            effective_area = self.spacecraft.cross_section_area_m2 * area_mod

            # Unmodeled Cd uncertainty
            effective_cd = self.spacecraft.drag_coefficient * (1.0 + self.spacecraft.cd_uncertainty_fraction * 0.5)

            # 3. Aerodynamic drag force & acceleration
            q_dynamic = 0.5 * true_rho * (v_rel ** 2)
            true_f_drag = q_dynamic * effective_cd * effective_area
            current_mass = self.spacecraft.mass_kg - self.actuator.cumulative_propellant_kg
            current_mass = max(10.0, current_mass)
            true_a_drag = true_f_drag / current_mass

            # 4. Sensor measurement (accelerometer with noise & bias)
            # Net acceleration along-track = a_thrust - a_drag
            accel_meas = (
                (self.actuator.actual_thrust_n / current_mass)
                - true_a_drag
                + self.disturbance_config.accelerometer_bias
                + self.disturbance_config.accelerometer_noise_sigma * math.sin(time_now * 13.7)
            )
            # Estimated drag acceleration from accelerometer
            est_a_drag = max(0.0, (self.actuator.actual_thrust_n / current_mass) - accel_meas)

            # 5. Tracking error: e = v_t - v_target
            velocity_error = v_t - v_target

            # Nominal feedforward thrust based on standard quiet density
            quiet_rho = compute_atmospheric_density(alt_km, QUIET_KP, QUIET_F10_7)
            nom_ff_drag_n = 0.5 * quiet_rho * (v_rel ** 2) * self.spacecraft.drag_coefficient * self.spacecraft.cross_section_area_m2
            nom_ff_accel = nom_ff_drag_n / current_mass

            # 6. Controller evaluation based on operational mode
            l1_diag = {}
            pid_diag = {}
            cmd_thrust_n = 0.0

            if self.control_mode == ADMUControlMode.UNCONTROLLED:
                cmd_thrust_n = 0.0

            elif self.control_mode == ADMUControlMode.FEEDFORWARD_ONLY:
                cmd_thrust_n = nom_ff_drag_n

            elif self.control_mode == ADMUControlMode.L1_ADAPTIVE:
                # L1 Adaptive Controller
                # Error passed is tracking error: e = v_actual - v_target
                # Control law aims to drive e -> 0 while canceling disturbance
                cmd_accel, l1_diag = self.l1.step(
                    dt=dt,
                    tracking_error=velocity_error,
                    applied_accel_cmd=self.actuator.actual_thrust_n / current_mass,
                    feedforward_accel=nom_ff_accel,
                )
                cmd_thrust_n = cmd_accel * current_mass

            elif self.control_mode == ADMUControlMode.PID_ANTI_WINDUP:
                # PID with Anti-Windup
                # Error = v_target - v_actual (positive when lagging behind target)
                cmd_accel, pid_diag = self.pid.step(
                    dt=dt,
                    error=-velocity_error,
                    measurement=v_t,
                    feedforward=nom_ff_accel,
                )
                cmd_thrust_n = cmd_accel * current_mass

            elif self.control_mode == ADMUControlMode.HYBRID_L1_PID:
                cmd_thrust_n, hybrid_diag = self.hybrid.step(
                    dt=dt,
                    error=velocity_error,
                    applied_thrust_n=self.actuator.actual_thrust_n,
                    spacecraft_mass_kg=current_mass,
                    feedforward_n=nom_ff_drag_n,
                )
                l1_diag = hybrid_diag.get("l1", {})
                pid_diag = hybrid_diag.get("pid", {})

            # 7. Actuator step execution
            actual_thrust_n, is_saturated = self.actuator.step(dt, cmd_thrust_n)

            # 8. Record metrics
            total_thrust_impulse += actual_thrust_n * dt
            total_drag_impulse += true_f_drag * dt
            chattering_sum += abs(actual_thrust_n - prev_thrust)
            if is_saturated:
                saturation_steps += 1
            prev_thrust = actual_thrust_n

            power_w = self.thruster.calculate_power_consumption_w(actual_thrust_n)

            telemetry_log.append(ADMUStepTelemetry(
                time_s=time_now,
                altitude_km=alt_km,
                orbital_speed_m_s=v_t,
                relative_speed_m_s=v_rel,
                true_density_kg_m3=true_rho,
                true_drag_force_n=true_f_drag,
                true_drag_accel_m_s2=true_a_drag,
                commanded_thrust_n=cmd_thrust_n,
                actual_thrust_n=actual_thrust_n,
                tracking_error_m_s=velocity_error,
                estimated_drag_accel_m_s2=est_a_drag,
                is_saturated=is_saturated,
                cumulative_propellant_kg=self.actuator.cumulative_propellant_kg,
                electrical_power_w=power_w,
                l1_sigma_hat=l1_diag.get("sigma_hat"),
                l1_filter_out=l1_diag.get("filtered_sigma"),
                pid_p=pid_diag.get("p_term"),
                pid_i=pid_diag.get("i_term"),
                pid_d=pid_diag.get("d_term"),
            ))

            # 9. 2-DOF Polar Orbit State Integration (Symplectic Euler / Verlet)
            # Along-track acceleration:
            dv_t_dt = - (v_r * v_t) / r + (actual_thrust_n / current_mass) - true_a_drag
            v_t += dv_t_dt * dt

            # Radial acceleration:
            dv_r_dt = (v_t ** 2) / r - (EARTH_MU_SI / (r ** 2))
            v_r += dv_r_dt * dt

            # Position updates:
            r += v_r * dt
            theta += (v_t / r) * dt

        # Summary performance metrics
        initial_alt = telemetry_log[0].altitude_km
        final_alt = telemetry_log[-1].altitude_km
        min_alt = min(t.altitude_km for t in telemetry_log)
        alt_loss = (initial_alt - final_alt) * 1000.0  # meters

        vel_errors = np.array([t.tracking_error_m_s for t in telemetry_log])
        rms_vel_err = float(np.sqrt(np.mean(vel_errors ** 2)))
        max_vel_err = float(np.max(np.abs(vel_errors)))
        mean_vel_err = float(np.mean(vel_errors))

        comp_efficiency = (total_thrust_impulse / total_drag_impulse * 100.0) if total_drag_impulse > 0 else 0.0
        chattering_index = chattering_sum / duration_s if duration_s > 0 else 0.0
        saturation_pct = (saturation_steps / n_steps) * 100.0 if n_steps > 0 else 0.0

        powers = [t.electrical_power_w for t in telemetry_log]
        avg_power = float(np.mean(powers))
        peak_power = float(np.max(powers))

        # Spacecraft survives if altitude does not decay below re-entry threshold (e.g. 150 km)
        survived = min_alt >= 160.0

        report = ADMUPerformanceReport(
            controller_type=self.control_mode,
            duration_s=duration_s,
            initial_altitude_km=initial_alt,
            final_altitude_km=final_alt,
            min_altitude_km=min_alt,
            altitude_loss_m=alt_loss,
            rms_velocity_error_m_s=rms_vel_err,
            max_velocity_error_m_s=max_vel_err,
            mean_velocity_error_m_s=mean_vel_err,
            drag_compensation_efficiency_pct=comp_efficiency,
            total_drag_impulse_n_s=total_drag_impulse,
            total_thrust_impulse_n_s=total_thrust_impulse,
            total_propellant_consumed_kg=self.actuator.cumulative_propellant_kg,
            average_power_w=avg_power,
            peak_power_w=peak_power,
            chattering_index_n_s=chattering_index,
            saturation_fraction_pct=saturation_pct,
            survived_storm=survived,
            diagnostics={
                "n_steps": n_steps,
                "dt_s": dt,
                "spacecraft": self.spacecraft.name,
                "thruster": self.thruster.name,
            },
        )

        return report, telemetry_log


# ==============================================================================
# BENCHMARK & COMPARATIVE EVALUATOR
# ==============================================================================

def compare_admu_controllers(
    spacecraft: Optional[VLEOSpacecraft] = None,
    thruster: Optional[VLEOThruster] = None,
    disturbance_config: Optional[DisturbanceConfig] = None,
    duration_s: float = 2400.0,
    dt: float = 0.5,
) -> Dict[str, ADMUPerformanceReport]:
    """
    Run comparative side-by-side evaluation of all control architectures under identical atmospheric disturbances.

    Returns:
        Dictionary mapping control mode names to their respective ADMUPerformanceReport.
    """
    sc = spacecraft or VLEOSpacecraft.skimsat_vleo()
    th = thruster or VLEOThruster.starlink_krypton()
    dc = disturbance_config or DisturbanceConfig(
        enable_waves=True,
        enable_storm=True,
        storm_start_time_s=600.0,
        storm_rise_time_s=300.0,
        storm_peak_multiplier=3.0,
        storm_decay_time_s=1800.0,
    )

    modes_to_test = [
        ADMUControlMode.UNCONTROLLED,
        ADMUControlMode.FEEDFORWARD_ONLY,
        ADMUControlMode.PID_ANTI_WINDUP,
        ADMUControlMode.L1_ADAPTIVE,
        ADMUControlMode.HYBRID_L1_PID,
    ]

    results: Dict[str, ADMUPerformanceReport] = {}

    for mode in modes_to_test:
        sim = ADMUSimulator(
            spacecraft=sc,
            thruster=th,
            disturbance_config=dc,
            control_mode=mode,
        )
        report, _ = sim.run_simulation(duration_s=duration_s, dt=dt)
        results[mode.value] = report

    return results


def tune_l1_drag_controller(
    spacecraft: VLEOSpacecraft,
    actuator_bandwidth_hz: float = 0.5,
    max_expected_drag_accel_m_s2: float = 0.010,
) -> L1AdaptiveController:
    """
    Synthesize an L1 adaptive controller matched to spacecraft physical parameters.

    Args:
        spacecraft: Spacecraft mass and geometry specifications.
        actuator_bandwidth_hz: Thruster bandwidth cutoff frequency (Hz).
        max_expected_drag_accel_m_s2: Conservative peak disturbance acceleration.

    Returns:
        Configured and tuned L1AdaptiveController instance.
    """
    # Filter cutoff chosen at half of actuator bandwidth to guarantee gain/phase margin
    filter_cutoff = max(0.05, actuator_bandwidth_hz * 0.5)

    # Reference model pole Am: chosen around -0.3 to -0.8 for smooth tracking
    am = -0.4

    # Fast adaptation gain Gamma: 500 to 2000
    gamma = 1000.0

    # Bounds on disturbance acceleration
    sigma_bound = max_expected_drag_accel_m_s2 * 2.0

    return L1AdaptiveController(
        am=am,
        gamma=gamma,
        cutoff_freq_hz=filter_cutoff,
        filter_type=FilterType.SECOND_ORDER_BUTTERWORTH,
        damping_ratio=0.7071,
        sigma_min=-sigma_bound,
        sigma_max=sigma_bound,
        k_base=0.12,
    )
