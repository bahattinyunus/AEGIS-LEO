"""
Space-Grade OBC Flight Computer Hardware Watchdog & Telemetry Monitor
====================================================================
Implements a hard real-time hardware supervisor for spaceborne On-Board Computers
(e.g., BAE RAD750, Cobham Gaisler GR740, Vorago VA416x0, SpaceCloud Linux/RTOS).

Monitors:
1. Single Event Upset (SEU) bit-flips & EDAC/ECC memory scrubbing.
2. Orbital thermal cycling (LEO daylight/eclipse hysteresis & Coffin-Manson fatigue).
3. Core clock throttling via Dynamic Voltage & Frequency Scaling (DVFS).
4. INT8 neural inference memory bounds, guard-band canary integrity, and tensor saturation.
5. Windowed hardware watchdog heartbeat (anti-runaway & anti-hang protection).

Dependencies:
- numpy
- standard library (math, time, dataclasses, enum, typing, hashlib)
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import time
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np


# ---------------------------------------------------------------------------
# Enums and System States
# ---------------------------------------------------------------------------

class WatchdogState(str, Enum):
    """Operational states of the OBC hardware watchdog."""
    NOMINAL = "NOMINAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    SAFE_MODE = "SAFE_MODE"
    RESET_TRIGGERED = "RESET_TRIGGERED"


class ThermalZone(str, Enum):
    """Thermal operational bands for space avionics."""
    COLD_WARNING = "COLD_WARNING"          # T < -25 deg C
    NOMINAL = "NOMINAL"                    # -25 deg C <= T <= +65 deg C
    ELEVATED = "ELEVATED"                  # +65 deg C < T <= +75 deg C
    THROTTLED = "THROTTLED"                # +75 deg C < T <= +85 deg C
    OVERTEMP_CRITICAL = "OVERTEMP_CRITICAL"# T > +85 deg C (thermal runaway risk)


class SEUSeverity(str, Enum):
    """Radiation-induced Single Event Effect severity levels."""
    NONE = "NONE"
    CORRECTABLE_SINGLE_BIT = "CORRECTABLE_SINGLE_BIT"  # SEC (Single Error Corrected)
    UNCORRECTABLE_MULTI_BIT = "UNCORRECTABLE_MULTI_BIT" # DED (Double Error Detected)
    BURST_FLIP_STORM = "BURST_FLIP_STORM"              # High rate (e.g. SAA or SPE)


class ThrottleReason(str, Enum):
    """Causal triggers for core clock frequency throttling."""
    NONE = "NONE"
    THERMAL_LIMIT = "THERMAL_LIMIT"
    RADIATION_STORM = "RADIATION_STORM"
    POWER_BROWNOUT = "POWER_BROWNOUT"
    MEMORY_PRESSURE = "MEMORY_PRESSURE"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"


class MitigationAction(str, Enum):
    """Autonomous mitigation actions ordered by the hardware watchdog."""
    NO_ACTION = "NO_ACTION"
    TRIGGER_MEMORY_SCRUB = "TRIGGER_MEMORY_SCRUB"
    THROTTLE_CLOCK = "THROTTLE_CLOCK"
    RESTORE_NOMINAL_CLOCK = "RESTORE_NOMINAL_CLOCK"
    FLUSH_INT8_SCRATCHPAD = "FLUSH_INT8_SCRATCHPAD"
    ENTER_SAFE_MODE = "ENTER_SAFE_MODE"
    FORCE_HARDWARE_RESET = "FORCE_HARDWARE_RESET"


# ---------------------------------------------------------------------------
# Telemetry Data Structures
# ---------------------------------------------------------------------------

@dataclass
class EDACStats:
    """EDAC / ECC error detection and correction telemetry."""
    single_bit_errors: int = 0
    multi_bit_errors: int = 0
    scrub_cycles: int = 0
    uptime_seconds: float = 0.0
    seu_rate_per_mb_day: float = 0.0
    burst_event_count: int = 0
    last_error_timestamp: Optional[float] = None
    severity: SEUSeverity = SEUSeverity.NONE


@dataclass
class ThermalStats:
    """Thermal cycle and die temperature telemetry."""
    die_temp_c: float = 25.0
    board_temp_c: float = 20.0
    min_temp_c: float = 25.0
    max_temp_c: float = 25.0
    thermal_rate_c_s: float = 0.0
    completed_thermal_cycles: int = 0
    current_cycle_amplitude: float = 0.0
    accumulated_fatigue_damage: float = 0.0
    fatigue_life_consumed_pct: float = 0.0
    zone: ThermalZone = ThermalZone.NOMINAL


@dataclass
class ClockThrottleStats:
    """Core clock Dynamic Voltage and Frequency Scaling (DVFS) telemetry."""
    current_clock_mhz: float = 400.0
    nominal_clock_mhz: float = 400.0
    throttle_ratio: float = 1.0
    is_throttled: bool = False
    throttle_reason: ThrottleReason = ThrottleReason.NONE
    total_throttled_time_s: float = 0.0
    duty_cycle_pct: float = 0.0


@dataclass
class INT8MemoryStats:
    """INT8 neural inference memory bounds and guard band telemetry."""
    allocated_bytes: int = 0
    budget_bytes: int = 1048576  # Default 1 MB on-chip SRAM/TCM
    usage_pct: float = 0.0
    peak_allocated_bytes: int = 0
    active_buffers_count: int = 0
    guard_bands_intact: bool = True
    corrupted_guard_bands: int = 0
    activation_saturation_rate: float = 0.0
    is_out_of_bounds: bool = False


@dataclass
class WatchdogReport:
    """Consolidated space-grade OBC flight hardware health report."""
    timestamp: float
    state: WatchdogState
    is_healthy: bool
    heartbeat_counter: int
    time_since_last_kick_s: float
    window_valid: bool
    edac: EDACStats
    thermal: ThermalStats
    clock: ClockThrottleStats
    memory: INT8MemoryStats
    active_alarms: List[str] = field(default_factory=list)
    recommended_action: MitigationAction = MitigationAction.NO_ACTION

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry report into JSON-serializable dictionary."""
        return {
            "timestamp": round(self.timestamp, 3),
            "state": self.state.value,
            "is_healthy": self.is_healthy,
            "heartbeat_counter": self.heartbeat_counter,
            "time_since_last_kick_s": round(self.time_since_last_kick_s, 4),
            "window_valid": self.window_valid,
            "edac": {
                "single_bit_errors": self.edac.single_bit_errors,
                "multi_bit_errors": self.edac.multi_bit_errors,
                "scrub_cycles": self.edac.scrub_cycles,
                "seu_rate_per_mb_day": round(self.edac.seu_rate_per_mb_day, 4),
                "burst_event_count": self.edac.burst_event_count,
                "severity": self.edac.severity.value,
            },
            "thermal": {
                "die_temp_c": round(self.thermal.die_temp_c, 2),
                "board_temp_c": round(self.thermal.board_temp_c, 2),
                "min_temp_c": round(self.thermal.min_temp_c, 2),
                "max_temp_c": round(self.thermal.max_temp_c, 2),
                "thermal_rate_c_s": round(self.thermal.thermal_rate_c_s, 4),
                "completed_thermal_cycles": self.thermal.completed_thermal_cycles,
                "current_cycle_amplitude": round(self.thermal.current_cycle_amplitude, 2),
                "fatigue_life_consumed_pct": round(self.thermal.fatigue_life_consumed_pct, 4),
                "zone": self.thermal.zone.value,
            },
            "clock": {
                "current_clock_mhz": round(self.clock.current_clock_mhz, 1),
                "nominal_clock_mhz": round(self.clock.nominal_clock_mhz, 1),
                "throttle_ratio": round(self.clock.throttle_ratio, 3),
                "is_throttled": self.clock.is_throttled,
                "throttle_reason": self.clock.throttle_reason.value,
                "duty_cycle_pct": round(self.clock.duty_cycle_pct, 2),
            },
            "memory": {
                "allocated_bytes": self.memory.allocated_bytes,
                "budget_bytes": self.memory.budget_bytes,
                "usage_pct": round(self.memory.usage_pct, 2),
                "peak_allocated_bytes": self.memory.peak_allocated_bytes,
                "active_buffers_count": self.memory.active_buffers_count,
                "guard_bands_intact": self.memory.guard_bands_intact,
                "corrupted_guard_bands": self.memory.corrupted_guard_bands,
                "activation_saturation_rate": round(self.memory.activation_saturation_rate, 4),
                "is_out_of_bounds": self.memory.is_out_of_bounds,
            },
            "active_alarms": list(self.active_alarms),
            "recommended_action": self.recommended_action.value,
        }


# ---------------------------------------------------------------------------
# Subsystem 1: SEU Bit-Flip & EDAC Monitor
# ---------------------------------------------------------------------------

class SEUWatchdog:
    """
    Tracks cosmic ray & solar proton induced Single Event Upsets (SEUs).
    Monitors Error Detection and Correction (EDAC / SECDED Hamming codes),
    scrubbing cycles, and SEU burst detection in South Atlantic Anomaly (SAA).
    """

    def __init__(
        self,
        monitored_memory_mb: float = 16.0,
        burst_time_window_s: float = 10.0,
        burst_threshold_flips: int = 6,
    ):
        self.monitored_mb = max(0.1, float(monitored_memory_mb))
        self.burst_window_s = float(burst_time_window_s)
        self.burst_threshold = int(burst_threshold_flips)

        self.single_bit_errors = 0
        self.multi_bit_errors = 0
        self.scrub_cycles = 0
        self.burst_event_count = 0
        self.last_error_time: Optional[float] = None
        self._flip_history: List[Tuple[float, int]] = []  # (timestamp, flip_count)
        self._severity = SEUSeverity.NONE

    def record_edac_event(
        self,
        single_bit_flips: int = 0,
        multi_bit_flips: int = 0,
        timestamp: Optional[float] = None,
    ) -> SEUSeverity:
        """
        Record EDAC hardware interrupt events.
        Single-bit errors are corrected in hardware (SEC);
        Multi-bit errors represent uncorrectable memory corruption (DED).
        """
        t = time.time() if timestamp is None else float(timestamp)
        sbe = max(0, int(single_bit_flips))
        mbe = max(0, int(multi_bit_flips))

        if sbe > 0 or mbe > 0:
            self.last_error_time = t
            self.single_bit_errors += sbe
            self.multi_bit_errors += mbe
            self._flip_history.append((t, sbe + mbe))

        # Prune old history outside burst window
        cutoff = t - self.burst_window_s
        self._flip_history = [entry for entry in self._flip_history if entry[0] >= cutoff]

        recent_flips = sum(count for _, count in self._flip_history)

        if mbe > 0:
            self._severity = SEUSeverity.UNCORRECTABLE_MULTI_BIT
        elif recent_flips >= self.burst_threshold:
            self._severity = SEUSeverity.BURST_FLIP_STORM
            self.burst_event_count += 1
        elif sbe > 0:
            self._severity = SEUSeverity.CORRECTABLE_SINGLE_BIT
        else:
            self._severity = SEUSeverity.NONE

        return self._severity

    def perform_memory_scrub(
        self,
        scrub_bytes: int = 1048576,
        simulated_corruptions: int = 0,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute scheduled background memory scrubbing cycle across RAM pages.
        Scrubbing reads words, evaluates parity/syndrome, repairs single-bit errors,
        and rewrites clean data to eliminate accumulation of double-bit faults.
        """
        t = time.time() if timestamp is None else float(timestamp)
        self.scrub_cycles += 1
        corruptions = max(0, int(simulated_corruptions))

        if corruptions > 0:
            self.record_edac_event(single_bit_flips=corruptions, multi_bit_flips=0, timestamp=t)

        scrubbed_mb = scrub_bytes / (1024.0 * 1024.0)
        return {
            "scrub_cycle": self.scrub_cycles,
            "bytes_scrubbed": scrub_bytes,
            "megabytes_scrubbed": round(scrubbed_mb, 3),
            "flips_repaired": corruptions,
            "timestamp": t,
            "status": "SCRUB_COMPLETE_NOMINAL" if self.multi_bit_errors == 0 else "UNCORRECTABLE_ERROR_PRESENT",
        }

    @staticmethod
    def compute_block_checksum(data: np.ndarray) -> int:
        """
        Calculate fast 32-bit Fletcher/CRC-like checksum for array integrity.
        """
        flat = data.astype(np.uint8, copy=False).ravel()
        sum1 = int(np.sum(flat, dtype=np.uint64) % 65535)
        indices = np.arange(1, len(flat) + 1, dtype=np.uint64)
        sum2 = int(np.sum(flat * indices, dtype=np.uint64) % 65535)
        return (sum2 << 16) | sum1

    def compute_seu_rate(self, uptime_s: float) -> float:
        """
        Compute normalized SEU rate (errors / Megabit / day).
        Standard aerospace metric for space weather radiation impact.
        """
        if uptime_s <= 0.0:
            return 0.0
        days = uptime_s / 86400.0
        total_flips = self.single_bit_errors + self.multi_bit_errors
        rate = total_flips / (self.monitored_mb * max(1e-4, days))
        return float(rate)

    def get_stats(self, uptime_s: float = 1.0) -> EDACStats:
        """Return current EDAC telemetry metrics."""
        rate = self.compute_seu_rate(uptime_s)
        return EDACStats(
            single_bit_errors=self.single_bit_errors,
            multi_bit_errors=self.multi_bit_errors,
            scrub_cycles=self.scrub_cycles,
            uptime_seconds=uptime_s,
            seu_rate_per_mb_day=rate,
            burst_event_count=self.burst_event_count,
            last_error_timestamp=self.last_error_time,
            severity=self._severity,
        )


# ---------------------------------------------------------------------------
# Subsystem 2: Orbital Thermal Cycling Monitor
# ---------------------------------------------------------------------------

class ThermalCycleMonitor:
    """
    Monitors LEO orbital thermal cycles (daylight/eclipse transitions),
    evaluates rate-of-change (dT/dt) for thermal runaway detection, and
    calculates accumulated low-cycle fatigue damage via Coffin-Manson equation:
        D = sum( (Delta_T_i / Delta_T_ref)^m / N_ref )
    """

    def __init__(
        self,
        t_cold_warning_c: float = -25.0,
        t_max_nominal_c: float = 65.0,
        t_throttle_c: float = 75.0,
        t_critical_c: float = 85.0,
        cycle_band_low_c: float = 10.0,
        cycle_band_high_c: float = 55.0,
        coffin_manson_m: float = 2.3,
        reference_fatigue_cycles: float = 40000.0,
        reference_delta_t: float = 50.0,
    ):
        self.t_cold_warning = float(t_cold_warning_c)
        self.t_max_nominal = float(t_max_nominal_c)
        self.t_throttle = float(t_throttle_c)
        self.t_critical = float(t_critical_c)

        # Hysteresis bands for detecting complete LEO day/night thermal cycles
        self.cycle_band_low = float(cycle_band_low_c)
        self.cycle_band_high = float(cycle_band_high_c)

        # Coffin-Manson parameters for space electronic solder interconnects (MIL-STD-883)
        self.cm_m = float(coffin_manson_m)
        self.ref_cycles = max(1.0, float(reference_fatigue_cycles))
        self.ref_delta_t = max(1.0, float(reference_delta_t))

        self.last_temp_c: float = 25.0
        self.last_time_s: Optional[float] = None
        self.min_temp_c: float = 25.0
        self.max_temp_c: float = 25.0
        self.thermal_rate_c_s: float = 0.0

        # Cycle counting state machine
        self._in_cooling_phase: bool = False
        self._cycle_peak: float = 25.0
        self._cycle_valley: float = 25.0
        self.completed_cycles: int = 0
        self.current_amplitude: float = 0.0
        self.accumulated_damage: float = 0.0

    def update(
        self,
        die_temp_c: float,
        board_temp_c: Optional[float] = None,
        timestamp: Optional[float] = None,
    ) -> ThermalZone:
        """
        Ingest new thermal sensor readings, calculate gradients, track cycles,
        and determine the thermal classification zone.
        """
        t = time.time() if timestamp is None else float(timestamp)
        temp = float(die_temp_c)

        # Update min / max envelope
        self.min_temp_c = min(self.min_temp_c, temp)
        self.max_temp_c = max(self.max_temp_c, temp)

        # Compute dT/dt rate of change
        if self.last_time_s is not None and t > self.last_time_s:
            dt = t - self.last_time_s
            self.thermal_rate_c_s = (temp - self.last_temp_c) / dt
        else:
            self.thermal_rate_c_s = 0.0

        # Thermal cycle detection with hysteresis
        if temp > self._cycle_peak:
            self._cycle_peak = temp
        if temp < self._cycle_valley:
            self._cycle_valley = temp

        # State transition: cooling -> heating
        if not self._in_cooling_phase and self.thermal_rate_c_s < -0.01 and temp <= self.cycle_band_low:
            self._in_cooling_phase = True
            self._cycle_valley = temp

        # Completed cycle transition: heating back past high threshold
        elif self._in_cooling_phase and temp >= self.cycle_band_high:
            self._in_cooling_phase = False
            self.completed_cycles += 1
            delta_t = max(0.0, self._cycle_peak - self._cycle_valley)
            self.current_amplitude = delta_t

            # Coffin-Manson low-cycle fatigue damage increment:
            # dD = (Delta_T / Delta_T_ref)^m / N_ref
            damage_increment = ((delta_t / self.ref_delta_t) ** self.cm_m) / self.ref_cycles
            self.accumulated_damage += damage_increment

            # Reset peak/valley tracking for the next cycle
            self._cycle_peak = temp
            self._cycle_valley = temp

        self.last_temp_c = temp
        self.last_time_s = t

        # Classify thermal zone
        return self.get_zone(temp)

    def get_zone(self, temp: float) -> ThermalZone:
        """Classify temperature into operational safety zones."""
        if temp > self.t_critical:
            return ThermalZone.OVERTEMP_CRITICAL
        elif temp > self.t_throttle:
            return ThermalZone.THROTTLED
        elif temp > self.t_max_nominal:
            return ThermalZone.ELEVATED
        elif temp < self.t_cold_warning:
            return ThermalZone.COLD_WARNING
        return ThermalZone.NOMINAL

    def get_stats(self, board_temp_c: float = 20.0) -> ThermalStats:
        """Produce comprehensive thermal status snapshot."""
        zone = self.get_zone(self.last_temp_c)
        fatigue_pct = min(100.0, self.accumulated_damage * 100.0)
        return ThermalStats(
            die_temp_c=self.last_temp_c,
            board_temp_c=float(board_temp_c),
            min_temp_c=self.min_temp_c,
            max_temp_c=self.max_temp_c,
            thermal_rate_c_s=self.thermal_rate_c_s,
            completed_thermal_cycles=self.completed_cycles,
            current_cycle_amplitude=self.current_amplitude,
            accumulated_fatigue_damage=self.accumulated_damage,
            fatigue_life_consumed_pct=fatigue_pct,
            zone=zone,
        )


# ---------------------------------------------------------------------------
# Subsystem 3: Core Clock Throttling (DVFS) Controller
# ---------------------------------------------------------------------------

class CoreClockThrottler:
    """
    Manages Dynamic Voltage and Frequency Scaling (DVFS) for space OBC cores.
    Throttles frequency under high thermal load, radiation storms, or power brownouts
    to preserve flight computer stability, while maintaining minimum flight loop deadlines.
    """

    def __init__(
        self,
        nominal_clock_mhz: float = 400.0,
        throttle_steps_mhz: Tuple[float, ...] = (300.0, 200.0, 100.0, 50.0),
        recovery_dwell_s: float = 10.0,
    ):
        self.nominal_clock = float(nominal_clock_mhz)
        self.steps = sorted([float(s) for s in throttle_steps_mhz], reverse=True)
        self.min_clock = self.steps[-1] if self.steps else 50.0
        self.recovery_dwell = float(recovery_dwell_s)

        self.current_clock = self.nominal_clock
        self.reason = ThrottleReason.NONE
        self.last_throttle_time: Optional[float] = None
        self.total_throttled_time_s: float = 0.0
        self.uptime_s: float = 0.0

    def evaluate(
        self,
        thermal_zone: ThermalZone,
        seu_severity: SEUSeverity = SEUSeverity.NONE,
        power_brownout: bool = False,
        uptime_s: float = 0.0,
        dt_s: float = 1.0,
    ) -> float:
        """
        Evaluate flight environmental stresses and compute optimal core clock.
        Applies hysteresis dwell time to prevent clock hunting/chattering.
        """
        self.uptime_s = max(self.uptime_s, float(uptime_s))
        dt = max(0.0, float(dt_s))

        # Check conditions demanding throttling
        target_clock = self.nominal_clock
        active_reason = ThrottleReason.NONE

        if thermal_zone == ThermalZone.OVERTEMP_CRITICAL:
            target_clock = self.min_clock
            active_reason = ThrottleReason.THERMAL_LIMIT
        elif power_brownout:
            target_clock = self.min_clock
            active_reason = ThrottleReason.POWER_BROWNOUT
        elif thermal_zone == ThermalZone.THROTTLED:
            target_clock = 200.0 if 200.0 in self.steps else self.min_clock
            active_reason = ThrottleReason.THERMAL_LIMIT
        elif seu_severity == SEUSeverity.BURST_FLIP_STORM:
            # Throttling clock during radiation storm reduces register latch setup/hold vulnerability
            target_clock = 200.0 if 200.0 in self.steps else self.min_clock
            active_reason = ThrottleReason.RADIATION_STORM
        elif thermal_zone == ThermalZone.ELEVATED:
            target_clock = 300.0 if 300.0 in self.steps else self.min_clock
            active_reason = ThrottleReason.THERMAL_LIMIT

        # Apply throttling or hysteresis recovery
        if target_clock < self.nominal_clock:
            self.current_clock = target_clock
            self.reason = active_reason
            self.last_throttle_time = self.uptime_s
            self.total_throttled_time_s += dt
        else:
            # Recovery hysteresis: ensure dwell time has passed since last throttle event
            can_recover = (
                self.last_throttle_time is None
                or (self.uptime_s - self.last_throttle_time) >= self.recovery_dwell
            )
            if can_recover:
                self.current_clock = self.nominal_clock
                self.reason = ThrottleReason.NONE
            else:
                # Still within recovery dwell window
                if self.current_clock < self.nominal_clock:
                    self.total_throttled_time_s += dt

        return self.current_clock

    def force_throttle(self, clock_mhz: float, reason: ThrottleReason) -> None:
        """Force manual throttle setting (e.g., ground operator command)."""
        self.current_clock = float(np.clip(clock_mhz, self.min_clock, self.nominal_clock))
        self.reason = reason

    def get_stats(self) -> ClockThrottleStats:
        """Return clock throttling telemetry."""
        is_th = self.current_clock < self.nominal_clock
        ratio = self.current_clock / max(1.0, self.nominal_clock)
        duty = (self.total_throttled_time_s / max(1.0, self.uptime_s)) * 100.0
        return ClockThrottleStats(
            current_clock_mhz=self.current_clock,
            nominal_clock_mhz=self.nominal_clock,
            throttle_ratio=ratio,
            is_throttled=is_th,
            throttle_reason=self.reason,
            total_throttled_time_s=self.total_throttled_time_s,
            duty_cycle_pct=min(100.0, duty),
        )


# ---------------------------------------------------------------------------
# Subsystem 4: INT8 Inference Memory Bounds & Canary Guard Guard
# ---------------------------------------------------------------------------

@dataclass
class MemoryBuffer:
    name: str
    size_bytes: int
    buffer_type: str
    guard_pattern: bytes
    allocated_at: float


class INT8InferenceMemoryGuard:
    """
    Enforces strict static memory ceilings and canary guard-bands for INT8 neural inference
    on flight computers. Analyzes activation dynamic range and saturation bounds to prevent
    quantization overflow and detect SEU-induced weight tensor corruption.
    """

    CANARY_BYTE = 0xA5
    GUARD_BAND_SIZE = 16  # 16 bytes before and after buffer

    def __init__(
        self,
        max_budget_bytes: int = 1048576,  # 1 MB total budget for weights + activations
        max_saturation_allowed: float = 0.15, # Max 15% clipping allowed before alert
    ):
        self.max_budget = int(max_budget_bytes)
        self.max_saturation = float(max_saturation_allowed)

        self.buffers: Dict[str, MemoryBuffer] = {}
        self.allocated_bytes: int = 0
        self.peak_allocated_bytes: int = 0
        self.corrupted_canary_count: int = 0
        self.last_saturation_rate: float = 0.0
        self.is_out_of_bounds: bool = False

    def allocate(
        self,
        name: str,
        size_bytes: int,
        buffer_type: str = "activation",
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Allocate memory buffer with canary guard bands.
        Returns True if allocation fits within hard budget, False otherwise.
        """
        t = time.time() if timestamp is None else float(timestamp)
        size = max(0, int(size_bytes))
        total_with_guard = size + 2 * self.GUARD_BAND_SIZE

        # Free existing buffer with same name if already present
        if name in self.buffers:
            self.free(name)

        if self.allocated_bytes + total_with_guard > self.max_budget:
            self.is_out_of_bounds = True
            return False

        pattern = bytes([self.CANARY_BYTE] * self.GUARD_BAND_SIZE)
        self.buffers[name] = MemoryBuffer(
            name=name,
            size_bytes=size,
            buffer_type=buffer_type,
            guard_pattern=pattern,
            allocated_at=t,
        )
        self.allocated_bytes += total_with_guard
        self.peak_allocated_bytes = max(self.peak_allocated_bytes, self.allocated_bytes)
        self.is_out_of_bounds = False
        return True

    def free(self, name: str) -> bool:
        """Deallocate a buffer and reclaim memory."""
        if name in self.buffers:
            buf = self.buffers.pop(name)
            self.allocated_bytes -= (buf.size_bytes + 2 * self.GUARD_BAND_SIZE)
            self.allocated_bytes = max(0, self.allocated_bytes)
            return True
        return False

    def flush_scratchpad(self) -> int:
        """
        Flush temporary activation/intermediate scratchpad buffers,
        retaining static model weights in memory.
        """
        to_remove = [k for k, v in self.buffers.items() if v.buffer_type == "activation"]
        freed = 0
        for k in to_remove:
            freed += self.buffers[k].size_bytes
            self.free(k)
        return freed

    def corrupt_canary_for_testing(self, name: str) -> None:
        """Simulate a buffer overrun corruption of a guard band."""
        if name in self.buffers:
            self.buffers[name].guard_pattern = b"\x00" * self.GUARD_BAND_SIZE

    def verify_guard_bands(self) -> Tuple[bool, List[str]]:
        """
        Scan all active memory buffers and verify canary guard-band patterns.
        Detects heap buffer overruns or scribbled memory.
        """
        corrupted = []
        expected = bytes([self.CANARY_BYTE] * self.GUARD_BAND_SIZE)
        for name, buf in self.buffers.items():
            if buf.guard_pattern != expected:
                corrupted.append(name)

        self.corrupted_canary_count = len(corrupted)
        is_intact = (len(corrupted) == 0)
        return is_intact, corrupted

    def validate_int8_tensor(
        self,
        tensor: np.ndarray,
        tensor_name: str = "tensor",
    ) -> Dict[str, Any]:
        """
        Analyze INT8 activation or weight tensor dynamic range bounds.
        Checks for:
        1. Numerical saturation at INT8 boundaries (-128 or +127).
        2. Non-finite values (NaN / Inf).
        3. Excessive clipping indicating quantization breakdown or SEU corruption.
        """
        arr = np.asarray(tensor)

        # Check finite
        if not np.all(np.isfinite(arr)):
            return {
                "tensor_name": tensor_name,
                "valid": False,
                "error": "NON_FINITE_VALUES_DETECTED",
                "saturation_rate": 1.0,
            }

        total_elements = max(1, arr.size)

        # Calculate INT8 saturation count
        # In INT8, clipping occurs at -128 and +127
        if np.issubdtype(arr.dtype, np.integer):
            neg_sat = np.sum(arr <= -128)
            pos_sat = np.sum(arr >= 127)
        else:
            # Simulated INT8 cast check
            neg_sat = np.sum(arr <= -128.0)
            pos_sat = np.sum(arr >= 127.0)

        sat_count = int(neg_sat + pos_sat)
        sat_rate = float(sat_count / total_elements)
        self.last_saturation_rate = sat_rate

        is_saturated = sat_rate > self.max_saturation
        return {
            "tensor_name": tensor_name,
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "element_count": total_elements,
            "min_val": float(np.min(arr)),
            "max_val": float(np.max(arr)),
            "saturated_elements": sat_count,
            "saturation_rate": round(sat_rate, 4),
            "exceeds_clipping_threshold": bool(is_saturated),
            "valid": not is_saturated,
        }

    def get_stats(self) -> INT8MemoryStats:
        """Produce memory subsystem telemetry snapshot."""
        usage_pct = (self.allocated_bytes / max(1, self.max_budget)) * 100.0
        intact, _ = self.verify_guard_bands()
        return INT8MemoryStats(
            allocated_bytes=self.allocated_bytes,
            budget_bytes=self.max_budget,
            usage_pct=min(100.0, usage_pct),
            peak_allocated_bytes=self.peak_allocated_bytes,
            active_buffers_count=len(self.buffers),
            guard_bands_intact=intact,
            corrupted_guard_bands=self.corrupted_canary_count,
            activation_saturation_rate=self.last_saturation_rate,
            is_out_of_bounds=self.is_out_of_bounds,
        )


# ---------------------------------------------------------------------------
# Master Class: Space-Grade OBC Hardware Watchdog
# ---------------------------------------------------------------------------

class OBCHardwareWatchdog:
    """
    Top-level space-grade flight computer hardware watchdog supervisor.

    Integrates:
    - Windowed hardware watchdog heartbeat (detects CPU lockup and runaway loops).
    - SEU bit-flip monitoring and EDAC error rate tracking.
    - Orbital thermal cycling and Coffin-Manson solder fatigue estimation.
    - Dynamic Voltage & Frequency Scaling (DVFS) core clock throttling.
    - INT8 neural network memory bounds, canary guard-bands, and tensor saturation.
    """

    def __init__(
        self,
        nominal_clock_mhz: float = 400.0,
        memory_budget_bytes: int = 1048576,  # 1 MB
        monitored_memory_mb: float = 16.0,
        window_min_s: float = 0.01,          # 10 ms min kick period (anti-runaway)
        window_max_s: float = 8.0,           # 8 s max kick period (anti-hang)
    ):
        self.window_min = float(window_min_s)
        self.window_max = float(window_max_s)

        # Subsystems
        self.seu = SEUWatchdog(monitored_memory_mb=monitored_memory_mb)
        self.thermal = ThermalCycleMonitor()
        self.throttler = CoreClockThrottler(nominal_clock_mhz=nominal_clock_mhz)
        self.memory = INT8InferenceMemoryGuard(max_budget_bytes=memory_budget_bytes)

        # Watchdog timing state
        self.heartbeat_counter: int = 0
        self.last_kick_time: Optional[float] = None
        self.window_valid: bool = True
        self.uptime_s: float = 0.0
        self.state: WatchdogState = WatchdogState.NOMINAL
        self.alarms: List[str] = []

    def kick(self, timestamp: Optional[float] = None) -> bool:
        """
        Service the windowed watchdog timer (heartbeat pulse).
        In space avionics, kicking too early implies runaway execution loop;
        kicking too late implies CPU hang / deadlock.
        Returns True if heartbeat falls within the valid timing window [min, max].
        """
        t = time.time() if timestamp is None else float(timestamp)
        self.heartbeat_counter += 1

        if self.last_kick_time is None:
            self.last_kick_time = t
            self.window_valid = True
            return True

        elapsed = t - self.last_kick_time
        self.last_kick_time = t

        # Check windowed timing constraints
        if elapsed < self.window_min:
            self.window_valid = False
            self.alarms.append(f"WATCHDOG_EARLY_KICK_VIOLATION ({elapsed*1000.0:.2f}ms < {self.window_min*1000.0:.2f}ms)")
            return False
        elif elapsed > self.window_max:
            self.window_valid = False
            self.alarms.append(f"WATCHDOG_TIMEOUT_EXPIRED ({elapsed:.2f}s > {self.window_max:.2f}s)")
            return False

        self.window_valid = True
        return True

    def process_telemetry_frame(
        self,
        die_temp_c: float,
        board_temp_c: float = 22.0,
        single_bit_flips: int = 0,
        multi_bit_flips: int = 0,
        power_brownout: bool = False,
        uptime_s: Optional[float] = None,
        activation_tensor: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> WatchdogReport:
        """
        Ingest a comprehensive flight computer telemetry frame, evaluate health
        across all subsystems, and determine autonomous mitigation orders.
        """
        t = time.time() if timestamp is None else float(timestamp)
        if uptime_s is not None:
            dt = max(0.0, uptime_s - self.uptime_s)
            self.uptime_s = float(uptime_s)
        else:
            dt = 1.0
            self.uptime_s += dt

        self.alarms.clear()

        # 1. Update SEU / EDAC
        seu_sev = self.seu.record_edac_event(
            single_bit_flips=single_bit_flips,
            multi_bit_flips=multi_bit_flips,
            timestamp=t,
        )

        # 2. Update Thermal Monitor
        thermal_zone = self.thermal.update(
            die_temp_c=die_temp_c,
            board_temp_c=board_temp_c,
            timestamp=t,
        )

        # 3. Update Clock Throttler
        self.throttler.evaluate(
            thermal_zone=thermal_zone,
            seu_severity=seu_sev,
            power_brownout=power_brownout,
            uptime_s=self.uptime_s,
            dt_s=dt,
        )

        # 4. Update INT8 Memory Tensor Checks
        if activation_tensor is not None:
            val = self.memory.validate_int8_tensor(activation_tensor)
            if not val["valid"]:
                self.alarms.append(f"INT8_TENSOR_SATURATION_WARNING ({val.get('saturation_rate', 0.0):.2%})")

        # Check guard band integrity
        guard_intact, corrupted_names = self.memory.verify_guard_bands()
        if not guard_intact:
            self.alarms.append(f"GUARD_BAND_CANARY_CORRUPTED ({', '.join(corrupted_names)})")

        if self.memory.is_out_of_bounds:
            self.alarms.append("INT8_MEMORY_CEILING_EXCEEDED")

        # 5. Check Watchdog Timer Window Expiration
        time_since_kick = 0.0
        if self.last_kick_time is not None:
            time_since_kick = max(0.0, t - self.last_kick_time)
            if time_since_kick > self.window_max:
                self.window_valid = False
                self.alarms.append(f"WATCHDOG_HEARTBEAT_TIMEOUT ({time_since_kick:.2f}s)")

        # Compile subsystem stats
        edac_stats = self.seu.get_stats(self.uptime_s)
        thermal_stats = self.thermal.get_stats(board_temp_c)
        clock_stats = self.throttler.get_stats()
        mem_stats = self.memory.get_stats()

        # Add alarms from subsystem events
        if seu_sev == SEUSeverity.UNCORRECTABLE_MULTI_BIT:
            self.alarms.append("EDAC_UNCORRECTABLE_DOUBLE_BIT_ERROR")
        elif seu_sev == SEUSeverity.BURST_FLIP_STORM:
            self.alarms.append("SEU_RADIATION_BURST_STORM_ACTIVE")

        if thermal_zone == ThermalZone.OVERTEMP_CRITICAL:
            self.alarms.append(f"OVERTEMP_CRITICAL ({die_temp_c:.1f}C > {self.thermal.t_critical}C)")
        elif thermal_zone == ThermalZone.THROTTLED:
            self.alarms.append(f"THERMAL_THROTTLE_ENGAGED ({die_temp_c:.1f}C)")

        if power_brownout:
            self.alarms.append("POWER_STAGE_BROWNOUT_DETECTED")

        # 6. Determine Watchdog State & Mitigation Action
        action = MitigationAction.NO_ACTION

        # Critical / Fatal conditions -> SAFE_MODE or RESET
        if (
            not self.window_valid
            or seu_sev == SEUSeverity.UNCORRECTABLE_MULTI_BIT
            or not guard_intact
        ):
            if not self.window_valid:
                self.state = WatchdogState.RESET_TRIGGERED
                action = MitigationAction.FORCE_HARDWARE_RESET
            else:
                self.state = WatchdogState.CRITICAL
                action = MitigationAction.ENTER_SAFE_MODE

        elif thermal_zone == ThermalZone.OVERTEMP_CRITICAL or power_brownout or self.memory.is_out_of_bounds:
            self.state = WatchdogState.CRITICAL
            if self.memory.is_out_of_bounds:
                action = MitigationAction.FLUSH_INT8_SCRATCHPAD
            else:
                action = MitigationAction.ENTER_SAFE_MODE

        elif (
            clock_stats.is_throttled
            or thermal_zone in (ThermalZone.THROTTLED, ThermalZone.ELEVATED)
            or seu_sev in (SEUSeverity.BURST_FLIP_STORM, SEUSeverity.CORRECTABLE_SINGLE_BIT)
            or mem_stats.usage_pct > 85.0
        ):
            self.state = WatchdogState.WARNING
            if seu_sev == SEUSeverity.BURST_FLIP_STORM or edac_stats.single_bit_errors > 5:
                action = MitigationAction.TRIGGER_MEMORY_SCRUB
            elif thermal_zone in (ThermalZone.THROTTLED, ThermalZone.ELEVATED):
                action = MitigationAction.THROTTLE_CLOCK
            elif mem_stats.usage_pct > 85.0:
                action = MitigationAction.FLUSH_INT8_SCRATCHPAD
            else:
                action = MitigationAction.NO_ACTION
        else:
            self.state = WatchdogState.NOMINAL
            action = MitigationAction.NO_ACTION

        is_healthy = self.state in (WatchdogState.NOMINAL, WatchdogState.WARNING)

        return WatchdogReport(
            timestamp=t,
            state=self.state,
            is_healthy=is_healthy,
            heartbeat_counter=self.heartbeat_counter,
            time_since_last_kick_s=time_since_kick,
            window_valid=self.window_valid,
            edac=edac_stats,
            thermal=thermal_stats,
            clock=clock_stats,
            memory=mem_stats,
            active_alarms=self.alarms,
            recommended_action=action,
        )

    def execute_autonomous_recovery(self) -> Dict[str, Any]:
        """
        Execute autonomous recovery protocols based on current watchdog state:
        - Memory scrub for SEU errors.
        - Scratchpad flush for memory pressure.
        - Safe-mode / Clock throttle for thermal stress.
        """
        recovery_log = []
        if self.state in (WatchdogState.WARNING, WatchdogState.CRITICAL):
            # 1. Flush scratchpad
            freed = self.memory.flush_scratchpad()
            if freed > 0:
                recovery_log.append(f"Flushed {freed} bytes of scratchpad memory.")

            # 2. Trigger EDAC memory scrub
            scrub_res = self.seu.perform_memory_scrub()
            recovery_log.append(f"Executed EDAC memory scrub cycle #{scrub_res['scrub_cycle']}.")

            # 3. Throttle clock
            if self.thermal.last_temp_c > self.thermal.t_max_nominal:
                self.throttler.force_throttle(200.0, ThrottleReason.THERMAL_LIMIT)
                recovery_log.append("Forced DVFS core clock down to 200 MHz.")

        return {
            "initial_state": self.state.value,
            "recovery_actions_taken": recovery_log,
            "success": len(recovery_log) > 0,
        }
