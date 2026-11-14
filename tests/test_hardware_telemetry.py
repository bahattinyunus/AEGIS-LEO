"""
Unit and Integration Tests for Space-Grade OBC Hardware Watchdog
===============================================================
Tests:
- SEU bit-flip & EDAC memory scrubbing monitor
- Orbital thermal cycling & Coffin-Manson fatigue estimation
- Dynamic Voltage & Frequency Scaling (DVFS) core clock throttling
- INT8 inference memory bounds, canary guard bands, and tensor saturation
- Windowed hardware watchdog timing violations and autonomous recovery
"""

import pytest
import numpy as np
import time
from aegis_leo.ai.hardware_telemetry import (
    OBCHardwareWatchdog,
    SEUWatchdog,
    ThermalCycleMonitor,
    CoreClockThrottler,
    INT8InferenceMemoryGuard,
    WatchdogState,
    ThermalZone,
    SEUSeverity,
    ThrottleReason,
    MitigationAction,
)


def test_seu_watchdog_edac_and_scrubbing():
    """Verify single-bit error correction, multi-bit detection, and rate calculation."""
    seu = SEUWatchdog(monitored_memory_mb=16.0, burst_time_window_s=5.0, burst_threshold_flips=5)

    # 1. Single bit flip
    sev = seu.record_edac_event(single_bit_flips=2, multi_bit_flips=0, timestamp=100.0)
    assert sev == SEUSeverity.CORRECTABLE_SINGLE_BIT
    assert seu.single_bit_errors == 2

    # 2. Rate computation (errors / MB / day)
    # 2 flips in 16 MB over 86400 seconds (1 day) -> 2 / 16 = 0.125 flips/MB/day
    rate = seu.compute_seu_rate(uptime_s=86400.0)
    assert np.isclose(rate, 0.125, atol=1e-3)

    # 3. Burst flip storm detection
    sev = seu.record_edac_event(single_bit_flips=4, multi_bit_flips=0, timestamp=102.0)
    # 2 + 4 = 6 flips within 2 seconds (> threshold 5)
    assert sev == SEUSeverity.BURST_FLIP_STORM
    assert seu.burst_event_count >= 1

    # 4. Multi-bit uncorrectable error (DED)
    sev = seu.record_edac_event(single_bit_flips=0, multi_bit_flips=1, timestamp=103.0)
    assert sev == SEUSeverity.UNCORRECTABLE_MULTI_BIT
    assert seu.multi_bit_errors == 1

    # 5. Scrubbing cycle
    scrub = seu.perform_memory_scrub(scrub_bytes=2 * 1024 * 1024, simulated_corruptions=3, timestamp=104.0)
    assert scrub["scrub_cycle"] == 1
    assert scrub["megabytes_scrubbed"] == 2.0
    assert scrub["flips_repaired"] == 3


def test_block_checksum_verification():
    """Verify array Fletcher/CRC checksum consistency and mutation sensitivity."""
    data = np.array([10, 20, 30, 40, 50], dtype=np.int8)
    chk1 = SEUWatchdog.compute_block_checksum(data)
    chk2 = SEUWatchdog.compute_block_checksum(data)
    assert chk1 == chk2

    # Mutate data (SEU bit-flip)
    mutated = data.copy()
    mutated[2] ^= 0x01
    chk_mutated = SEUWatchdog.compute_block_checksum(mutated)
    assert chk1 != chk_mutated


def test_thermal_cycle_monitoring_and_fatigue():
    """Verify LEO orbital thermal hysteresis cycle counting and Coffin-Manson fatigue."""
    monitor = ThermalCycleMonitor(
        t_cold_warning_c=-25.0,
        t_max_nominal_c=65.0,
        t_throttle_c=75.0,
        t_critical_c=85.0,
        cycle_band_low_c=10.0,
        cycle_band_high_c=50.0,
        coffin_manson_m=2.0,
        reference_fatigue_cycles=10000.0,
        reference_delta_t=50.0,
    )

    # Initial state
    assert monitor.get_zone(25.0) == ThermalZone.NOMINAL
    assert monitor.get_zone(70.0) == ThermalZone.ELEVATED
    assert monitor.get_zone(80.0) == ThermalZone.THROTTLED
    assert monitor.get_zone(90.0) == ThermalZone.OVERTEMP_CRITICAL
    assert monitor.get_zone(-30.0) == ThermalZone.COLD_WARNING

    # Simulate an orbit cycle:
    # 1. Warm initial in sunlight: 55°C at t=0
    monitor.update(die_temp_c=55.0, timestamp=0.0)

    # 2. Eclipse cooling down to 5°C at t=1800s
    monitor.update(die_temp_c=5.0, timestamp=1800.0)
    assert monitor._in_cooling_phase is True
    assert monitor.thermal_rate_c_s < 0.0

    # 3. Sunlight heating back to 55°C at t=3600s
    zone = monitor.update(die_temp_c=55.0, timestamp=3600.0)
    assert monitor.completed_cycles == 1
    assert monitor.current_amplitude >= 45.0  # (55 - 5 = 50)
    assert monitor.accumulated_damage > 0.0

    stats = monitor.get_stats(board_temp_c=22.0)
    assert stats.completed_thermal_cycles == 1
    assert stats.fatigue_life_consumed_pct > 0.0


def test_core_clock_throttler_dvfs():
    """Verify DVFS governor clock adjustments under thermal and radiation stresses."""
    throttler = CoreClockThrottler(nominal_clock_mhz=400.0, recovery_dwell_s=5.0)

    # 1. Nominal conditions -> 400 MHz
    clk = throttler.evaluate(thermal_zone=ThermalZone.NOMINAL, uptime_s=10.0, dt_s=1.0)
    assert clk == 400.0
    assert not throttler.get_stats().is_throttled

    # 2. Elevated thermal -> throttles to 300 MHz
    clk = throttler.evaluate(thermal_zone=ThermalZone.ELEVATED, uptime_s=11.0, dt_s=1.0)
    assert clk == 300.0
    stats = throttler.get_stats()
    assert stats.is_throttled
    assert stats.throttle_reason == ThrottleReason.THERMAL_LIMIT

    # 3. Critical overtemp -> drops to minimum (50 MHz)
    clk = throttler.evaluate(thermal_zone=ThermalZone.OVERTEMP_CRITICAL, uptime_s=12.0, dt_s=1.0)
    assert clk == 50.0

    # 4. Temperature cools to nominal, but dwell time must elapse before recovery
    clk = throttler.evaluate(thermal_zone=ThermalZone.NOMINAL, uptime_s=14.0, dt_s=2.0)
    # Elapsed 2s < recovery_dwell (5s), remains throttled
    assert clk == 50.0

    # 5. Dwell time passes (12 + 6 = 18s) -> recovers to 400 MHz
    clk = throttler.evaluate(thermal_zone=ThermalZone.NOMINAL, uptime_s=18.0, dt_s=4.0)
    assert clk == 400.0
    assert not throttler.get_stats().is_throttled


def test_int8_memory_guard_and_canary():
    """Verify INT8 allocation bounds, guard band canary checking, and saturation analysis."""
    # 100 KB budget
    guard = INT8InferenceMemoryGuard(max_budget_bytes=100 * 1024, max_saturation_allowed=0.10)

    # 1. Successful allocation within budget
    success = guard.allocate("weights_layer1", size_bytes=30 * 1024, buffer_type="weight")
    assert success is True
    assert guard.allocated_bytes > 30 * 1024

    # 2. Allocation exceeding budget -> rejected
    success = guard.allocate("huge_buffer", size_bytes=80 * 1024, buffer_type="activation")
    assert success is False
    assert guard.is_out_of_bounds is True

    # 3. Guard band verification (intact)
    intact, corrupted = guard.verify_guard_bands()
    assert intact is True
    assert len(corrupted) == 0

    # 4. Canary corruption simulation (buffer overrun)
    guard.corrupt_canary_for_testing("weights_layer1")
    intact, corrupted = guard.verify_guard_bands()
    assert intact is False
    assert "weights_layer1" in corrupted

    # 5. Scratchpad flushing
    guard.allocate("scratchpad_act", size_bytes=10 * 1024, buffer_type="activation")
    freed = guard.flush_scratchpad()
    assert freed == 10 * 1024
    assert "scratchpad_act" not in guard.buffers

    # 6. Tensor saturation analysis
    # Normal distribution
    normal_tensor = np.array([-50, 0, 50, 80], dtype=np.int8)
    val = guard.validate_int8_tensor(normal_tensor)
    assert val["valid"] is True
    assert val["saturation_rate"] == 0.0

    # Heavily clipped / saturated tensor
    saturated_tensor = np.array([-128, -128, 127, 127, 0], dtype=np.int8)
    val = guard.validate_int8_tensor(saturated_tensor)
    # 4 out of 5 elements saturated = 80% > 10% threshold
    assert val["valid"] is False
    assert val["exceeds_clipping_threshold"] is True
    assert val["saturation_rate"] == 0.8


def test_obc_hardware_watchdog_master_integration():
    """Verify integrated master watchdog, heartbeat windowing, and autonomous triage."""
    wd = OBCHardwareWatchdog(
        nominal_clock_mhz=400.0,
        memory_budget_bytes=512 * 1024,
        monitored_memory_mb=16.0,
        window_min_s=0.05,
        window_max_s=5.0,
    )

    # 1. Initial heartbeat kick
    assert wd.kick(timestamp=100.0) is True

    # 2. Early kick violation (< 0.05s)
    bad_kick = wd.kick(timestamp=100.02)
    assert bad_kick is False
    assert not wd.window_valid

    # Reset kick to nominal timing
    wd.kick(timestamp=101.0)
    assert wd.window_valid is True

    # 3. Nominal telemetry frame
    rep = wd.process_telemetry_frame(
        die_temp_c=40.0,
        board_temp_c=25.0,
        single_bit_flips=0,
        multi_bit_flips=0,
        power_brownout=False,
        uptime_s=10.0,
        timestamp=101.5,
    )
    assert rep.state == WatchdogState.NOMINAL
    assert rep.is_healthy is True
    assert rep.recommended_action == MitigationAction.NO_ACTION

    # 4. Elevated temp & SBE flips -> WARNING state
    rep = wd.process_telemetry_frame(
        die_temp_c=70.0,
        board_temp_c=35.0,
        single_bit_flips=2,
        multi_bit_flips=0,
        uptime_s=11.0,
        timestamp=102.0,
    )
    assert rep.state == WatchdogState.WARNING
    assert rep.recommended_action in (MitigationAction.THROTTLE_CLOCK, MitigationAction.TRIGGER_MEMORY_SCRUB)

    # 5. Multi-bit uncorrectable SEU error -> CRITICAL / SAFE_MODE
    rep = wd.process_telemetry_frame(
        die_temp_c=45.0,
        board_temp_c=25.0,
        single_bit_flips=0,
        multi_bit_flips=1,
        uptime_s=12.0,
        timestamp=102.5,
    )
    assert rep.state == WatchdogState.CRITICAL
    assert rep.recommended_action == MitigationAction.ENTER_SAFE_MODE
    assert not rep.is_healthy

    # 6. Autonomous recovery execution
    recovery = wd.execute_autonomous_recovery()
    assert recovery["initial_state"] == "CRITICAL"
    assert recovery["success"] is True

    # 7. Serialization check
    rep_dict = rep.to_dict()
    assert isinstance(rep_dict, dict)
    assert "state" in rep_dict
    assert "edac" in rep_dict
    assert "thermal" in rep_dict
    assert "clock" in rep_dict
    assert "memory" in rep_dict


def test_watchdog_timeout_triggers_reset():
    """Verify watchdog timeout (> window_max) triggers reset vector."""
    wd = OBCHardwareWatchdog(window_min_s=0.01, window_max_s=2.0)
    wd.kick(timestamp=100.0)

    # Process frame 3 seconds later without kicking watchdog
    rep = wd.process_telemetry_frame(
        die_temp_c=35.0,
        timestamp=103.0,
        uptime_s=3.0,
    )
    assert rep.state == WatchdogState.RESET_TRIGGERED
    assert rep.recommended_action == MitigationAction.FORCE_HARDWARE_RESET
    assert "WATCHDOG_HEARTBEAT_TIMEOUT" in " ".join(rep.active_alarms)
