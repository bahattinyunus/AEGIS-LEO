"""
AEGIS-LEO Telemetry Anomaly Detector
=======================================
Real-time statistical and physics-guided anomaly detection for onboard
satellite telemetry streams.

Capabilities:
  1. Z-score sliding-window outlier detection (each telemetry channel).
  2. Mahalanobis distance multivariate anomaly detection (channel cluster).
  3. Physics-consistency checks (orbit energy conservation, power balance,
     thermal equilibrium validation, Tsiolkovsky mass budget).
  4. Change-point detection using CUSUM (Cumulative SUM) control chart.
  5. SEU-induced bit-flip signature detection in PINN output stream.
  6. Composite anomaly severity scoring and priority ranking.

Design principle: all detectors are purely analytic (NumPy / SciPy);
no trained ML model weights are needed, making this suitable for
radiation-hardened onboard execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aegis_leo.constants import (
    EARTH_MU,
    EARTH_RADIUS_KM,
    STANDARD_GRAVITY,
    SOLAR_CONSTANT,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AnomalyCategory(str, Enum):
    STATISTICAL_OUTLIER  = "STATISTICAL_OUTLIER"
    PHYSICS_VIOLATION    = "PHYSICS_VIOLATION"
    CHANGE_POINT         = "CHANGE_POINT"
    SEU_BIT_FLIP         = "SEU_BIT_FLIP"
    MULTIVARIATE_OUTLIER = "MULTIVARIATE_OUTLIER"


class AnomalySeverity(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AnomalyEvent:
    event_id: str
    channel: str
    category: AnomalyCategory
    severity: AnomalySeverity
    detected_value: float
    expected_range: Tuple[float, float]
    deviation_sigma: float
    description: str
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["expected_range"] = list(self.expected_range)
        return d


@dataclass
class AnomalyReport:
    """Complete anomaly detection report for one telemetry epoch."""
    timestamp_utc: str
    satellite_id: str
    epoch_index: int
    total_channels: int
    anomaly_count: int
    critical_count: int
    warning_count: int
    overall_health_score: float     # 0–100 (100 = no anomalies)
    anomalies: List[AnomalyEvent]
    channel_z_scores: Dict[str, float]
    cusum_flags: Dict[str, bool]
    physics_checks: Dict[str, bool]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["anomalies"] = [a.to_dict() for a in self.anomalies]
        return d


# ---------------------------------------------------------------------------
# Sliding-window statistics tracker
# ---------------------------------------------------------------------------

class _ChannelStatTracker:
    """Maintains a rolling mean and standard deviation for one channel."""

    def __init__(self, window: int = 30, min_samples: int = 5):
        self._window    = window
        self._min_samp  = min_samples
        self._buf: List[float] = []
        # CUSUM state
        self._cusum_pos: float = 0.0
        self._cusum_neg: float = 0.0
        self._cusum_k: float   = 1.0    # slack (multiples of σ)
        self._cusum_h: float   = 5.0    # decision threshold (multiples of σ)

    def update(self, value: float) -> Tuple[float, bool]:
        """
        Update with a new sample.

        Returns
        -------
        z_score : float
            Z-score of *value* relative to rolling baseline.
        cusum_alarm : bool
            True if the CUSUM control chart fires.
        """
        self._buf.append(float(value))
        if len(self._buf) > self._window:
            self._buf.pop(0)

        if len(self._buf) < self._min_samp:
            return 0.0, False

        mu  = float(np.mean(self._buf[:-1]))   # exclude current
        std = max(float(np.std(self._buf[:-1])), 1e-9)
        z   = (value - mu) / std

        # CUSUM update
        slack = self._cusum_k * std
        self._cusum_pos = max(0.0, self._cusum_pos + (value - mu) - slack)
        self._cusum_neg = max(0.0, self._cusum_neg - (value - mu) - slack)
        threshold = self._cusum_h * std
        alarm = self._cusum_pos > threshold or self._cusum_neg > threshold
        if alarm:
            self._cusum_pos = 0.0
            self._cusum_neg = 0.0

        return z, alarm

    def mean(self) -> float:
        return float(np.mean(self._buf)) if self._buf else 0.0

    def std(self) -> float:
        return float(np.std(self._buf)) if len(self._buf) > 1 else 1.0


# ---------------------------------------------------------------------------
# Physics consistency checks
# ---------------------------------------------------------------------------

def _check_orbital_energy(
    altitude_km: float,
    velocity_km_s: float,
) -> Tuple[bool, float]:
    """
    Check vis-viva energy conservation.
    Expect |v - v_circ| / v_circ < 5%.
    """
    r_km = EARTH_RADIUS_KM + altitude_km
    v_circ = math.sqrt(EARTH_MU / r_km)
    deviation_pct = abs(velocity_km_s - v_circ) / v_circ * 100.0
    return (deviation_pct < 5.0), round(deviation_pct, 3)


def _check_power_balance(
    solar_array_power_w: float,
    bus_load_w: float,
    eclipse_flag: bool,
    battery_soc_pct: float,
) -> Tuple[bool, str]:
    """
    Verify power budget consistency.
    During eclipse: battery should be discharging (SOC decreasing).
    During sunlight: solar power should exceed or meet bus load.
    """
    if eclipse_flag:
        # Battery SOC should be non-trivially positive
        if battery_soc_pct < 5.0:
            return False, f"Battery SOC={battery_soc_pct:.1f}% critically low during eclipse"
        return True, "ECLIPSE_POWER_OK"
    else:
        if solar_array_power_w < bus_load_w * 0.85:
            return False, (
                f"Solar power {solar_array_power_w:.1f} W < 85% of bus load "
                f"{bus_load_w:.1f} W during sunlit phase"
            )
        return True, "SUNLIT_POWER_OK"


def _check_thermal_range(
    node_name: str,
    temp_c: float,
    low_limit_c: float,
    high_limit_c: float,
) -> Tuple[bool, str]:
    """Check if a thermal node temperature is within operational limits."""
    if temp_c < low_limit_c:
        return False, f"{node_name}: temp {temp_c:.1f}°C below limit {low_limit_c:.1f}°C"
    if temp_c > high_limit_c:
        return False, f"{node_name}: temp {temp_c:.1f}°C above limit {high_limit_c:.1f}°C"
    return True, f"{node_name}: OK ({temp_c:.1f}°C)"


def _check_tsiolkovsky_budget(
    fuel_before_kg: float,
    fuel_after_kg: float,
    mass_kg: float,
    isp_s: float,
    delta_v_m_s: float,
) -> Tuple[bool, float]:
    """
    Verify that the reported fuel consumption matches Tsiolkovsky's equation.
    Tolerates ±10% discrepancy.
    """
    if delta_v_m_s <= 0.0:
        return True, 0.0
    expected_dm = mass_kg * (1.0 - math.exp(-delta_v_m_s / (isp_s * STANDARD_GRAVITY)))
    actual_dm   = max(0.0, fuel_before_kg - fuel_after_kg)
    if expected_dm < 1e-9:
        return True, 0.0
    discrepancy_pct = abs(actual_dm - expected_dm) / expected_dm * 100.0
    return (discrepancy_pct < 10.0), round(discrepancy_pct, 2)


def _detect_seu_bit_flip(
    pinn_output_history: List[float],
    threshold_sigma: float = 4.0,
) -> Tuple[bool, float]:
    """
    Detect SEU-induced bit-flip in PINN output by flagging outputs that
    deviate > threshold_sigma from the rolling median (robust to outliers).
    """
    if len(pinn_output_history) < 3:
        return False, 0.0
    arr = np.array(pinn_output_history)
    med = float(np.median(arr[:-1]))
    mad = max(float(np.median(np.abs(arr[:-1] - med))), 1e-9)
    score = abs(arr[-1] - med) / (1.4826 * mad)   # ~σ for Gaussian
    return (score > threshold_sigma), round(float(score), 3)


# ---------------------------------------------------------------------------
# Multivariate Mahalanobis detector
# ---------------------------------------------------------------------------

def _mahalanobis_distance(x: np.ndarray, history: np.ndarray) -> float:
    """
    Compute Mahalanobis distance of observation *x* from the sample
    distribution defined by *history* (shape N × d).
    Falls back to Euclidean if covariance is ill-conditioned.
    """
    if history.shape[0] < history.shape[1] + 2:
        # Not enough samples for covariance; use scaled Euclidean
        mu  = np.mean(history, axis=0)
        std = np.std(history, axis=0) + 1e-9
        return float(np.linalg.norm((x - mu) / std))

    mu  = np.mean(history, axis=0)
    cov = np.cov(history.T) + np.eye(history.shape[1]) * 1e-6
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return float(np.linalg.norm(x - mu))
    diff = x - mu
    dist = float(np.sqrt(max(0.0, diff @ inv_cov @ diff)))
    return dist


# ---------------------------------------------------------------------------
# Main Anomaly Detector
# ---------------------------------------------------------------------------

class TelemetryAnomalyDetector:
    """
    Stateful real-time telemetry anomaly detector.

    Maintains per-channel sliding-window statistics and applies multiple
    detection strategies to each incoming telemetry frame.

    Parameters
    ----------
    satellite_id : str
    z_threshold : float
        Z-score threshold for single-channel outlier flagging (default 3.5).
    mahal_threshold : float
        Mahalanobis distance threshold for multivariate anomaly (default 4.0).
    window_size : int
        Rolling history window for statistical baselines.
    """

    _CHANNELS = [
        "altitude_km",
        "velocity_km_s",
        "fuel_kg",
        "battery_soc_pct",
        "solar_power_w",
        "bus_load_w",
        "temp_avionics_c",
        "temp_battery_c",
        "kp_index",
        "pinn_density_output",
    ]

    def __init__(
        self,
        satellite_id: str = "SAT-01",
        z_threshold: float = 3.5,
        mahal_threshold: float = 4.0,
        window_size: int = 30,
    ):
        self.satellite_id    = satellite_id
        self.z_threshold     = z_threshold
        self.mahal_threshold = mahal_threshold
        self._trackers: Dict[str, _ChannelStatTracker] = {
            ch: _ChannelStatTracker(window=window_size) for ch in self._CHANNELS
        }
        self._epoch: int = 0
        self._multi_history: List[np.ndarray] = []
        self._pinn_history: List[float] = []

    # ------------------------------------------------------------------
    def ingest_frame(self, frame: Dict[str, Any]) -> AnomalyReport:
        """
        Process one telemetry frame and return a complete AnomalyReport.

        Expected frame keys (all optional, use defaults for missing):
          altitude_km, velocity_km_s, fuel_kg, battery_soc_pct,
          solar_power_w, bus_load_w, temp_avionics_c, temp_battery_c,
          kp_index, pinn_density_output,
          eclipse_flag, isp_s, delta_v_m_s, fuel_kg_before_burn.
        """
        self._epoch += 1
        f = frame

        # --- Extract values with defaults ---
        alt     = float(f.get("altitude_km", 550.0))
        vel     = float(f.get("velocity_km_s", math.sqrt(EARTH_MU / (EARTH_RADIUS_KM + alt))))
        fuel    = float(f.get("fuel_kg", 10.0))
        soc     = float(f.get("battery_soc_pct", 80.0))
        s_pwr   = float(f.get("solar_power_w", 400.0))
        b_load  = float(f.get("bus_load_w", 180.0))
        t_avi   = float(f.get("temp_avionics_c", 25.0))
        t_bat   = float(f.get("temp_battery_c", 15.0))
        kp      = float(f.get("kp_index", 1.5))
        pinn_out = float(f.get("pinn_density_output", 2.4e-13))
        eclipse = bool(f.get("eclipse_flag", False))
        isp     = float(f.get("isp_s", 250.0))
        dv      = float(f.get("delta_v_m_s", 0.0))
        fuel_b  = float(f.get("fuel_kg_before_burn", fuel + 0.001 * dv))
        mass    = float(f.get("mass_kg", 260.0))

        chan_vals = {
            "altitude_km": alt,
            "velocity_km_s": vel,
            "fuel_kg": fuel,
            "battery_soc_pct": soc,
            "solar_power_w": s_pwr,
            "bus_load_w": b_load,
            "temp_avionics_c": t_avi,
            "temp_battery_c": t_bat,
            "kp_index": kp,
            "pinn_density_output": np.log10(max(pinn_out, 1e-25)) + 25.0,  # log-normalised
        }

        anomalies: List[AnomalyEvent] = []
        z_scores: Dict[str, float] = {}
        cusum_flags: Dict[str, bool] = {}

        # ---- 1. Per-channel Z-score & CUSUM ----
        for ch, val in chan_vals.items():
            z, cusum = self._trackers[ch].update(val)
            z_scores[ch] = round(float(z), 3)
            cusum_flags[ch] = cusum

            if abs(z) > self.z_threshold:
                sev = AnomalySeverity.CRITICAL if abs(z) > 5.0 else AnomalySeverity.WARNING
                mu  = self._trackers[ch].mean()
                std = self._trackers[ch].std()
                anomalies.append(AnomalyEvent(
                    event_id=f"STAT-{ch}-{self._epoch:05d}",
                    channel=ch,
                    category=AnomalyCategory.STATISTICAL_OUTLIER,
                    severity=sev,
                    detected_value=round(val, 6),
                    expected_range=(round(mu - 3*std, 4), round(mu + 3*std, 4)),
                    deviation_sigma=round(float(z), 3),
                    description=f"Channel '{ch}': value {val:.4g} deviates {z:.1f}σ from rolling baseline μ={mu:.4g}.",
                ))
            elif cusum:
                anomalies.append(AnomalyEvent(
                    event_id=f"CUSUM-{ch}-{self._epoch:05d}",
                    channel=ch,
                    category=AnomalyCategory.CHANGE_POINT,
                    severity=AnomalySeverity.WARNING,
                    detected_value=round(val, 6),
                    expected_range=(round(self._trackers[ch].mean() - 2*self._trackers[ch].std(), 4),
                                    round(self._trackers[ch].mean() + 2*self._trackers[ch].std(), 4)),
                    deviation_sigma=round(float(z), 3),
                    description=f"CUSUM change-point detected in '{ch}' at epoch {self._epoch}.",
                ))

        # ---- 2. Multivariate Mahalanobis ----
        vec = np.array([chan_vals[ch] for ch in self._CHANNELS])
        self._multi_history.append(vec)
        if len(self._multi_history) > 50:
            self._multi_history.pop(0)
        if len(self._multi_history) >= 10:
            hist_arr = np.array(self._multi_history[:-1])
            md = _mahalanobis_distance(vec, hist_arr)
            if md > self.mahal_threshold:
                sev = AnomalySeverity.CRITICAL if md > 7.0 else AnomalySeverity.WARNING
                anomalies.append(AnomalyEvent(
                    event_id=f"MAHAL-{self._epoch:05d}",
                    channel="ALL_CHANNELS",
                    category=AnomalyCategory.MULTIVARIATE_OUTLIER,
                    severity=sev,
                    detected_value=round(md, 4),
                    expected_range=(0.0, self.mahal_threshold),
                    deviation_sigma=round(md, 4),
                    description=f"Multivariate Mahalanobis distance {md:.2f} exceeds threshold {self.mahal_threshold}. "
                                "Correlated multi-channel anomaly detected.",
                ))

        # ---- 3. Physics consistency ----
        phys_ok: Dict[str, bool] = {}

        vis_viva_ok, dev_pct = _check_orbital_energy(alt, vel)
        phys_ok["vis_viva"] = vis_viva_ok
        if not vis_viva_ok:
            anomalies.append(AnomalyEvent(
                event_id=f"PHYS-VISVIVA-{self._epoch:05d}",
                channel="velocity_km_s",
                category=AnomalyCategory.PHYSICS_VIOLATION,
                severity=AnomalySeverity.WARNING,
                detected_value=round(vel, 4),
                expected_range=(round(math.sqrt(EARTH_MU / (EARTH_RADIUS_KM + alt)) * 0.95, 4),
                                round(math.sqrt(EARTH_MU / (EARTH_RADIUS_KM + alt)) * 1.05, 4)),
                deviation_sigma=dev_pct / 10.0,
                description=f"Vis-viva energy violation: velocity deviation {dev_pct:.1f}% from circular orbit speed.",
            ))

        pwr_ok, pwr_msg = _check_power_balance(s_pwr, b_load, eclipse, soc)
        phys_ok["power_balance"] = pwr_ok
        if not pwr_ok:
            anomalies.append(AnomalyEvent(
                event_id=f"PHYS-POWER-{self._epoch:05d}",
                channel="solar_power_w",
                category=AnomalyCategory.PHYSICS_VIOLATION,
                severity=AnomalySeverity.CRITICAL,
                detected_value=round(s_pwr, 2),
                expected_range=(round(b_load, 2), round(s_pwr * 2.0, 2)),
                deviation_sigma=(b_load - s_pwr) / max(b_load, 1.0) * 10.0,
                description=f"Power balance violation: {pwr_msg}",
            ))

        therm_avi_ok, therm_avi_msg = _check_thermal_range("AVIONICS", t_avi, -20.0, 65.0)
        phys_ok["thermal_avionics"] = therm_avi_ok
        therm_bat_ok, therm_bat_msg = _check_thermal_range("BATTERY", t_bat, -10.0, 45.0)
        phys_ok["thermal_battery"] = therm_bat_ok
        for ok, msg, ch_name in [(therm_avi_ok, therm_avi_msg, "temp_avionics_c"), (therm_bat_ok, therm_bat_msg, "temp_battery_c")]:
            if not ok:
                temp_val = t_avi if "AVIONICS" in msg else t_bat
                anomalies.append(AnomalyEvent(
                    event_id=f"PHYS-THERM-{self._epoch:05d}",
                    channel=ch_name,
                    category=AnomalyCategory.PHYSICS_VIOLATION,
                    severity=AnomalySeverity.CRITICAL,
                    detected_value=round(temp_val, 2),
                    expected_range=(-20.0, 65.0) if "AVIONICS" in msg else (-10.0, 45.0),
                    deviation_sigma=0.0,
                    description=msg,
                ))

        if dv > 0.0:
            tsiol_ok, disc_pct = _check_tsiolkovsky_budget(fuel_b, fuel, mass, isp, dv)
            phys_ok["tsiolkovsky"] = tsiol_ok
            if not tsiol_ok:
                anomalies.append(AnomalyEvent(
                    event_id=f"PHYS-TSIOL-{self._epoch:05d}",
                    channel="fuel_kg",
                    category=AnomalyCategory.PHYSICS_VIOLATION,
                    severity=AnomalySeverity.WARNING,
                    detected_value=round(fuel, 4),
                    expected_range=(round(fuel_b * 0.9, 4), round(fuel_b, 4)),
                    deviation_sigma=disc_pct / 10.0,
                    description=f"Tsiolkovsky mass budget discrepancy {disc_pct:.1f}% — possible fuel sensor drift.",
                ))
        else:
            phys_ok["tsiolkovsky"] = True

        # ---- 4. SEU bit-flip detection in PINN output ----
        self._pinn_history.append(float(np.log10(max(pinn_out, 1e-25))))
        if len(self._pinn_history) > 20:
            self._pinn_history.pop(0)
        seu_detected, seu_sigma = _detect_seu_bit_flip(self._pinn_history)
        phys_ok["seu_check"] = not seu_detected
        if seu_detected:
            anomalies.append(AnomalyEvent(
                event_id=f"SEU-PINN-{self._epoch:05d}",
                channel="pinn_density_output",
                category=AnomalyCategory.SEU_BIT_FLIP,
                severity=AnomalySeverity.CRITICAL,
                detected_value=round(pinn_out, 6),
                expected_range=(0.0, float("inf")),
                deviation_sigma=round(seu_sigma, 3),
                description=f"Possible SEU bit-flip in PINN output: modified Z-score {seu_sigma:.1f}σ. "
                            "TMR majority voting should be triggered immediately.",
            ))

        # ---- Overall health score ----
        n_crit    = len([a for a in anomalies if a.severity == AnomalySeverity.CRITICAL])
        n_warn    = len([a for a in anomalies if a.severity == AnomalySeverity.WARNING])
        health    = max(0.0, 100.0 - n_crit * 20.0 - n_warn * 7.0)

        return AnomalyReport(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            satellite_id=self.satellite_id,
            epoch_index=self._epoch,
            total_channels=len(self._CHANNELS),
            anomaly_count=len(anomalies),
            critical_count=n_crit,
            warning_count=n_warn,
            overall_health_score=round(health, 1),
            anomalies=anomalies,
            channel_z_scores=z_scores,
            cusum_flags=cusum_flags,
            physics_checks=phys_ok,
            metadata={
                "z_threshold": self.z_threshold,
                "mahal_threshold": self.mahal_threshold,
                "epoch": self._epoch,
            },
        )


# ---------------------------------------------------------------------------
# Convenience function (stateless single-call API)
# ---------------------------------------------------------------------------

def detect_telemetry_anomalies(
    satellite_id: str,
    frames: List[Dict[str, Any]],
    z_threshold: float = 3.5,
) -> List[AnomalyReport]:
    """
    Process a batch of sequential telemetry frames and return anomaly
    reports for each.  The detector accumulates statistics across frames
    to enable trend and change-point analysis.

    Parameters
    ----------
    satellite_id : str
    frames : list[dict]
        Ordered sequence of telemetry dicts (oldest first).
    z_threshold : float
        Z-score outlier threshold.
    """
    detector = TelemetryAnomalyDetector(
        satellite_id=satellite_id,
        z_threshold=z_threshold,
    )
    return [detector.ingest_frame(f) for f in frames]
