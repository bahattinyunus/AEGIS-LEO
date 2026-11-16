"""
AEGIS-LEO Mission Analytics Engine
====================================
Comprehensive mission health monitoring, subsystem trend analysis, and
key-performance-indicator (KPI) dashboard generation for multi-epoch
LEO satellite operations.

The engine ingests rolling telemetry snapshots and produces:
  - Per-subsystem health scores  (0 – 100)
  - Trend vectors (improving / stable / degrading)
  - Anomaly flags with severity (INFO / WARNING / CRITICAL)
  - Fuel budget burn-rate projection
  - Radiation accumulation and remaining shielding life estimation
  - Mission lifecycle stage classification
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
    COLLISION_PROBABILITY_THRESHOLD,
    UPDATED_FCC_LIFETIME_YEARS,
    MAX_LEO_LIFETIME_YEARS,
    QUIET_KP,
    QUIET_F10_7,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class HealthGrade(str, Enum):
    """Subsystem health classification."""
    EXCELLENT = "EXCELLENT"   # >= 90
    NOMINAL   = "NOMINAL"     # 70 – 89
    CAUTION   = "CAUTION"     # 50 – 69
    WARNING   = "WARNING"     # 30 – 49
    CRITICAL  = "CRITICAL"    # < 30


class TrendDirection(str, Enum):
    """Short-window trend over the last N snapshots."""
    IMPROVING  = "IMPROVING"
    STABLE     = "STABLE"
    DEGRADING  = "DEGRADING"
    VOLATILE   = "VOLATILE"   # High variance, no clear monotonic direction


class MissionStage(str, Enum):
    LAUNCH_AND_EARLY_ORBIT = "LAUNCH_AND_EARLY_ORBIT"
    COMMISSIONING          = "COMMISSIONING"
    NOMINAL_OPERATIONS     = "NOMINAL_OPERATIONS"
    ANOMALY_RECOVERY       = "ANOMALY_RECOVERY"
    END_OF_LIFE            = "END_OF_LIFE"
    DEORBIT_SEQUENCE       = "DEORBIT_SEQUENCE"


class AnomalySeverity(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Data-class definitions
# ---------------------------------------------------------------------------

@dataclass
class AnomalyFlag:
    subsystem: str
    severity: AnomalySeverity
    description: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SubsystemHealth:
    name: str
    health_score: float          # 0 – 100
    grade: HealthGrade
    trend: TrendDirection
    anomalies: List[AnomalyFlag] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FuelBudgetProjection:
    current_propellant_kg: float
    nominal_burn_rate_kg_per_day: float
    storm_burn_rate_kg_per_day: float
    days_remaining_nominal: float
    days_remaining_worst_case: float
    fuel_fraction_pct: float            # % of initial loaded propellant
    lifecycle_limited_by_fuel: bool


@dataclass
class RadiationLifeProjection:
    tid_accumulated_krad: float
    tid_budget_remaining_krad: float
    tid_annual_rate_krad_per_year: float
    years_to_tid_limit: float
    seu_rate_per_day: float
    scrub_interval_s: float
    lifecycle_limited_by_radiation: bool


@dataclass
class SubsystemTrendReport:
    subsystem: str
    history: List[float]        # historical health scores (oldest → newest)
    trend: TrendDirection
    slope_per_epoch: float      # +ve = improving, −ve = degrading


@dataclass
class MissionHealthSnapshot:
    """Complete instantaneous health picture of the mission."""
    timestamp_utc: str
    satellite_id: str
    satellite_name: str
    mission_stage: MissionStage
    overall_health_score: float
    overall_grade: HealthGrade

    subsystems: Dict[str, SubsystemHealth] = field(default_factory=dict)
    anomalies: List[AnomalyFlag] = field(default_factory=list)
    fuel_projection: Optional[FuelBudgetProjection] = None
    radiation_projection: Optional[RadiationLifeProjection] = None
    kpi_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _grade_from_score(score: float) -> HealthGrade:
    if score >= 90:
        return HealthGrade.EXCELLENT
    if score >= 70:
        return HealthGrade.NOMINAL
    if score >= 50:
        return HealthGrade.CAUTION
    if score >= 30:
        return HealthGrade.WARNING
    return HealthGrade.CRITICAL


def _trend_from_history(history: List[float]) -> Tuple[TrendDirection, float]:
    """Return (TrendDirection, slope_per_epoch) from a list of scalar values."""
    if len(history) < 3:
        return TrendDirection.STABLE, 0.0

    x = np.arange(len(history), dtype=float)
    y = np.array(history, dtype=float)
    # Detrended standard deviation check
    coeffs = np.polyfit(x, y, 1)
    slope = float(coeffs[0])
    fitted = np.polyval(coeffs, x)
    residual_std = float(np.std(y - fitted))

    # Classify
    if residual_std > 10.0:
        return TrendDirection.VOLATILE, slope
    if slope > 0.5:
        return TrendDirection.IMPROVING, slope
    if slope < -0.5:
        return TrendDirection.DEGRADING, slope
    return TrendDirection.STABLE, slope


# ---------------------------------------------------------------------------
# Scoring sub-functions (each returns a score in [0, 100] + anomalies)
# ---------------------------------------------------------------------------

def _score_propulsion(
    fuel_kg: float,
    initial_fuel_kg: float,
    isp_s: float,
    sat_mass_kg: float,
    kp: float,
) -> Tuple[float, List[AnomalyFlag]]:
    """Propulsion / fuel budget health."""
    anomalies: List[AnomalyFlag] = []
    fraction = max(0.0, fuel_kg / max(initial_fuel_kg, 1.0))
    score = fraction * 100.0

    # Storm burns increase consumption
    if kp >= 7.0 and fraction < 0.15:
        flag = AnomalyFlag(
            subsystem="PROPULSION",
            severity=AnomalySeverity.CRITICAL,
            description=f"Critical fuel reserve ({fuel_kg:.1f} kg) during G{int(max(1,kp-4))} storm. "
                        "Emergency evasion capability severely constrained.",
            metric_name="fuel_fraction",
            current_value=fraction,
            threshold_value=0.15,
        )
        anomalies.append(flag)
        score = max(score - 20.0, 0.0)
    elif fraction < 0.08:
        flag = AnomalyFlag(
            subsystem="PROPULSION",
            severity=AnomalySeverity.CRITICAL,
            description=f"Fuel critically depleted: {fuel_kg:.2f} kg remaining. EOL deorbit window narrowing.",
            metric_name="fuel_fraction",
            current_value=fraction,
            threshold_value=0.08,
        )
        anomalies.append(flag)
    elif fraction < 0.20:
        anomalies.append(AnomalyFlag(
            subsystem="PROPULSION",
            severity=AnomalySeverity.WARNING,
            description=f"Fuel reserves below 20% ({fuel_kg:.1f} kg). Proactive evasions should be minimised.",
            metric_name="fuel_fraction",
            current_value=fraction,
            threshold_value=0.20,
        ))

    return float(np.clip(score, 0.0, 100.0)), anomalies


def _score_risk(adri_score: float, pc: float) -> Tuple[float, List[AnomalyFlag]]:
    """Collision risk / A-DRI health (inverted: low risk → high score)."""
    anomalies: List[AnomalyFlag] = []
    risk_health = (1.0 - adri_score) * 100.0

    if adri_score >= 0.80 or pc >= 1e-3:
        anomalies.append(AnomalyFlag(
            subsystem="COLLISION_RISK",
            severity=AnomalySeverity.CRITICAL,
            description=f"A-DRI={adri_score:.3f} CRITICAL. Pc={pc:.2e}. EMERGENCY_AUTONOMOUS_EVASION commanded.",
            metric_name="a_dri",
            current_value=adri_score,
            threshold_value=0.80,
        ))
    elif adri_score >= 0.60 or pc >= COLLISION_PROBABILITY_THRESHOLD:
        anomalies.append(AnomalyFlag(
            subsystem="COLLISION_RISK",
            severity=AnomalySeverity.WARNING,
            description=f"A-DRI={adri_score:.3f} HIGH. Pc={pc:.2e} exceeds NASA 1e-4 threshold. Proactive burn scheduled.",
            metric_name="a_dri",
            current_value=adri_score,
            threshold_value=0.60,
        ))

    return float(np.clip(risk_health, 0.0, 100.0)), anomalies


def _score_power(power_margin_positive: bool, dod_pct: float, beta_deg: float) -> Tuple[float, List[AnomalyFlag]]:
    """Eclipse/power budget health."""
    anomalies: List[AnomalyFlag] = []
    score = 100.0

    if not power_margin_positive:
        score -= 40.0
        anomalies.append(AnomalyFlag(
            subsystem="POWER",
            severity=AnomalySeverity.CRITICAL,
            description=f"Negative power margin. DoD={dod_pct:.1f}% — battery may not survive full eclipse. "
                        f"Beta angle={beta_deg:.1f}°.",
            metric_name="power_margin_positive",
            current_value=0.0,
            threshold_value=1.0,
        ))
    elif dod_pct > 80.0:
        score -= 20.0
        anomalies.append(AnomalyFlag(
            subsystem="POWER",
            severity=AnomalySeverity.WARNING,
            description=f"Battery DoD={dod_pct:.1f}% exceeds 80% — accelerated degradation likely.",
            metric_name="dod_pct",
            current_value=dod_pct,
            threshold_value=80.0,
        ))
    elif dod_pct > 65.0:
        score -= 10.0
        anomalies.append(AnomalyFlag(
            subsystem="POWER",
            severity=AnomalySeverity.INFO,
            description=f"Battery DoD={dod_pct:.1f}% elevated. Monitor charge cycles.",
            metric_name="dod_pct",
            current_value=dod_pct,
            threshold_value=65.0,
        ))

    return float(np.clip(score, 0.0, 100.0)), anomalies


def _score_thermal(all_within_limits: bool, peak_aero_w: float, altitude_km: float, kp: float) -> Tuple[float, List[AnomalyFlag]]:
    """Thermal control health."""
    anomalies: List[AnomalyFlag] = []
    score = 100.0

    if not all_within_limits:
        score -= 35.0
        anomalies.append(AnomalyFlag(
            subsystem="THERMAL",
            severity=AnomalySeverity.CRITICAL,
            description="One or more thermal nodes exceeding design limits. Risk of component damage.",
            metric_name="all_nodes_within_limits",
            current_value=0.0,
            threshold_value=1.0,
        ))

    if altitude_km < 300.0 and kp >= 6.0 and peak_aero_w > 10.0:
        score -= 20.0
        anomalies.append(AnomalyFlag(
            subsystem="THERMAL",
            severity=AnomalySeverity.WARNING,
            description=f"VLEO aero-heating Q={peak_aero_w:.1f} W elevated during Kp={kp:.1f} storm. "
                        "Active radiator orientation recommended.",
            metric_name="peak_aero_heating_w",
            current_value=peak_aero_w,
            threshold_value=10.0,
        ))

    return float(np.clip(score, 0.0, 100.0)), anomalies


def _score_comms(total_passes: int, total_data_mb: float, link_margin_ok: bool) -> Tuple[float, List[AnomalyFlag]]:
    """Ground station communications health."""
    anomalies: List[AnomalyFlag] = []
    score = 100.0

    if total_passes == 0:
        score = 10.0
        anomalies.append(AnomalyFlag(
            subsystem="COMMS",
            severity=AnomalySeverity.CRITICAL,
            description="No ground station passes scheduled. Blackout risk — swarm autonomous mode required.",
            metric_name="total_passes",
            current_value=0.0,
            threshold_value=1.0,
        ))
    elif total_passes < 3:
        score -= 20.0
        anomalies.append(AnomalyFlag(
            subsystem="COMMS",
            severity=AnomalySeverity.WARNING,
            description=f"Only {total_passes} ground passes scheduled. Limited command uplink window.",
            metric_name="total_passes",
            current_value=float(total_passes),
            threshold_value=3.0,
        ))

    if total_data_mb < 50.0 and total_passes > 0:
        score -= 15.0
        anomalies.append(AnomalyFlag(
            subsystem="COMMS",
            severity=AnomalySeverity.WARNING,
            description=f"Downlink volume low ({total_data_mb:.1f} MB). Consider higher link band or longer passes.",
            metric_name="total_data_mb",
            current_value=total_data_mb,
            threshold_value=50.0,
        ))

    return float(np.clip(score, 0.0, 100.0)), anomalies


def _score_radiation(tid_krad_yr: float, seu_per_day: float, scrub_s: float) -> Tuple[float, List[AnomalyFlag]]:
    """Radiation environment health."""
    anomalies: List[AnomalyFlag] = []
    score = 100.0

    if tid_krad_yr > 10.0:
        score -= 30.0
        anomalies.append(AnomalyFlag(
            subsystem="RADIATION",
            severity=AnomalySeverity.CRITICAL,
            description=f"TID rate {tid_krad_yr:.1f} krad/yr exceeds 10 krad/yr limit. "
                        "Component lifetime severely impacted.",
            metric_name="tid_krad_per_year",
            current_value=tid_krad_yr,
            threshold_value=10.0,
        ))
    elif tid_krad_yr > 5.0:
        score -= 15.0
        anomalies.append(AnomalyFlag(
            subsystem="RADIATION",
            severity=AnomalySeverity.WARNING,
            description=f"Elevated TID rate {tid_krad_yr:.1f} krad/yr. Increase scrub interval vigilance.",
            metric_name="tid_krad_per_year",
            current_value=tid_krad_yr,
            threshold_value=5.0,
        ))

    if seu_per_day > 5.0:
        score -= 20.0
        anomalies.append(AnomalyFlag(
            subsystem="RADIATION",
            severity=AnomalySeverity.WARNING,
            description=f"SEU rate {seu_per_day:.1f}/day — TMR scrub interval should be ≤{scrub_s:.0f} s.",
            metric_name="seu_per_day",
            current_value=seu_per_day,
            threshold_value=5.0,
        ))

    return float(np.clip(score, 0.0, 100.0)), anomalies


def _score_security(authentic: bool, replay_blocked: bool, tamper_blocked: bool) -> Tuple[float, List[AnomalyFlag]]:
    """PQC telecommand security health."""
    anomalies: List[AnomalyFlag] = []
    score = 100.0

    if not authentic:
        score -= 50.0
        anomalies.append(AnomalyFlag(
            subsystem="SECURITY",
            severity=AnomalySeverity.CRITICAL,
            description="PQC telecommand authentication FAILED. Command rejected — potential adversarial intrusion.",
            metric_name="authentic_command_verified",
            current_value=0.0,
            threshold_value=1.0,
        ))
    if not replay_blocked:
        score -= 25.0
        anomalies.append(AnomalyFlag(
            subsystem="SECURITY",
            severity=AnomalySeverity.WARNING,
            description="Replay attack not blocked. Sliding window counter may be misconfigured.",
            metric_name="replay_attack_blocked",
            current_value=0.0,
            threshold_value=1.0,
        ))
    if not tamper_blocked:
        score -= 25.0
        anomalies.append(AnomalyFlag(
            subsystem="SECURITY",
            severity=AnomalySeverity.WARNING,
            description="Tamper attack not blocked. Merkle chain integrity check failed.",
            metric_name="tamper_attack_blocked",
            current_value=0.0,
            threshold_value=1.0,
        ))

    return float(np.clip(score, 0.0, 100.0)), anomalies


# ---------------------------------------------------------------------------
# Main Analytics Engine
# ---------------------------------------------------------------------------

class MissionAnalyticsEngine:
    """
    Stateful engine that accepts rolling telemetry snapshots and maintains
    per-subsystem health histories for trend analysis.

    Usage::

        engine = MissionAnalyticsEngine(satellite_id="SAT-01", initial_fuel_kg=20.0)
        snapshot = engine.ingest_snapshot(telemetry_dict)
        print(snapshot.overall_grade)
    """

    _HISTORY_DEPTH = 10  # Keep last N snapshots for trend computation

    def __init__(self, satellite_id: str = "SAT-01", initial_fuel_kg: float = 20.0):
        self.satellite_id = satellite_id
        self.initial_fuel_kg = max(initial_fuel_kg, 0.1)
        self._health_history: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------
    def ingest_snapshot(self, telemetry: Dict[str, Any]) -> MissionHealthSnapshot:
        """
        Primary entry point. Accepts a flat telemetry dict and returns a
        fully-scored MissionHealthSnapshot.

        Required telemetry keys (use defaults for missing):
          satellite_name, altitude_km, mass_kg, fuel_remaining_kg, isp_s,
          kp, f10_7, adri_score, collision_pc, power_margin_positive,
          dod_pct, beta_deg, all_thermal_ok, peak_aero_w, total_passes,
          total_data_mb, tid_krad_yr, seu_per_day, scrub_s,
          pqc_authentic, pqc_replay_ok, pqc_tamper_ok.
        """
        t = telemetry  # shorthand

        sat_name   = str(t.get("satellite_name", self.satellite_id))
        alt_km     = float(t.get("altitude_km", 550.0))
        mass_kg    = float(t.get("mass_kg", 260.0))
        fuel_kg    = float(t.get("fuel_remaining_kg", self.initial_fuel_kg * 0.5))
        isp_s      = float(t.get("isp_s", 250.0))
        kp         = float(t.get("kp", QUIET_KP))
        f107       = float(t.get("f10_7", QUIET_F10_7))

        adri       = float(t.get("adri_score", 0.0))
        pc         = float(t.get("collision_pc", 1e-7))

        pwr_ok     = bool(t.get("power_margin_positive", True))
        dod        = float(t.get("dod_pct", 40.0))
        beta       = float(t.get("beta_deg", 20.0))

        therm_ok   = bool(t.get("all_thermal_ok", True))
        aero_w     = float(t.get("peak_aero_w", 1.0))

        passes     = int(t.get("total_passes", 5))
        data_mb    = float(t.get("total_data_mb", 200.0))
        link_ok    = bool(t.get("link_margin_ok", True))

        tid_yr     = float(t.get("tid_krad_yr", 1.0))
        seu_day    = float(t.get("seu_per_day", 0.1))
        scrub_s    = float(t.get("scrub_s", 300.0))

        pqc_auth   = bool(t.get("pqc_authentic", True))
        pqc_replay = bool(t.get("pqc_replay_ok", True))
        pqc_tamper = bool(t.get("pqc_tamper_ok", True))

        # --- Score each subsystem ---
        s_prop,  a_prop  = _score_propulsion(fuel_kg, self.initial_fuel_kg, isp_s, mass_kg, kp)
        s_risk,  a_risk  = _score_risk(adri, pc)
        s_pwr,   a_pwr   = _score_power(pwr_ok, dod, beta)
        s_therm, a_therm = _score_thermal(therm_ok, aero_w, alt_km, kp)
        s_comms, a_comms = _score_comms(passes, data_mb, link_ok)
        s_rad,   a_rad   = _score_radiation(tid_yr, seu_day, scrub_s)
        s_sec,   a_sec   = _score_security(pqc_auth, pqc_replay, pqc_tamper)

        sub_scores = {
            "PROPULSION":     s_prop,
            "COLLISION_RISK": s_risk,
            "POWER":          s_pwr,
            "THERMAL":        s_therm,
            "COMMS":          s_comms,
            "RADIATION":      s_rad,
            "SECURITY":       s_sec,
        }

        all_anomalies = a_prop + a_risk + a_pwr + a_therm + a_comms + a_rad + a_sec

        # --- Update trend histories ---
        subsystems: Dict[str, SubsystemHealth] = {}
        for name, score in sub_scores.items():
            hist = self._health_history.setdefault(name, [])
            hist.append(score)
            if len(hist) > self._HISTORY_DEPTH:
                hist.pop(0)
            trend, _ = _trend_from_history(hist)
            ss_anomalies = [a for a in all_anomalies if a.subsystem == name]
            grade = _grade_from_score(score)
            subsystems[name] = SubsystemHealth(
                name=name,
                health_score=round(score, 1),
                grade=grade,
                trend=trend,
                anomalies=ss_anomalies,
                metrics={"raw_score": score},
            )

        # --- Overall health (weighted average) ---
        weights = {
            "PROPULSION": 1.5,
            "COLLISION_RISK": 2.0,
            "POWER": 1.0,
            "THERMAL": 1.0,
            "COMMS": 0.8,
            "RADIATION": 1.0,
            "SECURITY": 1.2,
        }
        total_w = sum(weights.values())
        overall = sum(sub_scores[k] * weights[k] for k in weights) / total_w
        overall = float(np.clip(overall, 0.0, 100.0))

        # --- Mission stage ---
        fuel_frac = fuel_kg / self.initial_fuel_kg
        if fuel_frac > 0.90:
            stage = MissionStage.LAUNCH_AND_EARLY_ORBIT
        elif fuel_frac > 0.70:
            stage = MissionStage.COMMISSIONING
        elif fuel_frac > 0.15:
            stage = MissionStage.NOMINAL_OPERATIONS
        elif len([a for a in all_anomalies if a.severity == AnomalySeverity.CRITICAL]) >= 2:
            stage = MissionStage.ANOMALY_RECOVERY
        elif fuel_frac <= 0.05:
            stage = MissionStage.DEORBIT_SEQUENCE
        else:
            stage = MissionStage.END_OF_LIFE

        # --- Fuel projection ---
        v_circ_km_s = math.sqrt(EARTH_MU / (EARTH_RADIUS_KM + alt_km))
        dv_per_storm_m_s = 0.4 if kp >= 5.0 else 0.0
        dv_nominal_per_day = 0.012     # typical station-keeping
        dv_storm_per_day   = 0.012 + (dv_per_storm_m_s / 86400.0) * 3600.0  # rough

        def _burn_rate(dv_m_s_per_day: float) -> float:
            """Tsiolkovsky mass flow for daily delta-v budget."""
            dm = mass_kg * (1.0 - math.exp(-dv_m_s_per_day / (isp_s * 9.80665)))
            return max(dm, 1e-6)

        nom_rate  = _burn_rate(dv_nominal_per_day)
        strm_rate = _burn_rate(dv_storm_per_day)
        fuel_proj = FuelBudgetProjection(
            current_propellant_kg=round(fuel_kg, 3),
            nominal_burn_rate_kg_per_day=round(nom_rate, 5),
            storm_burn_rate_kg_per_day=round(strm_rate, 5),
            days_remaining_nominal=round(fuel_kg / max(nom_rate, 1e-9), 1),
            days_remaining_worst_case=round(fuel_kg / max(strm_rate, 1e-9), 1),
            fuel_fraction_pct=round(fuel_frac * 100.0, 2),
            lifecycle_limited_by_fuel=(fuel_frac < 0.25),
        )

        # --- Radiation life projection ---
        tid_budget = float(t.get("component_tid_tolerance_krad", 30.0))
        years_used = float(t.get("mission_elapsed_years", 1.0))
        tid_accumulated = tid_yr * years_used
        rad_proj = RadiationLifeProjection(
            tid_accumulated_krad=round(tid_accumulated, 3),
            tid_budget_remaining_krad=round(max(0.0, tid_budget - tid_accumulated), 3),
            tid_annual_rate_krad_per_year=round(tid_yr, 4),
            years_to_tid_limit=round(max(0.0, (tid_budget - tid_accumulated) / max(tid_yr, 1e-6)), 1),
            seu_rate_per_day=round(seu_day, 4),
            scrub_interval_s=round(scrub_s, 1),
            lifecycle_limited_by_radiation=(tid_accumulated > tid_budget * 0.80),
        )

        # --- KPI summary ---
        kpi = {
            "overall_health_pct": round(overall, 1),
            "critical_anomaly_count": len([a for a in all_anomalies if a.severity == AnomalySeverity.CRITICAL]),
            "warning_anomaly_count": len([a for a in all_anomalies if a.severity == AnomalySeverity.WARNING]),
            "fuel_days_remaining": fuel_proj.days_remaining_nominal,
            "tid_years_remaining": rad_proj.years_to_tid_limit,
            "risk_level": "CRITICAL" if adri >= 0.80 else ("HIGH" if adri >= 0.60 else ("ELEVATED" if adri >= 0.30 else "NOMINAL")),
            "mission_stage": stage.value,
            "kp_index": kp,
        }

        return MissionHealthSnapshot(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            satellite_id=self.satellite_id,
            satellite_name=sat_name,
            mission_stage=stage,
            overall_health_score=round(overall, 1),
            overall_grade=_grade_from_score(overall),
            subsystems=subsystems,
            anomalies=all_anomalies,
            fuel_projection=fuel_proj,
            radiation_projection=rad_proj,
            kpi_summary=kpi,
        )

    def get_subsystem_trend_reports(self) -> List[SubsystemTrendReport]:
        """Return trend reports for all tracked subsystems."""
        reports = []
        for name, hist in self._health_history.items():
            trend, slope = _trend_from_history(hist)
            reports.append(SubsystemTrendReport(
                subsystem=name,
                history=list(hist),
                trend=trend,
                slope_per_epoch=round(slope, 4),
            ))
        return reports


# ---------------------------------------------------------------------------
# Convenience function (stateless single-call API)
# ---------------------------------------------------------------------------

def analyze_mission_health(
    satellite_id: str,
    telemetry: Dict[str, Any],
    initial_fuel_kg: float = 20.0,
) -> MissionHealthSnapshot:
    """
    Stateless single-call convenience wrapper for the analytics engine.

    This function creates a temporary engine instance and processes one
    telemetry snapshot. Use ``MissionAnalyticsEngine`` directly when you
    need rolling trend analysis across multiple epochs.
    """
    engine = MissionAnalyticsEngine(satellite_id=satellite_id, initial_fuel_kg=initial_fuel_kg)
    return engine.ingest_snapshot(telemetry)
