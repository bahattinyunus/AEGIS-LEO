"""
AEGIS-LEO Space Weather Forecast Engine
=========================================
24–72 hour probabilistic Kp and F10.7 forecasting using a multi-step
autoregressive physics-informed approach.

The engine uses:
  1. Historical Kp/F10.7 baseline patterns from NOAA SWPC statistics.
  2. Active Region proxies (sunspot number, CME transit model) to modulate
     geomagnetic activity forecasts.
  3. A recurrent linear prediction filter (AR(4)) with a storm onset
     injection model for CME-driven sudden commencement events.
  4. Probabilistic ensemble (N = 50 realisations) to produce confidence
     intervals around Kp and density forecasts.

Output:
  - Hourly Kp and F10.7 traces for the forecast horizon.
  - Thermospheric density factor μ(t) relative to quiet baseline.
  - Along-track drift velocity accumulation for the primary orbit.
  - Storm onset probability estimate P_storm(t) over the horizon.
  - Early-warning flag when P_storm > 70%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from aegis_leo.constants import (
    QUIET_KP,
    QUIET_F10_7,
    EARTH_RADIUS_KM,
    EARTH_MU,
)
from aegis_leo.physics.atmosphere import compute_atmospheric_density


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# NOAA SWPC 1-hour Kp baseline autocorrelation (AR(4)) coefficients
# Derived from 1998–2024 OMNI2 hourly data (empirical approximation).
_KP_AR4_COEFFS: np.ndarray = np.array([0.68, 0.14, 0.09, 0.05])

# Quiet Kp mean & std for Gaussian baseline noise
_KP_QUIET_MEAN: float = 1.4
_KP_QUIET_STD: float  = 0.9

# CME transit time distribution: mean 36 h, std 10 h (STEREO statistics)
_CME_TRANSIT_MEAN_H: float = 36.0
_CME_TRANSIT_STD_H: float  = 10.0

# F10.7 autocorrelation (27-day rotation, simplified AR(2))
_F107_AR2_COEFFS: np.ndarray = np.array([0.92, 0.06])
_F107_NOISE_STD: float = 8.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ForecastHour:
    """Single-hour forecast state."""
    hour_offset: int            # hours from forecast epoch (0 = now)
    kp_p50: float               # median Kp
    kp_p10: float               # 10th-percentile (optimistic)
    kp_p90: float               # 90th-percentile (pessimistic)
    f10_7_p50: float
    storm_onset_probability: float          # 0–1
    density_factor_p50: float              # ρ / ρ_quiet at given orbit
    along_track_drift_km_hr: float         # expected cumulative drift per hour


@dataclass
class StormOnsetAlert:
    """Active early-warning alert for a predicted storm onset."""
    alert_id: str
    cme_speed_km_s: float
    predicted_onset_hour: float
    onset_confidence_pct: float
    expected_peak_kp: float
    impact_description: str


@dataclass
class SpaceWeatherForecast:
    """Full 24–72 hour forecast product."""
    timestamp_utc: str
    horizon_hours: int
    ensemble_size: int

    current_kp: float
    current_f10_7: float
    forecast_hours: List[ForecastHour]

    max_kp_forecast: float
    max_kp_hour: int
    storm_probability_24h: float
    storm_probability_72h: float

    total_drift_km_24h: float
    total_drift_km_72h: float

    alerts: List[StormOnsetAlert] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# CME transit model
# ---------------------------------------------------------------------------

class CMETransitModel:
    """
    Simple CME arrival-time and peak-Kp model.

    Parameters
    ----------
    cme_speed_km_s : float
        Plane-of-sky CME speed from coronagraph (LASCO/STEREO).
    base_kp : float
        Background Kp at time of CME launch.
    """

    def __init__(self, cme_speed_km_s: float = 0.0, base_kp: float = QUIET_KP):
        self.cme_speed = max(0.0, cme_speed_km_s)
        self.base_kp   = base_kp

    def is_active(self) -> bool:
        return self.cme_speed > 300.0

    def predicted_transit_hours(self) -> float:
        """Average transit time model from Wang et al. (2004)."""
        if self.cme_speed < 300.0:
            return float("inf")
        # Empirical: t_transit ≈ 1.5e8 / v_cme  (seconds) → hours
        t_s = 1.5e8 / max(self.cme_speed, 1.0)
        return t_s / 3600.0

    def predicted_peak_kp(self) -> float:
        """Scale peak Kp with CME speed (Newell et al. 2007 proxy)."""
        if self.cme_speed < 300.0:
            return self.base_kp
        kp_add = min(7.0, 0.009 * (self.cme_speed - 300.0) ** 0.6)
        return float(np.clip(self.base_kp + kp_add, 0.0, 9.0))

    def onset_probability(self, hours_ahead: float) -> float:
        """P(storm onset at time T | CME detected now) using lognormal arrival."""
        if not self.is_active():
            return 0.0
        mu_h = self.predicted_transit_hours()
        sig  = _CME_TRANSIT_STD_H
        # Gaussian CDF approximation
        z = (hours_ahead - mu_h) / sig
        return float(np.clip(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))), 0.0, 1.0))


# ---------------------------------------------------------------------------
# AR(4) Kp autoregressive predictor
# ---------------------------------------------------------------------------

def _ar_step(
    history: np.ndarray,
    coeffs: np.ndarray,
    mu: float,
    noise_std: float,
    rng: np.random.Generator,
) -> float:
    """One-step AR prediction with additive Gaussian noise."""
    h = history[-len(coeffs):][::-1]      # most-recent first
    pred = mu * (1.0 - coeffs.sum()) + float(np.dot(coeffs, h - mu))
    return float(np.clip(pred + rng.normal(0.0, noise_std), 0.0, 9.0))


def _f107_ar_step(
    history: np.ndarray,
    coeffs: np.ndarray,
    rng: np.random.Generator,
) -> float:
    h = history[-len(coeffs):][::-1]
    mu = float(np.mean(h))
    pred = mu * (1.0 - coeffs.sum()) + float(np.dot(coeffs, h - mu))
    return float(np.clip(pred + rng.normal(0.0, _F107_NOISE_STD), 65.0, 450.0))


# ---------------------------------------------------------------------------
# Density and drift helpers
# ---------------------------------------------------------------------------

def _density_factor(kp: float, f10_7: float, altitude_km: float) -> float:
    """Return ρ(kp,f10.7) / ρ(quiet) at altitude."""
    quiet_rho = compute_atmospheric_density(altitude_km, QUIET_KP, QUIET_F10_7)
    storm_rho = compute_atmospheric_density(altitude_km, kp, f10_7)
    return float(storm_rho / max(quiet_rho, 1e-25))


def _along_track_drift_km_per_hour(
    kp: float,
    f10_7: float,
    altitude_km: float,
    area_m2: float = 2.2,
    mass_kg: float = 260.0,
    cd: float = 2.2,
) -> float:
    """
    Approximate along-track drift rate (km / hour) relative to a quiet-day
    prediction, induced by density enhancement.

    Δa_drag ∝ (ρ_storm − ρ_quiet) × (Cd × A / m) × v²
    Δv_drift ≈ Δa × T_orbit  (first-order secular accumulation)
    """
    r_km = EARTH_RADIUS_KM + altitude_km
    v_km_s = math.sqrt(EARTH_MU / r_km)
    quiet_rho = compute_atmospheric_density(altitude_km, QUIET_KP, QUIET_F10_7)
    storm_rho = compute_atmospheric_density(altitude_km, kp, f10_7)
    delta_rho  = max(0.0, storm_rho - quiet_rho)        # kg / m³
    # Drag acceleration delta (m/s²) → convert units
    bstar = (cd * area_m2) / max(mass_kg, 1.0)          # m²/kg
    delta_a = 0.5 * delta_rho * bstar * (v_km_s * 1000.0) ** 2    # m/s²
    t_orbit_s = 2.0 * math.pi * math.sqrt((r_km * 1000.0)**3 / (EARTH_MU * 1e9))
    drift_m_per_orbit = delta_a * t_orbit_s**2            # s × m/s² = m/s → m
    orbits_per_hour = 3600.0 / max(t_orbit_s, 1.0)
    drift_km_per_hour = (drift_m_per_orbit / 1000.0) * orbits_per_hour
    return max(0.0, drift_km_per_hour)


# ---------------------------------------------------------------------------
# Main forecaster
# ---------------------------------------------------------------------------

class SpaceWeatherForecaster:
    """
    Generates probabilistic multi-step Kp and F10.7 forecasts.

    Parameters
    ----------
    current_kp : float
        Latest measured Kp (0–9).
    current_f10_7 : float
        Latest measured F10.7 (SFU).
    kp_history : list[float], optional
        Last 4+ hours of Kp observations (oldest first). Used to seed the
        AR(4) predictor.  Defaults to constant quiet baseline.
    cme_speed_km_s : float, optional
        Detected CME plane-of-sky speed.  Set to 0 if no CME is active.
    orbit_altitude_km : float, optional
        Primary satellite altitude for density / drift projections.
    ensemble_size : int
        Number of Monte Carlo ensemble members.
    rng_seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        current_kp: float = QUIET_KP,
        current_f10_7: float = QUIET_F10_7,
        kp_history: Optional[List[float]] = None,
        cme_speed_km_s: float = 0.0,
        orbit_altitude_km: float = 550.0,
        ensemble_size: int = 50,
        rng_seed: int = 42,
    ):
        self.current_kp     = float(np.clip(current_kp, 0.0, 9.0))
        self.current_f10_7  = float(np.clip(current_f10_7, 65.0, 450.0))
        self.kp_history     = list(kp_history or [QUIET_KP] * 4)
        self.cme_model      = CMETransitModel(cme_speed_km_s, current_kp)
        self.altitude_km    = orbit_altitude_km
        self.ensemble_size  = max(10, ensemble_size)
        self._rng_seed      = rng_seed

    # ------------------------------------------------------------------
    def forecast(self, horizon_hours: int = 24) -> SpaceWeatherForecast:
        """Generate a full forecast product."""
        horizon_hours = int(np.clip(horizon_hours, 1, 72))
        rng_master = np.random.default_rng(self._rng_seed)

        # Seed each ensemble member differently
        member_seeds = rng_master.integers(0, 1_000_000, size=self.ensemble_size)

        # Ensemble container: [member, hour]
        kp_ens   = np.zeros((self.ensemble_size, horizon_hours))
        f107_ens = np.zeros((self.ensemble_size, horizon_hours))

        for m, seed in enumerate(member_seeds):
            rng_m = np.random.default_rng(int(seed))
            kp_buf   = list(self.kp_history[-4:]) + [self.current_kp]
            f107_buf = [self.current_f10_7, self.current_f10_7]

            for h in range(horizon_hours):
                kp_next   = _ar_step(np.array(kp_buf), _KP_AR4_COEFFS, _KP_QUIET_MEAN, _KP_QUIET_STD, rng_m)
                f107_next = _f107_ar_step(np.array(f107_buf), _F107_AR2_COEFFS, rng_m)

                # Inject CME storm signal
                if self.cme_model.is_active():
                    p_onset = self.cme_model.onset_probability(h + 1)
                    # Probabilistic Kp injection
                    if rng_m.uniform() < (p_onset - self.cme_model.onset_probability(h)):
                        kp_next = max(kp_next, self.cme_model.predicted_peak_kp() * rng_m.uniform(0.7, 1.0))
                        kp_next = float(np.clip(kp_next, 0.0, 9.0))

                kp_ens[m, h]   = kp_next
                f107_ens[m, h] = f107_next
                kp_buf.append(kp_next)
                f107_buf.append(f107_next)

        # Aggregate percentiles
        kp_p10  = np.percentile(kp_ens,  10, axis=0)
        kp_p50  = np.percentile(kp_ens,  50, axis=0)
        kp_p90  = np.percentile(kp_ens,  90, axis=0)
        f107_p50 = np.percentile(f107_ens, 50, axis=0)

        # Build hourly forecast objects
        hours: List[ForecastHour] = []
        cumulative_drift_km = 0.0
        for h in range(horizon_hours):
            kp_h   = float(kp_p50[h])
            f107_h = float(f107_p50[h])
            df     = _density_factor(kp_h, f107_h, self.altitude_km)
            drift  = _along_track_drift_km_per_hour(kp_h, f107_h, self.altitude_km)
            cumulative_drift_km += drift
            p_storm = max(
                float(np.mean(kp_ens[:, h] >= 5.0)),
                self.cme_model.onset_probability(h + 1) if self.cme_model.is_active() else 0.0,
            )
            hours.append(ForecastHour(
                hour_offset=h + 1,
                kp_p50=round(kp_h, 2),
                kp_p10=round(float(kp_p10[h]), 2),
                kp_p90=round(float(kp_p90[h]), 2),
                f10_7_p50=round(f107_h, 2),
                storm_onset_probability=round(p_storm, 3),
                density_factor_p50=round(df, 3),
                along_track_drift_km_hr=round(drift, 4),
            ))

        # Storm probability summaries
        kp_24  = kp_ens[:, :min(24, horizon_hours)]
        kp_72  = kp_ens[:, :min(72, horizon_hours)]
        p_storm_24 = float(np.mean(np.any(kp_24 >= 5.0, axis=1)))
        p_storm_72 = float(np.mean(np.any(kp_72 >= 5.0, axis=1)))

        # Drift totals
        drift_24h = sum(h.along_track_drift_km_hr for h in hours[:24])
        drift_72h = sum(h.along_track_drift_km_hr for h in hours[:72])

        # Max Kp
        kp_flat = kp_p50
        max_kp_hour = int(np.argmax(kp_flat)) + 1
        max_kp      = float(np.max(kp_flat))

        # Alerts
        alerts: List[StormOnsetAlert] = []
        if self.cme_model.is_active():
            transit_h = self.cme_model.predicted_transit_hours()
            if transit_h <= horizon_hours:
                p_onset_max = max(
                    self.cme_model.onset_probability(h + 1) for h in range(min(int(transit_h) + 10, horizon_hours))
                )
                if p_onset_max >= 0.40:
                    impact = (
                        f"CME arrival expected at T+{transit_h:.1f}h. "
                        f"Peak Kp≈{self.cme_model.predicted_peak_kp():.1f}. "
                        f"Expected along-track drift ≈{drift_72h:.2f} km. "
                        "Recommend proactive micro-burn scheduling."
                    )
                    alerts.append(StormOnsetAlert(
                        alert_id=f"CME-ALERT-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
                        cme_speed_km_s=self.cme_model.cme_speed,
                        predicted_onset_hour=round(transit_h, 1),
                        onset_confidence_pct=round(p_onset_max * 100.0, 1),
                        expected_peak_kp=round(self.cme_model.predicted_peak_kp(), 1),
                        impact_description=impact,
                    ))

        # Early-warning: p_storm_24h > 0.70 and no CME explicitly tracked
        if p_storm_24 >= 0.70 and not self.cme_model.is_active():
            alerts.append(StormOnsetAlert(
                alert_id=f"STAT-WARN-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
                cme_speed_km_s=0.0,
                predicted_onset_hour=float(max_kp_hour),
                onset_confidence_pct=round(p_storm_24 * 100.0, 1),
                expected_peak_kp=round(max_kp, 1),
                impact_description=(
                    f"Statistical storm onset probability {p_storm_24*100:.0f}% in next 24 h "
                    "based on AR(4) background activity model."
                ),
            ))

        return SpaceWeatherForecast(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            horizon_hours=horizon_hours,
            ensemble_size=self.ensemble_size,
            current_kp=round(self.current_kp, 2),
            current_f10_7=round(self.current_f10_7, 2),
            forecast_hours=hours,
            max_kp_forecast=round(max_kp, 2),
            max_kp_hour=max_kp_hour,
            storm_probability_24h=round(p_storm_24, 3),
            storm_probability_72h=round(p_storm_72, 3),
            total_drift_km_24h=round(drift_24h, 3),
            total_drift_km_72h=round(drift_72h, 3),
            alerts=alerts,
            metadata={
                "cme_active": self.cme_model.is_active(),
                "cme_speed_km_s": self.cme_model.cme_speed,
                "orbit_altitude_km": self.altitude_km,
                "ar4_coeffs": _KP_AR4_COEFFS.tolist(),
            },
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_space_weather_forecast(
    current_kp: float = QUIET_KP,
    current_f10_7: float = QUIET_F10_7,
    horizon_hours: int = 48,
    cme_speed_km_s: float = 0.0,
    orbit_altitude_km: float = 550.0,
    kp_history: Optional[List[float]] = None,
) -> SpaceWeatherForecast:
    """
    Single-call convenience wrapper for the Space Weather Forecaster.

    Parameters
    ----------
    current_kp : float
        Current Kp index (0–9).
    current_f10_7 : float
        Current F10.7 solar flux (SFU).
    horizon_hours : int
        Forecast horizon in hours (1–72).
    cme_speed_km_s : float
        Detected CME speed in km/s (0 if none active).
    orbit_altitude_km : float
        Primary satellite altitude for density / drift forecasts.
    kp_history : list[float], optional
        Last N hours of observed Kp values (oldest→newest). Improves
        AR predictor accuracy when real data is available.
    """
    forecaster = SpaceWeatherForecaster(
        current_kp=current_kp,
        current_f10_7=current_f10_7,
        kp_history=kp_history,
        cme_speed_km_s=cme_speed_km_s,
        orbit_altitude_km=orbit_altitude_km,
    )
    return forecaster.forecast(horizon_hours=horizon_hours)
