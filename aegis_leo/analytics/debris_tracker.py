"""
AEGIS-LEO Advanced Debris Tracker
====================================
Multi-object LEO debris tracking with prioritised risk ranking,
close-approach chain detection, and fleet-wide conjunction risk matrix.

Capabilities:
  - Maintains a live catalogue of tracked debris objects with TLE-like
    Keplerian elements (parametric, not real SGP4).
  - Computes per-object A-DRI inspired risk score using orbital geometry.
  - Detects first-order close-approach chains (secondary conjunctions that
    could be triggered by a primary evasion manoeuvre).
  - Produces a DebrisFleetRiskMatrix summarising all pairings, sortable by
    risk score, TCA, or miss distance.
  - Supports synthetic catalogue generation for testing.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    EARTH_MU,
    EARTH_J2,
    COLLISION_PROBABILITY_THRESHOLD,
    DEBRIS_CRITICAL_DISTANCE_KM,
    DEBRIS_WARNING_DISTANCE_KM,
)


# ---------------------------------------------------------------------------
# Enumerations & base dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    """
    Minimal Keplerian state for a tracked LEO object (satellite or debris).

    All angles in degrees; altitude_km is mean semi-major-axis altitude above
    the WGS84 equatorial radius.
    """
    object_id: str
    name: str
    altitude_km: float
    inclination_deg: float
    raan_deg: float = 0.0
    eccentricity: float = 0.0
    arg_perigee_deg: float = 0.0
    mean_anomaly_deg: float = 0.0
    mass_kg: float = 100.0
    area_m2: float = 1.0
    is_active_satellite: bool = False
    # Derived / cached
    semi_major_axis_km: float = field(init=False)
    mean_motion_deg_s: float = field(init=False)
    orbital_period_s: float = field(init=False)

    def __post_init__(self):
        self.semi_major_axis_km = EARTH_RADIUS_KM + self.altitude_km
        mu = EARTH_MU
        n_rad_s = math.sqrt(mu / self.semi_major_axis_km ** 3)
        self.mean_motion_deg_s = math.degrees(n_rad_s)
        self.orbital_period_s = 2.0 * math.pi / n_rad_s

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConjunctionEvent:
    """A single conjunction event between two tracked objects."""
    object_a_id: str
    object_b_id: str
    object_a_name: str
    object_b_name: str
    tca_seconds_from_now: float
    miss_distance_km: float
    relative_speed_km_s: float
    risk_score: float          # 0–1, A-DRI-inspired
    is_critical: bool
    is_secondary_chain: bool   # triggered by primary evasion?
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DebrisFleetRiskMatrix:
    """Aggregated risk matrix across all tracked objects."""
    timestamp_utc: str
    primary_satellite_id: str
    total_tracked_objects: int
    total_conjunctions_screened: int
    critical_conjunctions: int
    warning_conjunctions: int
    highest_risk_score: float
    highest_risk_pairing: str
    min_miss_distance_km: float
    conjunction_events: List[ConjunctionEvent] = field(default_factory=list)
    secondary_chain_risk: bool = False
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["conjunction_events"] = [e.to_dict() for e in self.conjunction_events]
        return d

    def top_threats(self, n: int = 5) -> List[ConjunctionEvent]:
        """Return top-N threats by risk_score descending."""
        return sorted(self.conjunction_events, key=lambda e: e.risk_score, reverse=True)[:n]


# ---------------------------------------------------------------------------
# Geometric helpers (analytic, no numerical integration)
# ---------------------------------------------------------------------------

def _relative_velocity_km_s(obj_a: TrackedObject, obj_b: TrackedObject) -> float:
    """
    Approximate relative speed using vis-viva circular orbits and
    inclination difference.  Conservative upper-bound estimate.
    """
    mu = EARTH_MU
    va = math.sqrt(mu / obj_a.semi_major_axis_km)
    vb = math.sqrt(mu / obj_b.semi_major_axis_km)
    dinc = abs(obj_a.inclination_deg - obj_b.inclination_deg)
    # Dot-product approximation for coplanar intersection speed
    v_rel = math.sqrt(va**2 + vb**2 - 2.0 * va * vb * math.cos(math.radians(dinc)))
    return max(v_rel, 0.01)


def _secular_j2_raan_rate(obj: TrackedObject) -> float:
    """J2 nodal precession rate in deg/s (Brouwer first-order)."""
    a   = obj.semi_major_axis_km
    inc = math.radians(obj.inclination_deg)
    e   = obj.eccentricity
    n   = math.sqrt(EARTH_MU / a**3)   # rad/s
    raan_dot = -1.5 * n * EARTH_J2 * (EARTH_RADIUS_KM / a)**2 * math.cos(inc) / (1.0 - e**2)**2
    return math.degrees(raan_dot)       # deg/s


def _propagate_raan(obj: TrackedObject, t_seconds: float) -> float:
    """RAAN at epoch + t_seconds accounting for J2 drift."""
    return (obj.raan_deg + _secular_j2_raan_rate(obj) * t_seconds) % 360.0


def _raan_crossing_time(obj_a: TrackedObject, obj_b: TrackedObject,
                        horizon_s: float = 21600.0) -> Optional[float]:
    """
    Returns approximate time (seconds from now) when RAAN planes align
    within 5°, or None if no crossing in the horizon.
    """
    dt = 60.0
    t = 0.0
    while t <= horizon_s:
        raan_a = _propagate_raan(obj_a, t)
        raan_b = _propagate_raan(obj_b, t)
        diff = abs((raan_a - raan_b + 180.0) % 360.0 - 180.0)
        if diff < 5.0:
            return t
        t += dt
    return None


def _altitude_shell_overlap(obj_a: TrackedObject, obj_b: TrackedObject) -> bool:
    """True if the two objects share an altitude shell within ±50 km."""
    return abs(obj_a.altitude_km - obj_b.altitude_km) < 50.0


def _compute_conjunction_risk_score(
    miss_km: float,
    rel_speed_km_s: float,
    obj_a: TrackedObject,
    obj_b: TrackedObject,
) -> float:
    """
    A-DRI-inspired scalar risk score in [0, 1] for a conjunction event.
    Uses: miss distance, relative speed, altitude, area-to-mass ratio.
    """
    # Collision probability proxy (Gaussian hard-body model, sigma ~ 0.3 km)
    sigma = 0.30
    hb    = 0.015  # combined hard-body radius km
    pc = math.exp(-0.5 * ((miss_km - hb) / max(sigma, 0.01))**2) if miss_km > hb else 1.0
    pc = max(0.0, min(1.0, pc))

    # Speed factor (> 10 km/s is typical worst-case for head-on LEO)
    speed_factor = min(rel_speed_km_s / 12.0, 1.0)

    # Area/mass weighting (larger A/m = more drag uncertainty)
    atm_a = obj_a.area_m2 / max(obj_a.mass_kg, 1.0)
    atm_b = obj_b.area_m2 / max(obj_b.mass_kg, 1.0)
    atm_factor = min((atm_a + atm_b) / 0.1, 1.0)

    # Altitude factor (denser debris population 550–650 km)
    alt_mid = (obj_a.altitude_km + obj_b.altitude_km) / 2.0
    alt_factor = 1.0 if 500.0 <= alt_mid <= 700.0 else 0.7

    score = 0.50 * pc + 0.25 * speed_factor + 0.15 * atm_factor + 0.10 * alt_factor
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Debris Tracker Engine
# ---------------------------------------------------------------------------

class DebrisTrackerEngine:
    """
    Maintains a catalogue of tracked LEO objects and performs all-pairs
    conjunction screening.

    Typical use::

        tracker = DebrisTrackerEngine()
        tracker.add_object(TrackedObject(...))
        matrix = tracker.compute_risk_matrix(primary_id="SAT-01")
    """

    def __init__(self):
        self._catalogue: Dict[str, TrackedObject] = {}

    # ------------------------------------------------------------------
    # Catalogue management
    # ------------------------------------------------------------------

    def add_object(self, obj: TrackedObject) -> None:
        """Register or update a tracked object."""
        self._catalogue[obj.object_id] = obj

    def remove_object(self, object_id: str) -> bool:
        return self._catalogue.pop(object_id, None) is not None

    def list_objects(self) -> List[TrackedObject]:
        return list(self._catalogue.values())

    @property
    def catalogue_size(self) -> int:
        return len(self._catalogue)

    # ------------------------------------------------------------------
    # Screening core
    # ------------------------------------------------------------------

    def screen_pair(
        self,
        obj_a: TrackedObject,
        obj_b: TrackedObject,
        horizon_s: float = 21600.0,
        is_secondary: bool = False,
    ) -> Optional[ConjunctionEvent]:
        """
        Screen a single object pair.  Returns a ConjunctionEvent if a
        close-approach is detected within the horizon, else None.
        """
        if obj_a.object_id == obj_b.object_id:
            return None

        # Fast shell filter
        if not _altitude_shell_overlap(obj_a, obj_b):
            return None

        # Approximate TCA via RAAN-plane crossing
        tca = _raan_crossing_time(obj_a, obj_b, horizon_s)
        if tca is None:
            # No RAAN crossing; still check for near-co-orbital passes
            inc_diff = abs(obj_a.inclination_deg - obj_b.inclination_deg)
            raan_diff = abs((obj_a.raan_deg - obj_b.raan_deg + 180.0) % 360.0 - 180.0)
            if inc_diff > 10.0 or raan_diff > 20.0:
                return None
            # Co-orbital pass: estimate TCA from mean anomaly separation
            ma_diff = abs((obj_a.mean_anomaly_deg - obj_b.mean_anomaly_deg + 180.0) % 360.0 - 180.0)
            avg_n = (obj_a.mean_motion_deg_s + obj_b.mean_motion_deg_s) / 2.0
            tca = (ma_diff / max(abs(obj_a.mean_motion_deg_s - obj_b.mean_motion_deg_s), 1e-9)) % horizon_s

        # Miss distance approximation
        alt_diff = abs(obj_a.altitude_km - obj_b.altitude_km)
        raan_at_tca_a = _propagate_raan(obj_a, tca)
        raan_at_tca_b = _propagate_raan(obj_b, tca)
        raan_sep = abs((raan_at_tca_a - raan_at_tca_b + 180.0) % 360.0 - 180.0)
        raan_miss = raan_sep * math.radians(1.0) * obj_a.semi_major_axis_km  # km

        miss_km = math.sqrt(alt_diff**2 + raan_miss**2)
        miss_km = max(miss_km, 0.001)

        rel_speed = _relative_velocity_km_s(obj_a, obj_b)
        risk = _compute_conjunction_risk_score(miss_km, rel_speed, obj_a, obj_b)

        # Only report if below warning threshold or risk > 0.2
        if miss_km > DEBRIS_WARNING_DISTANCE_KM * 2.0 and risk < 0.20:
            return None

        is_crit = miss_km < DEBRIS_CRITICAL_DISTANCE_KM or risk >= 0.60
        desc = (
            f"{'[SECONDARY-CHAIN] ' if is_secondary else ''}"
            f"Miss={miss_km:.3f} km at TCA+{tca/3600:.2f}h, "
            f"V_rel={rel_speed:.2f} km/s, Risk={risk:.3f}"
        )

        return ConjunctionEvent(
            object_a_id=obj_a.object_id,
            object_b_id=obj_b.object_id,
            object_a_name=obj_a.name,
            object_b_name=obj_b.name,
            tca_seconds_from_now=round(tca, 1),
            miss_distance_km=round(miss_km, 4),
            relative_speed_km_s=round(rel_speed, 4),
            risk_score=round(risk, 4),
            is_critical=is_crit,
            is_secondary_chain=is_secondary,
            description=desc,
        )

    def compute_risk_matrix(
        self,
        primary_id: str,
        horizon_hours: float = 6.0,
        detect_secondary_chains: bool = True,
    ) -> DebrisFleetRiskMatrix:
        """
        Compute the full fleet risk matrix for all objects relative to
        the primary satellite.

        Parameters
        ----------
        primary_id : str
            Object ID of the primary (active) satellite.
        horizon_hours : float
            Look-ahead horizon in hours.
        detect_secondary_chains : bool
            If True, also check whether a proposed evasive manoeuvre by the
            primary could create new conjunctions with other objects.
        """
        horizon_s = horizon_hours * 3600.0
        primary = self._catalogue.get(primary_id)
        if primary is None:
            raise KeyError(f"Primary object '{primary_id}' not in catalogue.")

        events: List[ConjunctionEvent] = []
        debris_objects = [o for o in self._catalogue.values() if o.object_id != primary_id]

        for deb in debris_objects:
            ev = self.screen_pair(primary, deb, horizon_s=horizon_s, is_secondary=False)
            if ev:
                events.append(ev)

        # Secondary chain detection: for each evasion that raises primary altitude
        # by ~0.5 km, check if the new orbit intersects other objects.
        if detect_secondary_chains and events:
            # Create a "manoeuvred" proxy of the primary
            manoeuvred = TrackedObject(
                object_id=primary.object_id + "_MAN",
                name=primary.name + "_POST_MANOEUVRE",
                altitude_km=primary.altitude_km + 0.5,
                inclination_deg=primary.inclination_deg,
                raan_deg=primary.raan_deg,
                mean_anomaly_deg=primary.mean_anomaly_deg + 0.1,
                mass_kg=primary.mass_kg,
                area_m2=primary.area_m2,
                is_active_satellite=True,
            )
            for deb in debris_objects:
                # Skip objects already flagged as primaries
                if any(e.object_b_id == deb.object_id for e in events):
                    continue
                sec_ev = self.screen_pair(manoeuvred, deb, horizon_s=horizon_s, is_secondary=True)
                if sec_ev:
                    sec_ev.object_a_id = primary.object_id
                    sec_ev.object_a_name = primary.name
                    events.append(sec_ev)

        # Build matrix summary
        critical  = [e for e in events if e.is_critical]
        warnings  = [e for e in events if not e.is_critical]
        secondary = [e for e in events if e.is_secondary_chain]

        top = max(events, key=lambda e: e.risk_score) if events else None
        min_miss = min(e.miss_distance_km for e in events) if events else float("inf")

        matrix = DebrisFleetRiskMatrix(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            primary_satellite_id=primary_id,
            total_tracked_objects=len(self._catalogue),
            total_conjunctions_screened=len(debris_objects),
            critical_conjunctions=len(critical),
            warning_conjunctions=len(warnings),
            highest_risk_score=round(top.risk_score, 4) if top else 0.0,
            highest_risk_pairing=f"{top.object_a_name} ↔ {top.object_b_name}" if top else "NONE",
            min_miss_distance_km=round(min_miss, 4) if events else 0.0,
            conjunction_events=sorted(events, key=lambda e: e.risk_score, reverse=True),
            secondary_chain_risk=len(secondary) > 0,
            summary={
                "primary_id": primary_id,
                "horizon_hours": horizon_hours,
                "catalogue_size": self.catalogue_size,
                "events_found": len(events),
                "critical_count": len(critical),
                "secondary_chain_count": len(secondary),
            },
        )
        return matrix


# ---------------------------------------------------------------------------
# Synthetic catalogue builders
# ---------------------------------------------------------------------------

def _random_debris_object(altitude_km: float, seed_offset: int = 0) -> TrackedObject:
    rng = np.random.default_rng(42 + seed_offset)
    return TrackedObject(
        object_id=f"DEB-{uuid.uuid4().hex[:8].upper()}",
        name=f"DEBRIS-LEO-{seed_offset:03d}",
        altitude_km=altitude_km + rng.uniform(-40.0, 40.0),
        inclination_deg=rng.uniform(45.0, 100.0),
        raan_deg=rng.uniform(0.0, 360.0),
        eccentricity=rng.uniform(0.0, 0.015),
        arg_perigee_deg=rng.uniform(0.0, 360.0),
        mean_anomaly_deg=rng.uniform(0.0, 360.0),
        mass_kg=rng.uniform(0.1, 500.0),
        area_m2=rng.uniform(0.01, 5.0),
        is_active_satellite=False,
    )


def build_synthetic_catalogue(
    primary: TrackedObject,
    num_debris: int = 50,
) -> DebrisTrackerEngine:
    """
    Build a DebrisTrackerEngine pre-populated with *primary* and
    *num_debris* randomly generated debris objects clustered around
    the primary's altitude.
    """
    engine = DebrisTrackerEngine()
    engine.add_object(primary)
    for i in range(num_debris):
        engine.add_object(_random_debris_object(primary.altitude_km, seed_offset=i))
    return engine


def build_debris_risk_matrix(
    satellite_id: str,
    altitude_km: float,
    inclination_deg: float,
    num_debris: int = 80,
    horizon_hours: float = 6.0,
) -> DebrisFleetRiskMatrix:
    """
    Convenience function: creates a synthetic catalogue and returns the
    risk matrix for the given primary satellite parameters.
    """
    primary = TrackedObject(
        object_id=satellite_id,
        name=satellite_id,
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        mass_kg=260.0,
        area_m2=2.2,
        is_active_satellite=True,
    )
    engine = build_synthetic_catalogue(primary, num_debris=num_debris)
    return engine.compute_risk_matrix(primary_id=satellite_id, horizon_hours=horizon_hours)
