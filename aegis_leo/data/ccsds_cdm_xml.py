"""
CCSDS 508.0-B-1 Conjunction Data Message (CDM) XML and KVN Parser, Validator & Exporter.

Compliant with CCSDS 508.0-B-1 (Blue Book, Conjunction Data Message) standards.
Provides:
- Complete parsing of XML and KVN formats using standard library xml.etree.
- Comprehensive schema and astrodynamical validation (positive-definite covariance, bounds).
- Full 6x6 and 3x3 covariance matrix ingestion, symmetrization, and combined RTN covariance.
- Rigorous miss distance and relative speed verification against state vectors.
- Screening volume penetration evaluation (Sphere, Box, Ellipsoid).
- Robust ISO 8601 / CCSDS TCA and epoch timestamp parsing with microsecond precision.
- Export to standards-compliant XML and KVN.
"""

from __future__ import annotations

import re
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET
import numpy as np


# ---------------------------------------------------------------------------
# Timestamp Utilities
# ---------------------------------------------------------------------------

def parse_ccsds_timestamp(ts_str: str) -> datetime:
    """
    Parse a CCSDS or ISO 8601 timestamp string into a timezone-aware UTC datetime.
    Supports formats:
    - YYYY-MM-DDTHH:MM:SS
    - YYYY-MM-DDTHH:MM:SS.ffffff
    - YYYY-MM-DDTHH:MM:SS.fff
    - YYYY-MM-DDTHH:MM:SSZ / with timezone offsets (+/-HH:MM)
    - Day of year format: YYYY-DDDTHH:MM:SS[.fff]
    """
    if not ts_str:
        raise ValueError("Timestamp string cannot be empty")

    s = ts_str.strip()

    # Handle day-of-year format: YYYY-DDDTHH:MM:SS...
    doy_match = re.match(r"^(\d{4})-(\d{3})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?Z?$", s)
    if doy_match:
        year = int(doy_match.group(1))
        doy = int(doy_match.group(2))
        hour = int(doy_match.group(3))
        minute = int(doy_match.group(4))
        second = int(doy_match.group(5))
        microsec = 0
        if doy_match.group(6):
            frac = doy_match.group(6)
            microsec = int(frac[:6].ljust(6, "0"))
        base_dt = datetime(year, 1, 1, tzinfo=timezone.utc)
        return base_dt + timedelta(days=doy - 1, hours=hour, minutes=minute, seconds=second, microseconds=microsec)

    # Standard ISO-8601 parsing
    # Normalize trailing Z to +00:00 for datetime.fromisoformat
    if s.endswith("Z") or s.endswith("z"):
        s = s[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception as e:
        raise ValueError(f"Unable to parse CCSDS timestamp '{ts_str}': {e}") from e


def format_ccsds_timestamp(dt: datetime, include_subseconds: bool = True) -> str:
    """Format a datetime into CCSDS UTC format: YYYY-MM-DDTHH:MM:SS[.fff]"""
    utc_dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    if include_subseconds:
        millis = utc_dt.microsecond // 1000
        return utc_dt.strftime(f"%Y-%m-%dT%H:%M:%S.{millis:03d}")
    return utc_dt.strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class CDMHeader:
    """CCSDS CDM Header information."""
    ccsds_cdm_vers: str = "1.0"
    creation_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    originator: str = "AEGIS-LEO SSA SHIELD"
    message_for: Optional[str] = None
    message_id: str = field(default_factory=lambda: f"CDM-{uuid.uuid4().hex[:8].upper()}")
    comment: List[str] = field(default_factory=list)


@dataclass
class CDMRelativeMetadata:
    """CCSDS CDM Relative Metadata and Conjunction Summary Data."""
    tca: datetime
    miss_distance: float  # meters
    relative_speed: Optional[float] = None  # m/s
    relative_position_r: Optional[float] = None  # meters
    relative_position_t: Optional[float] = None  # meters
    relative_position_n: Optional[float] = None  # meters
    relative_velocity_r: Optional[float] = None  # m/s
    relative_velocity_t: Optional[float] = None  # m/s
    relative_velocity_n: Optional[float] = None  # m/s
    start_screen_volume: Optional[datetime] = None
    stop_screen_volume: Optional[datetime] = None
    screen_volume_frame: Optional[str] = "RTN"
    screen_volume_shape: Optional[str] = "ELLIPSOID"  # ELLIPSOID, BOX, SPHERE
    screen_volume_x: Optional[float] = None  # meters (semi-major axis or half-width)
    screen_volume_y: Optional[float] = None  # meters
    screen_volume_z: Optional[float] = None  # meters
    screen_volume_radius: Optional[float] = None  # meters for SPHERE
    screen_entry_time: Optional[datetime] = None
    screen_exit_time: Optional[datetime] = None
    collision_probability: Optional[float] = None
    collision_probability_method: Optional[str] = None
    collision_percentile: Optional[float] = None
    screening_threshold: Optional[float] = None
    comment: List[str] = field(default_factory=list)

    def relative_position_vector(self) -> Optional[np.ndarray]:
        """Return relative position vector [R, T, N] in meters if available."""
        if (
            self.relative_position_r is not None
            and self.relative_position_t is not None
            and self.relative_position_n is not None
        ):
            return np.array([self.relative_position_r, self.relative_position_t, self.relative_position_n], dtype=float)
        return None

    def relative_velocity_vector(self) -> Optional[np.ndarray]:
        """Return relative velocity vector [R, T, N] in m/s if available."""
        if (
            self.relative_velocity_r is not None
            and self.relative_velocity_t is not None
            and self.relative_velocity_n is not None
        ):
            return np.array([self.relative_velocity_r, self.relative_velocity_t, self.relative_velocity_n], dtype=float)
        return None

    def is_position_in_screening_volume(self, rel_pos_m: Union[np.ndarray, List[float], Tuple[float, float, float]]) -> bool:
        """
        Check if a given relative position vector [x, y, z] in meters is inside
        the configured screening volume.
        """
        pos = np.asarray(rel_pos_m, dtype=float)
        if len(pos) != 3:
            raise ValueError("Position vector must have 3 elements [x, y, z]")

        shape = (self.screen_volume_shape or "ELLIPSOID").upper()

        if shape == "SPHERE":
            rad = self.screen_volume_radius or self.screen_volume_x
            if rad is None or rad <= 0.0:
                return True
            return float(np.linalg.norm(pos)) <= rad

        elif shape == "BOX":
            sx = self.screen_volume_x
            sy = self.screen_volume_y
            sz = self.screen_volume_z
            if sx is not None and abs(pos[0]) > sx / 2.0:
                return False
            if sy is not None and abs(pos[1]) > sy / 2.0:
                return False
            if sz is not None and abs(pos[2]) > sz / 2.0:
                return False
            return True

        elif shape == "ELLIPSOID":
            a = self.screen_volume_x
            b = self.screen_volume_y
            c = self.screen_volume_z
            if not a or not b or not c:
                return True
            norm_dist_sq = (pos[0] / a) ** 2 + (pos[1] / b) ** 2 + (pos[2] / c) ** 2
            return norm_dist_sq <= 1.0

        return True


@dataclass
class CDMCovariance:
    """
    CCSDS CDM 6x6 (or 3x3 position) Covariance Matrix in RTN or specified reference frame.
    All lower triangular elements defined in CCSDS 508.0-B-1:
    Position terms: [m**2]
    Cross terms: [m**2/s]
    Velocity terms: [m**2/s**2]
    """
    ref_frame: str = "RTN"
    # Position 3x3 lower triangle
    cr_r: Optional[float] = None
    ct_r: Optional[float] = None
    ct_t: Optional[float] = None
    cn_r: Optional[float] = None
    cn_t: Optional[float] = None
    cn_n: Optional[float] = None

    # Velocity cross terms (Row 4: R_DOT)
    crdot_r: Optional[float] = None
    crdot_t: Optional[float] = None
    crdot_n: Optional[float] = None
    crdot_rdot: Optional[float] = None

    # Velocity cross terms (Row 5: T_DOT)
    ctdot_r: Optional[float] = None
    ctdot_t: Optional[float] = None
    ctdot_n: Optional[float] = None
    ctdot_rdot: Optional[float] = None
    ctdot_tdot: Optional[float] = None

    # Velocity cross terms (Row 6: N_DOT)
    cndot_r: Optional[float] = None
    cndot_t: Optional[float] = None
    cndot_n: Optional[float] = None
    cndot_rdot: Optional[float] = None
    cndot_tdot: Optional[float] = None
    cndot_ndot: Optional[float] = None

    def to_pos_covariance_matrix(self) -> np.ndarray:
        """
        Reconstruct the 3x3 symmetric position covariance matrix in meters^2.
        Order: [R, T, N]
        """
        cov3 = np.zeros((3, 3), dtype=float)
        cr_r = self.cr_r or 0.0
        ct_r = self.ct_r or 0.0
        ct_t = self.ct_t or 0.0
        cn_r = self.cn_r or 0.0
        cn_t = self.cn_t or 0.0
        cn_n = self.cn_n or 0.0

        cov3[0, 0] = cr_r
        cov3[1, 0] = ct_r
        cov3[0, 1] = ct_r
        cov3[1, 1] = ct_t
        cov3[2, 0] = cn_r
        cov3[0, 2] = cn_r
        cov3[2, 1] = cn_t
        cov3[1, 2] = cn_t
        cov3[2, 2] = cn_n
        return cov3

    def to_full_covariance_matrix(self) -> np.ndarray:
        """
        Reconstruct the full 6x6 symmetric covariance matrix:
        Order: [R, T, N, R_DOT, T_DOT, N_DOT]
        Units: pos in m^2, pos-vel in m^2/s, vel in m^2/s^2
        """
        cov6 = np.zeros((6, 6), dtype=float)
        # Position 3x3
        cov6[:3, :3] = self.to_pos_covariance_matrix()

        # Row 3 (RDOT)
        crdot_r = self.crdot_r or 0.0
        crdot_t = self.crdot_t or 0.0
        crdot_n = self.crdot_n or 0.0
        crdot_rdot = self.crdot_rdot or 0.0
        cov6[3, 0] = cov6[0, 3] = crdot_r
        cov6[3, 1] = cov6[1, 3] = crdot_t
        cov6[3, 2] = cov6[2, 3] = crdot_n
        cov6[3, 3] = crdot_rdot

        # Row 4 (TDOT)
        ctdot_r = self.ctdot_r or 0.0
        ctdot_t = self.ctdot_t or 0.0
        ctdot_n = self.ctdot_n or 0.0
        ctdot_rdot = self.ctdot_rdot or 0.0
        ctdot_tdot = self.ctdot_tdot or 0.0
        cov6[4, 0] = cov6[0, 4] = ctdot_r
        cov6[4, 1] = cov6[1, 4] = ctdot_t
        cov6[4, 2] = cov6[2, 4] = ctdot_n
        cov6[4, 3] = cov6[3, 4] = ctdot_rdot
        cov6[4, 4] = ctdot_tdot

        # Row 5 (NDOT)
        cndot_r = self.cndot_r or 0.0
        cndot_t = self.cndot_t or 0.0
        cndot_n = self.cndot_n or 0.0
        cndot_rdot = self.cndot_rdot or 0.0
        cndot_tdot = self.cndot_tdot or 0.0
        cndot_ndot = self.cndot_ndot or 0.0
        cov6[5, 0] = cov6[0, 5] = cndot_r
        cov6[5, 1] = cov6[1, 5] = cndot_t
        cov6[5, 2] = cov6[2, 5] = cndot_n
        cov6[5, 3] = cov6[3, 5] = cndot_rdot
        cov6[5, 4] = cov6[4, 5] = cndot_tdot
        cov6[5, 5] = cndot_ndot

        return cov6

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, ref_frame: str = "RTN") -> CDMCovariance:
        """Create CDMCovariance from 3x3 or 6x6 numpy array."""
        m = np.asarray(matrix, dtype=float)
        if m.shape == (3, 3):
            return cls(
                ref_frame=ref_frame,
                cr_r=float(m[0, 0]),
                ct_r=float(m[1, 0]),
                ct_t=float(m[1, 1]),
                cn_r=float(m[2, 0]),
                cn_t=float(m[2, 1]),
                cn_n=float(m[2, 2]),
            )
        elif m.shape == (6, 6):
            return cls(
                ref_frame=ref_frame,
                cr_r=float(m[0, 0]),
                ct_r=float(m[1, 0]),
                ct_t=float(m[1, 1]),
                cn_r=float(m[2, 0]),
                cn_t=float(m[2, 1]),
                cn_n=float(m[2, 2]),
                crdot_r=float(m[3, 0]),
                crdot_t=float(m[3, 1]),
                crdot_n=float(m[3, 2]),
                crdot_rdot=float(m[3, 3]),
                ctdot_r=float(m[4, 0]),
                ctdot_t=float(m[4, 1]),
                ctdot_n=float(m[4, 2]),
                ctdot_rdot=float(m[4, 3]),
                ctdot_tdot=float(m[4, 4]),
                cndot_r=float(m[5, 0]),
                cndot_t=float(m[5, 1]),
                cndot_n=float(m[5, 2]),
                cndot_rdot=float(m[5, 3]),
                cndot_tdot=float(m[5, 4]),
                cndot_ndot=float(m[5, 5]),
            )
        else:
            raise ValueError(f"Matrix shape must be (3,3) or (6,6), got {m.shape}")

    def is_positive_semidefinite(self, tol: float = -1e-6) -> Tuple[bool, float]:
        """
        Check whether the 3x3 position covariance is positive semi-definite.
        Returns (is_psd, min_eigenvalue).
        """
        cov = self.to_pos_covariance_matrix()
        if np.all(cov == 0.0):
            return False, 0.0
        eigvals = np.linalg.eigvalsh(cov)
        min_eig = float(np.min(eigvals))
        return (min_eig >= tol), min_eig

    def one_sigma_position_errors(self) -> Tuple[float, float, float]:
        """Returns 1-sigma standard deviations in R, T, N (meters)."""
        sig_r = math.sqrt(max(0.0, self.cr_r or 0.0))
        sig_t = math.sqrt(max(0.0, self.ct_t or 0.0))
        sig_n = math.sqrt(max(0.0, self.cn_n or 0.0))
        return sig_r, sig_t, sig_n


@dataclass
class CDMStateVector:
    """State vector at epoch in km and km/s."""
    epoch: Optional[datetime] = None
    x: Optional[float] = None  # km
    y: Optional[float] = None  # km
    z: Optional[float] = None  # km
    x_dot: Optional[float] = None  # km/s
    y_dot: Optional[float] = None  # km/s
    z_dot: Optional[float] = None  # km/s

    def position_km(self) -> np.ndarray:
        """Position vector in km."""
        return np.array([self.x or 0.0, self.y or 0.0, self.z or 0.0], dtype=float)

    def velocity_km_s(self) -> np.ndarray:
        """Velocity vector in km/s."""
        return np.array([self.x_dot or 0.0, self.y_dot or 0.0, self.z_dot or 0.0], dtype=float)

    def position_m(self) -> np.ndarray:
        """Position vector in meters."""
        return self.position_km() * 1000.0

    def velocity_m_s(self) -> np.ndarray:
        """Velocity vector in m/s."""
        return self.velocity_km_s() * 1000.0


@dataclass
class CDMObjectMetadata:
    """Metadata describing an object in the conjunction."""
    object: str = "OBJECT1"  # OBJECT1 or OBJECT2
    object_designator: str = ""
    catalog_name: Optional[str] = "SATCAT"
    object_name: Optional[str] = None
    international_designator: Optional[str] = None
    object_type: Optional[str] = "PAYLOAD"  # PAYLOAD, ROCKET BODY, DEBRIS, UNKNOWN, OTHER
    operator_contact_position: Optional[str] = None
    operator_organization: Optional[str] = None
    operator_phone: Optional[str] = None
    operator_email: Optional[str] = None
    ephemeris_name: Optional[str] = None
    covariance_method: Optional[str] = "CALCULATED"
    maneuverable: Optional[str] = "YES"  # YES, NO, UNKNOWN
    orbit_center: Optional[str] = "EARTH"
    ref_frame: Optional[str] = "EME2000"
    gravity_model: Optional[str] = None
    atmospheric_model: Optional[str] = None
    n_body_perturbations: Optional[str] = None
    solar_rad_pressure: Optional[str] = None
    earth_tides: Optional[str] = None
    intrack_thrust: Optional[str] = None
    comment: List[str] = field(default_factory=list)


@dataclass
class CDMObjectData:
    """Data section for an object in the conjunction."""
    time_lastob_start: Optional[datetime] = None
    time_lastob_end: Optional[datetime] = None
    recommended_od_span: Optional[float] = None  # days
    actual_od_span: Optional[float] = None  # days
    obs_available: Optional[int] = None
    obs_used: Optional[int] = None
    tracks_available: Optional[int] = None
    tracks_used: Optional[int] = None
    residuals_accepted: Optional[float] = None  # %
    weighted_rms: Optional[float] = None
    spherical_coefficients: Optional[float] = None
    cd_area_over_mass: Optional[float] = None  # m**2/kg
    cr_area_over_mass: Optional[float] = None  # m**2/kg
    thrust_acceleration: Optional[float] = None  # m/s**2
    sedr: Optional[float] = None  # W/kg
    state_vector: Optional[CDMStateVector] = None
    covariance: Optional[CDMCovariance] = None
    comment: List[str] = field(default_factory=list)


@dataclass
class CDMObject:
    """Full representation of a single object (metadata + data) in a CDM."""
    metadata: CDMObjectMetadata
    data: CDMObjectData = field(default_factory=CDMObjectData)


@dataclass
class ValidationResult:
    """Result of schema and physical validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


@dataclass
class CDMMessage:
    """
    Complete CCSDS Conjunction Data Message (CDM).
    Encapsulates Header, Relative Metadata/Data, Object 1, and Object 2.
    """
    header: CDMHeader
    relative_metadata: CDMRelativeMetadata
    object1: CDMObject
    object2: CDMObject
    comment: List[str] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Miss Distance & Relative Kinematics Verification
    # -----------------------------------------------------------------------

    def verify_miss_distance(self, tolerance_m: float = 1.0) -> Tuple[bool, float, float]:
        """
        Verify the recorded miss distance against computed Euclidean distance
        between Object 1 and Object 2 state vectors (or relative position vector).

        Returns:
            (is_consistent, computed_miss_m, recorded_miss_m)
        """
        recorded = float(self.relative_metadata.miss_distance)
        computed = None

        # Check relative position vector in relative metadata first
        rel_pos = self.relative_metadata.relative_position_vector()
        if rel_pos is not None:
            computed = float(np.linalg.norm(rel_pos))
        elif (
            self.object1.data.state_vector is not None
            and self.object2.data.state_vector is not None
        ):
            r1 = self.object1.data.state_vector.position_m()
            r2 = self.object2.data.state_vector.position_m()
            computed = float(np.linalg.norm(r1 - r2))

        if computed is None:
            return True, recorded, recorded

        is_consistent = abs(computed - recorded) <= tolerance_m
        return is_consistent, computed, recorded

    def verify_relative_speed(self, tolerance_m_s: float = 0.1) -> Tuple[bool, float, float]:
        """
        Verify recorded relative speed against computed relative speed
        from state vectors or relative velocity vector.

        Returns:
            (is_consistent, computed_speed_m_s, recorded_speed_m_s)
        """
        recorded = self.relative_metadata.relative_speed
        if recorded is None:
            return True, 0.0, 0.0

        computed = None
        rel_vel = self.relative_metadata.relative_velocity_vector()
        if rel_vel is not None:
            computed = float(np.linalg.norm(rel_vel))
        elif (
            self.object1.data.state_vector is not None
            and self.object2.data.state_vector is not None
        ):
            v1 = self.object1.data.state_vector.velocity_m_s()
            v2 = self.object2.data.state_vector.velocity_m_s()
            computed = float(np.linalg.norm(v1 - v2))

        if computed is None:
            return True, recorded, recorded

        is_consistent = abs(computed - recorded) <= tolerance_m_s
        return is_consistent, computed, recorded

    # -----------------------------------------------------------------------
    # Combined Covariance Matrix
    # -----------------------------------------------------------------------

    def combined_pos_covariance_rtn(self) -> Optional[np.ndarray]:
        """
        Compute combined 3x3 position covariance matrix C_combined = C1 + C2 in RTN frame.
        Returns None if either covariance is missing.
        """
        cov1 = self.object1.data.covariance
        cov2 = self.object2.data.covariance
        if cov1 is None or cov2 is None:
            return None
        return cov1.to_pos_covariance_matrix() + cov2.to_pos_covariance_matrix()

    # -----------------------------------------------------------------------
    # Screening Volume Checks
    # -----------------------------------------------------------------------

    def check_screening_volume_penetration(self) -> Dict[str, Any]:
        """
        Assess whether conjunction relative state enters the screening volume.
        Returns diagnostic dictionary.
        """
        rel = self.relative_metadata
        rel_pos = rel.relative_position_vector()
        if rel_pos is None and (
            self.object1.data.state_vector is not None
            and self.object2.data.state_vector is not None
        ):
            rel_pos = (
                self.object2.data.state_vector.position_m()
                - self.object1.data.state_vector.position_m()
            )

        if rel_pos is None:
            # Fall back to scalar miss distance
            rel_pos = np.array([rel.miss_distance, 0.0, 0.0])

        is_inside = rel.is_position_in_screening_volume(rel_pos)
        return {
            "is_penetrating": is_inside,
            "shape": rel.screen_volume_shape,
            "frame": rel.screen_volume_frame,
            "relative_position_m": rel_pos.tolist(),
            "miss_distance_m": rel.miss_distance,
            "entry_time": rel.screen_entry_time.isoformat() if rel.screen_entry_time else None,
            "exit_time": rel.screen_exit_time.isoformat() if rel.screen_exit_time else None,
        }

    # -----------------------------------------------------------------------
    # Schema & Consistency Validation
    # -----------------------------------------------------------------------

    def validate(self) -> ValidationResult:
        """Run full schema and astrodynamical validation."""
        errors: List[str] = []
        warnings: List[str] = []

        # Header checks
        if not self.header.ccsds_cdm_vers:
            errors.append("Header CCSDS_CDM_VERS is required.")
        elif self.header.ccsds_cdm_vers not in ("1.0", "2.0"):
            warnings.append(f"Uncommon CCSDS_CDM_VERS '{self.header.ccsds_cdm_vers}' (expected '1.0').")

        if not self.header.originator:
            errors.append("Header ORIGINATOR is required.")
        if not self.header.message_id:
            errors.append("Header MESSAGE_ID is required.")

        # Relative metadata checks
        if self.relative_metadata.tca is None:
            errors.append("Relative metadata TCA is required.")
        if self.relative_metadata.miss_distance is None:
            errors.append("Relative metadata MISS_DISTANCE is required.")
        elif self.relative_metadata.miss_distance < 0.0:
            errors.append(f"MISS_DISTANCE must be non-negative, got {self.relative_metadata.miss_distance}.")

        if self.relative_metadata.relative_speed is not None and self.relative_metadata.relative_speed < 0.0:
            errors.append("RELATIVE_SPEED must be non-negative.")

        pc = self.relative_metadata.collision_probability
        if pc is not None and not (0.0 <= pc <= 1.0):
            errors.append(f"COLLISION_PROBABILITY must be between 0.0 and 1.0, got {pc}.")

        # Objects check
        for obj_idx, obj in enumerate((self.object1, self.object2), start=1):
            lbl = f"Object {obj_idx}"
            if not obj.metadata.object_designator and not obj.metadata.object_name:
                errors.append(f"{lbl} requires at least OBJECT_DESIGNATOR or OBJECT_NAME.")

            # Covariance checks
            cov = obj.data.covariance
            if cov is not None:
                # Diagonal terms must be non-negative
                for diag_name, diag_val in (("CR_R", cov.cr_r), ("CT_T", cov.ct_t), ("CN_N", cov.cn_n)):
                    if diag_val is not None and diag_val < 0.0:
                        errors.append(f"{lbl} covariance diagonal {diag_name} cannot be negative ({diag_val}).")

                # Positive semi-definiteness
                is_psd, min_eig = cov.is_positive_semidefinite()
                if not is_psd and min_eig < -1e-4:
                    warnings.append(
                        f"{lbl} covariance matrix is not positive semi-definite (min eigenvalue: {min_eig:.4e})."
                    )

        # Consistency verification
        miss_ok, calc_m, rec_m = self.verify_miss_distance(tolerance_m=5.0)
        if not miss_ok:
            warnings.append(f"Reported MISS_DISTANCE ({rec_m:.2f} m) diverges from state vectors ({calc_m:.2f} m).")

        return ValidationResult(is_valid=(len(errors) == 0), errors=errors, warnings=warnings)

    # -----------------------------------------------------------------------
    # Exporters
    # -----------------------------------------------------------------------

    def to_xml(self, pretty: bool = True) -> str:
        """Export CDM message to CCSDS 508.0-B-1 XML string."""
        return export_cdm_xml(self, pretty_print=pretty)

    def to_kvn(self) -> str:
        """Export CDM message to CCSDS 508.0-B-1 KVN string."""
        return export_cdm_kvn(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Python dictionary representation compatible with AEGIS-LEO."""
        cov_rtn = {}
        if self.object1.data.covariance:
            cov = self.object1.data.covariance
            cov_rtn = {
                "c_rr": cov.cr_r or 0.0,
                "c_tt": cov.ct_t or 0.0,
                "c_nn": cov.cn_n or 0.0,
                "c_rt": cov.ct_r or 0.0,
                "c_rn": cov.cn_r or 0.0,
                "c_tn": cov.cn_t or 0.0,
            }

        return {
            "ccsds_cdm_vers": self.header.ccsds_cdm_vers,
            "message_id": self.header.message_id,
            "creation_date": self.header.creation_date.isoformat(),
            "originator": self.header.originator,
            "tca": self.relative_metadata.tca.isoformat(),
            "miss_distance_m": round(self.relative_metadata.miss_distance, 2),
            "relative_speed_m_s": round(self.relative_metadata.relative_speed or 0.0, 2),
            "collision_probability": self.relative_metadata.collision_probability,
            "collision_probability_method": self.relative_metadata.collision_probability_method,
            "primary_object": {
                "object_id": self.object1.metadata.object_designator,
                "object_name": self.object1.metadata.object_name,
                "object_type": self.object1.metadata.object_type,
            },
            "secondary_object": {
                "object_id": self.object2.metadata.object_designator,
                "object_name": self.object2.metadata.object_name,
                "object_type": self.object2.metadata.object_type,
            },
            "covariance_matrix_rtn_m2": cov_rtn,
            "alert_status": (
                "CRITICAL"
                if (self.relative_metadata.collision_probability or 0.0) > 1e-4
                else "NOMINAL"
            ),
        }


# ---------------------------------------------------------------------------
# XML Parsing & Exporting
# ---------------------------------------------------------------------------

def _strip_ns(tag: str) -> str:
    """Strip XML namespace prefix if present."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_text(element: Optional[ET.Element], *tag_names: str) -> Optional[str]:
    """Find text in element searching case-insensitively or matching multiple aliases."""
    if element is None:
        return None

    # First try direct child lookup
    for child in element:
        clean_tag = _strip_ns(child.tag).upper()
        for target in tag_names:
            if clean_tag == target.upper():
                return child.text.strip() if child.text else None

    # Search descendants if not found directly
    for descendant in element.iter():
        clean_tag = _strip_ns(descendant.tag).upper()
        for target in tag_names:
            if clean_tag == target.upper():
                return descendant.text.strip() if descendant.text else None
    return None


def _find_float(element: Optional[ET.Element], *tag_names: str) -> Optional[float]:
    val = _find_text(element, *tag_names)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            return None
    return None


def _find_int(element: Optional[ET.Element], *tag_names: str) -> Optional[int]:
    val = _find_text(element, *tag_names)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            return None
    return None


def _find_datetime(element: Optional[ET.Element], *tag_names: str) -> Optional[datetime]:
    val = _find_text(element, *tag_names)
    if val is not None:
        try:
            return parse_ccsds_timestamp(val)
        except ValueError:
            return None
    return None


def parse_cdm_xml(xml_content: Union[str, bytes]) -> CDMMessage:
    """
    Parse a CCSDS 508.0-B-1 CDM XML string or byte string.
    Compliant with official CCSDS XML schema and common variations.
    """
    if isinstance(xml_content, str):
        root = ET.fromstring(xml_content)
    else:
        root = ET.fromstring(xml_content.decode("utf-8"))

    # Extract Header
    header_elem = None
    for child in root:
        if _strip_ns(child.tag).upper() == "HEADER":
            header_elem = child
            break

    vers = root.attrib.get("version", root.attrib.get("id", "1.0"))
    if vers.startswith("CCSDS_CDM_VERS"):
        vers = root.attrib.get("version", "1.0")

    header = CDMHeader(
        ccsds_cdm_vers=_find_text(header_elem, "CCSDS_CDM_VERS") or vers or "1.0",
        creation_date=_find_datetime(header_elem, "CREATION_DATE") or datetime.now(timezone.utc),
        originator=_find_text(header_elem, "ORIGINATOR") or "UNKNOWN",
        message_for=_find_text(header_elem, "MESSAGE_FOR"),
        message_id=_find_text(header_elem, "MESSAGE_ID") or f"CDM-{uuid.uuid4().hex[:8].upper()}",
    )
    if header_elem is not None:
        for c in header_elem.findall(".//COMMENT"):
            if c.text:
                header.comment.append(c.text.strip())

    # Locate body or relative metadata
    body_elem = None
    for child in root:
        if _strip_ns(child.tag).upper() == "BODY":
            body_elem = child
            break

    search_root = body_elem if body_elem is not None else root

    rel_elem = None
    for child in search_root.iter():
        tag = _strip_ns(child.tag).upper()
        if tag in ("RELATIVEMETADATA", "RELATIVEMETADATADATA", "RELATIVE_METADATA_DATA"):
            rel_elem = child
            break

    rel_ctx = rel_elem if rel_elem is not None else search_root

    # Parse Relative Metadata / Data
    tca = _find_datetime(rel_ctx, "TCA")
    if tca is None:
        raise ValueError("Missing mandatory relative metadata field: TCA")

    miss_dist = _find_float(rel_ctx, "MISS_DISTANCE")
    if miss_dist is None:
        raise ValueError("Missing mandatory relative metadata field: MISS_DISTANCE")

    relative_metadata = CDMRelativeMetadata(
        tca=tca,
        miss_distance=miss_dist,
        relative_speed=_find_float(rel_ctx, "RELATIVE_SPEED"),
        relative_position_r=_find_float(rel_ctx, "RELATIVE_POSITION_R"),
        relative_position_t=_find_float(rel_ctx, "RELATIVE_POSITION_T"),
        relative_position_n=_find_float(rel_ctx, "RELATIVE_POSITION_N"),
        relative_velocity_r=_find_float(rel_ctx, "RELATIVE_VELOCITY_R"),
        relative_velocity_t=_find_float(rel_ctx, "RELATIVE_VELOCITY_T"),
        relative_velocity_n=_find_float(rel_ctx, "RELATIVE_VELOCITY_N"),
        start_screen_volume=_find_datetime(rel_ctx, "START_SCREEN_VOLUME"),
        stop_screen_volume=_find_datetime(rel_ctx, "STOP_SCREEN_VOLUME"),
        screen_volume_frame=_find_text(rel_ctx, "SCREEN_VOLUME_FRAME") or "RTN",
        screen_volume_shape=_find_text(rel_ctx, "SCREEN_VOLUME_SHAPE") or "ELLIPSOID",
        screen_volume_x=_find_float(rel_ctx, "SCREEN_VOLUME_X", "SCREEN_VOLUME_RADIUS"),
        screen_volume_y=_find_float(rel_ctx, "SCREEN_VOLUME_Y"),
        screen_volume_z=_find_float(rel_ctx, "SCREEN_VOLUME_Z"),
        screen_volume_radius=_find_float(rel_ctx, "SCREEN_VOLUME_RADIUS"),
        screen_entry_time=_find_datetime(rel_ctx, "SCREEN_ENTRY_TIME"),
        screen_exit_time=_find_datetime(rel_ctx, "SCREEN_EXIT_TIME"),
        collision_probability=_find_float(rel_ctx, "COLLISION_PROBABILITY"),
        collision_probability_method=_find_text(rel_ctx, "COLLISION_PROBABILITY_METHOD"),
        collision_percentile=_find_float(rel_ctx, "COLLISION_PERCENTILE"),
        screening_threshold=_find_float(rel_ctx, "SCREENING_THRESHOLD"),
    )
    if rel_elem is not None:
        for c in rel_elem.findall(".//COMMENT"):
            if c.text:
                relative_metadata.comment.append(c.text.strip())

    # Locate segments (Object 1 and Object 2)
    segments = []
    for elem in search_root.iter():
        if _strip_ns(elem.tag).upper() == "SEGMENT":
            segments.append(elem)

    if len(segments) < 2:
        # Check if objects are tagged directly as OBJECT1 and OBJECT2
        obj1_elem = search_root.find(".//OBJECT1")
        obj2_elem = search_root.find(".//OBJECT2")
        if obj1_elem is not None and obj2_elem is not None:
            segments = [obj1_elem, obj2_elem]
        else:
            raise ValueError(f"CCSDS CDM requires at least 2 object segments, found {len(segments)}")

    def _parse_segment(seg_elem: ET.Element, default_obj_name: str) -> CDMObject:
        meta_elem = None
        data_elem = None
        for child in seg_elem:
            tag = _strip_ns(child.tag).upper()
            if tag == "METADATA":
                meta_elem = child
            elif tag == "DATA":
                data_elem = child

        m_ctx = meta_elem if meta_elem is not None else seg_elem
        d_ctx = data_elem if data_elem is not None else seg_elem

        # Metadata
        obj_meta = CDMObjectMetadata(
            object=_find_text(m_ctx, "OBJECT") or default_obj_name,
            object_designator=_find_text(m_ctx, "OBJECT_DESIGNATOR") or "",
            catalog_name=_find_text(m_ctx, "CATALOG_NAME") or "SATCAT",
            object_name=_find_text(m_ctx, "OBJECT_NAME"),
            international_designator=_find_text(m_ctx, "INTERNATIONAL_DESIGNATOR"),
            object_type=_find_text(m_ctx, "OBJECT_TYPE") or "PAYLOAD",
            operator_contact_position=_find_text(m_ctx, "OPERATOR_CONTACT_POSITION"),
            operator_organization=_find_text(m_ctx, "OPERATOR_ORGANIZATION"),
            operator_phone=_find_text(m_ctx, "OPERATOR_PHONE"),
            operator_email=_find_text(m_ctx, "OPERATOR_EMAIL"),
            ephemeris_name=_find_text(m_ctx, "EPHEMERIS_NAME"),
            covariance_method=_find_text(m_ctx, "COVARIANCE_METHOD") or "CALCULATED",
            maneuverable=_find_text(m_ctx, "MANEUVERABLE") or "YES",
            orbit_center=_find_text(m_ctx, "ORBIT_CENTER") or "EARTH",
            ref_frame=_find_text(m_ctx, "REF_FRAME") or "EME2000",
            gravity_model=_find_text(m_ctx, "GRAVITY_MODEL"),
            atmospheric_model=_find_text(m_ctx, "ATMOSPHERIC_MODEL"),
            n_body_perturbations=_find_text(m_ctx, "N_BODY_PERTURBATIONS"),
            solar_rad_pressure=_find_text(m_ctx, "SOLAR_RAD_PRESSURE"),
            earth_tides=_find_text(m_ctx, "EARTH_TIDES"),
            intrack_thrust=_find_text(m_ctx, "INTRACK_THRUST"),
        )
        if meta_elem is not None:
            for c in meta_elem.findall(".//COMMENT"):
                if c.text:
                    obj_meta.comment.append(c.text.strip())

        # State vector
        sv_elem = None
        for child in d_ctx.iter():
            if _strip_ns(child.tag).upper() in ("STATEVECTOR", "STATE_VECTOR"):
                sv_elem = child
                break
        sv_ctx = sv_elem if sv_elem is not None else d_ctx

        sv = None
        if any(_find_float(sv_ctx, tag) is not None for tag in ("X", "Y", "Z")):
            sv = CDMStateVector(
                epoch=_find_datetime(sv_ctx, "EPOCH"),
                x=_find_float(sv_ctx, "X"),
                y=_find_float(sv_ctx, "Y"),
                z=_find_float(sv_ctx, "Z"),
                x_dot=_find_float(sv_ctx, "X_DOT"),
                y_dot=_find_float(sv_ctx, "Y_DOT"),
                z_dot=_find_float(sv_ctx, "Z_DOT"),
            )

        # Covariance Matrix
        cov_elem = None
        for child in d_ctx.iter():
            if _strip_ns(child.tag).upper() in ("COVARIANCEMATRIX", "COVARIANCE_MATRIX"):
                cov_elem = child
                break
        cov_ctx = cov_elem if cov_elem is not None else d_ctx

        cov = None
        if any(_find_float(cov_ctx, tag) is not None for tag in ("CR_R", "CT_T", "CN_N")):
            cov = CDMCovariance(
                ref_frame=_find_text(cov_ctx, "REF_FRAME") or "RTN",
                cr_r=_find_float(cov_ctx, "CR_R"),
                ct_r=_find_float(cov_ctx, "CT_R"),
                ct_t=_find_float(cov_ctx, "CT_T"),
                cn_r=_find_float(cov_ctx, "CN_R"),
                cn_t=_find_float(cov_ctx, "CN_T"),
                cn_n=_find_float(cov_ctx, "CN_N"),
                crdot_r=_find_float(cov_ctx, "CRDOT_R"),
                crdot_t=_find_float(cov_ctx, "CRDOT_T"),
                crdot_n=_find_float(cov_ctx, "CRDOT_N"),
                crdot_rdot=_find_float(cov_ctx, "CRDOT_RDOT"),
                ctdot_r=_find_float(cov_ctx, "CTDOT_R"),
                ctdot_t=_find_float(cov_ctx, "CTDOT_T"),
                ctdot_n=_find_float(cov_ctx, "CTDOT_N"),
                ctdot_rdot=_find_float(cov_ctx, "CTDOT_RDOT"),
                ctdot_tdot=_find_float(cov_ctx, "CTDOT_TDOT"),
                cndot_r=_find_float(cov_ctx, "CNDOT_R"),
                cndot_t=_find_float(cov_ctx, "CNDOT_T"),
                cndot_n=_find_float(cov_ctx, "CNDOT_N"),
                cndot_rdot=_find_float(cov_ctx, "CNDOT_RDOT"),
                cndot_tdot=_find_float(cov_ctx, "CNDOT_TDOT"),
                cndot_ndot=_find_float(cov_ctx, "CNDOT_NDOT"),
            )

        obj_data = CDMObjectData(
            time_lastob_start=_find_datetime(d_ctx, "TIME_LASTOB_START"),
            time_lastob_end=_find_datetime(d_ctx, "TIME_LASTOB_END"),
            recommended_od_span=_find_float(d_ctx, "RECOMMENDED_OD_SPAN"),
            actual_od_span=_find_float(d_ctx, "ACTUAL_OD_SPAN"),
            obs_available=_find_int(d_ctx, "OBS_AVAILABLE"),
            obs_used=_find_int(d_ctx, "OBS_USED"),
            tracks_available=_find_int(d_ctx, "TRACKS_AVAILABLE"),
            tracks_used=_find_int(d_ctx, "TRACKS_USED"),
            residuals_accepted=_find_float(d_ctx, "RESIDUALS_ACCEPTED"),
            weighted_rms=_find_float(d_ctx, "WEIGHTED_RMS"),
            spherical_coefficients=_find_float(d_ctx, "SPHERICAL_COEFFICIENTS"),
            cd_area_over_mass=_find_float(d_ctx, "CD_AREA_OVER_MASS"),
            cr_area_over_mass=_find_float(d_ctx, "CR_AREA_OVER_MASS"),
            thrust_acceleration=_find_float(d_ctx, "THRUST_ACCELERATION"),
            sedr=_find_float(d_ctx, "SEDR"),
            state_vector=sv,
            covariance=cov,
        )
        if data_elem is not None:
            for c in data_elem.findall(".//COMMENT"):
                if c.text:
                    obj_data.comment.append(c.text.strip())

        return CDMObject(metadata=obj_meta, data=obj_data)

    obj1 = _parse_segment(segments[0], "OBJECT1")
    obj2 = _parse_segment(segments[1], "OBJECT2")

    return CDMMessage(
        header=header,
        relative_metadata=relative_metadata,
        object1=obj1,
        object2=obj2,
    )


def export_cdm_xml(cdm: CDMMessage, pretty_print: bool = True) -> str:
    """Export CDMMessage into official CCSDS 508.0-B-1 XML representation."""
    root = ET.Element("cdm", {
        "id": "CCSDS_CDM_VERS",
        "version": cdm.header.ccsds_cdm_vers,
    })

    # Header
    header_el = ET.SubElement(root, "header")
    for comm in cdm.header.comment:
        ET.SubElement(header_el, "COMMENT").text = comm
    ET.SubElement(header_el, "CREATION_DATE").text = format_ccsds_timestamp(cdm.header.creation_date)
    ET.SubElement(header_el, "ORIGINATOR").text = cdm.header.originator
    if cdm.header.message_for:
        ET.SubElement(header_el, "MESSAGE_FOR").text = cdm.header.message_for
    ET.SubElement(header_el, "MESSAGE_ID").text = cdm.header.message_id

    # Body
    body_el = ET.SubElement(root, "body")

    # Relative Metadata
    rel_el = ET.SubElement(body_el, "relativeMetadataData")
    for comm in cdm.relative_metadata.comment:
        ET.SubElement(rel_el, "COMMENT").text = comm
    ET.SubElement(rel_el, "TCA").text = format_ccsds_timestamp(cdm.relative_metadata.tca)
    ET.SubElement(rel_el, "MISS_DISTANCE", {"units": "m"}).text = f"{cdm.relative_metadata.miss_distance:.3f}"
    if cdm.relative_metadata.relative_speed is not None:
        ET.SubElement(rel_el, "RELATIVE_SPEED", {"units": "m/s"}).text = f"{cdm.relative_metadata.relative_speed:.3f}"

    if (
        cdm.relative_metadata.relative_position_r is not None
        or cdm.relative_metadata.relative_velocity_r is not None
    ):
        rel_sv_el = ET.SubElement(rel_el, "relativeStateVector")
        if cdm.relative_metadata.relative_position_r is not None:
            ET.SubElement(rel_sv_el, "RELATIVE_POSITION_R", {"units": "m"}).text = f"{cdm.relative_metadata.relative_position_r:.3f}"
            ET.SubElement(rel_sv_el, "RELATIVE_POSITION_T", {"units": "m"}).text = f"{cdm.relative_metadata.relative_position_t:.3f}"
            ET.SubElement(rel_sv_el, "RELATIVE_POSITION_N", {"units": "m"}).text = f"{cdm.relative_metadata.relative_position_n:.3f}"
        if cdm.relative_metadata.relative_velocity_r is not None:
            ET.SubElement(rel_sv_el, "RELATIVE_VELOCITY_R", {"units": "m/s"}).text = f"{cdm.relative_metadata.relative_velocity_r:.3f}"
            ET.SubElement(rel_sv_el, "RELATIVE_VELOCITY_T", {"units": "m/s"}).text = f"{cdm.relative_metadata.relative_velocity_t:.3f}"
            ET.SubElement(rel_sv_el, "RELATIVE_VELOCITY_N", {"units": "m/s"}).text = f"{cdm.relative_metadata.relative_velocity_n:.3f}"

    if cdm.relative_metadata.start_screen_volume:
        ET.SubElement(rel_el, "START_SCREEN_VOLUME").text = format_ccsds_timestamp(cdm.relative_metadata.start_screen_volume)
    if cdm.relative_metadata.stop_screen_volume:
        ET.SubElement(rel_el, "STOP_SCREEN_VOLUME").text = format_ccsds_timestamp(cdm.relative_metadata.stop_screen_volume)
    if cdm.relative_metadata.screen_volume_frame:
        ET.SubElement(rel_el, "SCREEN_VOLUME_FRAME").text = cdm.relative_metadata.screen_volume_frame
    if cdm.relative_metadata.screen_volume_shape:
        ET.SubElement(rel_el, "SCREEN_VOLUME_SHAPE").text = cdm.relative_metadata.screen_volume_shape
    if cdm.relative_metadata.screen_volume_x is not None:
        ET.SubElement(rel_el, "SCREEN_VOLUME_X", {"units": "m"}).text = f"{cdm.relative_metadata.screen_volume_x:.3f}"
    if cdm.relative_metadata.screen_volume_y is not None:
        ET.SubElement(rel_el, "SCREEN_VOLUME_Y", {"units": "m"}).text = f"{cdm.relative_metadata.screen_volume_y:.3f}"
    if cdm.relative_metadata.screen_volume_z is not None:
        ET.SubElement(rel_el, "SCREEN_VOLUME_Z", {"units": "m"}).text = f"{cdm.relative_metadata.screen_volume_z:.3f}"
    if cdm.relative_metadata.screen_entry_time:
        ET.SubElement(rel_el, "SCREEN_ENTRY_TIME").text = format_ccsds_timestamp(cdm.relative_metadata.screen_entry_time)
    if cdm.relative_metadata.screen_exit_time:
        ET.SubElement(rel_el, "SCREEN_EXIT_TIME").text = format_ccsds_timestamp(cdm.relative_metadata.screen_exit_time)

    if cdm.relative_metadata.collision_probability is not None:
        ET.SubElement(rel_el, "COLLISION_PROBABILITY").text = f"{cdm.relative_metadata.collision_probability:.6e}"
    if cdm.relative_metadata.collision_probability_method:
        ET.SubElement(rel_el, "COLLISION_PROBABILITY_METHOD").text = cdm.relative_metadata.collision_probability_method

    # Helper for segments
    def _export_segment(obj: CDMObject):
        seg_el = ET.SubElement(body_el, "segment")
        meta_el = ET.SubElement(seg_el, "metadata")
        for comm in obj.metadata.comment:
            ET.SubElement(meta_el, "COMMENT").text = comm
        ET.SubElement(meta_el, "OBJECT").text = obj.metadata.object
        ET.SubElement(meta_el, "OBJECT_DESIGNATOR").text = obj.metadata.object_designator
        if obj.metadata.catalog_name:
            ET.SubElement(meta_el, "CATALOG_NAME").text = obj.metadata.catalog_name
        if obj.metadata.object_name:
            ET.SubElement(meta_el, "OBJECT_NAME").text = obj.metadata.object_name
        if obj.metadata.international_designator:
            ET.SubElement(meta_el, "INTERNATIONAL_DESIGNATOR").text = obj.metadata.international_designator
        if obj.metadata.object_type:
            ET.SubElement(meta_el, "OBJECT_TYPE").text = obj.metadata.object_type
        if obj.metadata.operator_contact_position:
            ET.SubElement(meta_el, "OPERATOR_CONTACT_POSITION").text = obj.metadata.operator_contact_position
        if obj.metadata.operator_organization:
            ET.SubElement(meta_el, "OPERATOR_ORGANIZATION").text = obj.metadata.operator_organization
        if obj.metadata.ephemeris_name:
            ET.SubElement(meta_el, "EPHEMERIS_NAME").text = obj.metadata.ephemeris_name
        if obj.metadata.covariance_method:
            ET.SubElement(meta_el, "COVARIANCE_METHOD").text = obj.metadata.covariance_method
        if obj.metadata.maneuverable:
            ET.SubElement(meta_el, "MANEUVERABLE").text = obj.metadata.maneuverable
        if obj.metadata.ref_frame:
            ET.SubElement(meta_el, "REF_FRAME").text = obj.metadata.ref_frame

        data_el = ET.SubElement(seg_el, "data")
        for comm in obj.data.comment:
            ET.SubElement(data_el, "COMMENT").text = comm

        # State Vector
        sv = obj.data.state_vector
        if sv is not None:
            sv_el = ET.SubElement(data_el, "stateVector")
            if sv.epoch:
                ET.SubElement(sv_el, "EPOCH").text = format_ccsds_timestamp(sv.epoch)
            if sv.x is not None:
                ET.SubElement(sv_el, "X", {"units": "km"}).text = f"{sv.x:.6f}"
                ET.SubElement(sv_el, "Y", {"units": "km"}).text = f"{sv.y:.6f}"
                ET.SubElement(sv_el, "Z", {"units": "km"}).text = f"{sv.z:.6f}"
            if sv.x_dot is not None:
                ET.SubElement(sv_el, "X_DOT", {"units": "km/s"}).text = f"{sv.x_dot:.6f}"
                ET.SubElement(sv_el, "Y_DOT", {"units": "km/s"}).text = f"{sv.y_dot:.6f}"
                ET.SubElement(sv_el, "Z_DOT", {"units": "km/s"}).text = f"{sv.z_dot:.6f}"

        # Covariance Matrix
        cov = obj.data.covariance
        if cov is not None:
            cov_el = ET.SubElement(data_el, "covarianceMatrix")
            ET.SubElement(cov_el, "REF_FRAME").text = cov.ref_frame
            for term, unit in [
                ("CR_R", "m**2"), ("CT_R", "m**2"), ("CT_T", "m**2"),
                ("CN_R", "m**2"), ("CN_T", "m**2"), ("CN_N", "m**2"),
                ("CRDOT_R", "m**2/s"), ("CRDOT_T", "m**2/s"), ("CRDOT_N", "m**2/s"), ("CRDOT_RDOT", "m**2/s**2"),
                ("CTDOT_R", "m**2/s"), ("CTDOT_T", "m**2/s"), ("CTDOT_N", "m**2/s"), ("CTDOT_RDOT", "m**2/s**2"), ("CTDOT_TDOT", "m**2/s**2"),
                ("CNDOT_R", "m**2/s"), ("CNDOT_T", "m**2/s"), ("CNDOT_N", "m**2/s"), ("CNDOT_RDOT", "m**2/s**2"), ("CNDOT_TDOT", "m**2/s**2"), ("CNDOT_NDOT", "m**2/s**2")
            ]:
                val = getattr(cov, term.lower(), None)
                if val is not None:
                    ET.SubElement(cov_el, term, {"units": unit}).text = f"{val:.6e}"

    _export_segment(cdm.object1)
    _export_segment(cdm.object2)

    if pretty_print:
        ET.indent(root, space="  ", level=0)

    raw_xml = ET.tostring(root, encoding="utf-8").decode("utf-8")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{raw_xml}'


# ---------------------------------------------------------------------------
# KVN (Keyword = Value Notation) Parsing & Exporting
# ---------------------------------------------------------------------------

def parse_cdm_kvn(kvn_text: str) -> CDMMessage:
    """
    Parse a CCSDS 508.0-B-1 CDM Keyword = Value Notation (KVN) text.
    Handles units in brackets e.g. "MISS_DISTANCE [m] = 120.5".
    """
    lines = kvn_text.splitlines()

    header_dict: Dict[str, Any] = {}
    rel_dict: Dict[str, Any] = {}
    obj_dicts: List[Dict[str, Any]] = []
    current_obj: Optional[Dict[str, Any]] = None

    header_keys = {"CCSDS_CDM_VERS", "CREATION_DATE", "ORIGINATOR", "MESSAGE_FOR", "MESSAGE_ID"}
    comments: List[str] = []

    for line in lines:
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue

        if cleaned.upper().startswith("COMMENT"):
            comm = cleaned[7:].strip()
            if comm.startswith("="):
                comm = comm[1:].strip()
            comments.append(comm)
            continue

        if "=" not in cleaned:
            continue

        key_part, val_part = cleaned.split("=", 1)
        # Strip unit in brackets, e.g. "MISS_DISTANCE [m]" -> "MISS_DISTANCE"
        key_raw = re.sub(r"\[.*?\]", "", key_part).strip().upper()
        val = val_part.strip()

        if key_raw == "OBJECT":
            # Start of a new object segment
            current_obj = {"OBJECT": val, "comments": list(comments)}
            comments.clear()
            obj_dicts.append(current_obj)
            continue

        if current_obj is not None:
            current_obj[key_raw] = val
        elif key_raw in header_keys:
            header_dict[key_raw] = val
        else:
            rel_dict[key_raw] = val

    if len(obj_dicts) < 2:
        raise ValueError(f"CCSDS CDM KVN requires at least 2 OBJECT blocks, found {len(obj_dicts)}")

    # Build Header
    header = CDMHeader(
        ccsds_cdm_vers=header_dict.get("CCSDS_CDM_VERS", "1.0"),
        creation_date=parse_ccsds_timestamp(header_dict["CREATION_DATE"]) if "CREATION_DATE" in header_dict else datetime.now(timezone.utc),
        originator=header_dict.get("ORIGINATOR", "UNKNOWN"),
        message_for=header_dict.get("MESSAGE_FOR"),
        message_id=header_dict.get("MESSAGE_ID", f"CDM-{uuid.uuid4().hex[:8].upper()}"),
    )

    # Build Relative Metadata
    tca_str = rel_dict.get("TCA")
    if not tca_str:
        raise ValueError("Missing mandatory relative metadata field: TCA")
    miss_str = rel_dict.get("MISS_DISTANCE")
    if not miss_str:
        raise ValueError("Missing mandatory relative metadata field: MISS_DISTANCE")

    def _get_float(d: Dict[str, Any], *k: str) -> Optional[float]:
        for key in k:
            if key in d:
                try:
                    return float(d[key])
                except ValueError:
                    pass
        return None

    def _get_int(d: Dict[str, Any], *k: str) -> Optional[int]:
        for key in k:
            if key in d:
                try:
                    return int(d[key])
                except ValueError:
                    pass
        return None

    def _get_dt(d: Dict[str, Any], *k: str) -> Optional[datetime]:
        for key in k:
            if key in d:
                try:
                    return parse_ccsds_timestamp(d[key])
                except ValueError:
                    pass
        return None

    relative_metadata = CDMRelativeMetadata(
        tca=parse_ccsds_timestamp(tca_str),
        miss_distance=float(miss_str),
        relative_speed=_get_float(rel_dict, "RELATIVE_SPEED"),
        relative_position_r=_get_float(rel_dict, "RELATIVE_POSITION_R"),
        relative_position_t=_get_float(rel_dict, "RELATIVE_POSITION_T"),
        relative_position_n=_get_float(rel_dict, "RELATIVE_POSITION_N"),
        relative_velocity_r=_get_float(rel_dict, "RELATIVE_VELOCITY_R"),
        relative_velocity_t=_get_float(rel_dict, "RELATIVE_VELOCITY_T"),
        relative_velocity_n=_get_float(rel_dict, "RELATIVE_VELOCITY_N"),
        start_screen_volume=_get_dt(rel_dict, "START_SCREEN_VOLUME"),
        stop_screen_volume=_get_dt(rel_dict, "STOP_SCREEN_VOLUME"),
        screen_volume_frame=rel_dict.get("SCREEN_VOLUME_FRAME", "RTN"),
        screen_volume_shape=rel_dict.get("SCREEN_VOLUME_SHAPE", "ELLIPSOID"),
        screen_volume_x=_get_float(rel_dict, "SCREEN_VOLUME_X", "SCREEN_VOLUME_RADIUS"),
        screen_volume_y=_get_float(rel_dict, "SCREEN_VOLUME_Y"),
        screen_volume_z=_get_float(rel_dict, "SCREEN_VOLUME_Z"),
        screen_volume_radius=_get_float(rel_dict, "SCREEN_VOLUME_RADIUS"),
        screen_entry_time=_get_dt(rel_dict, "SCREEN_ENTRY_TIME"),
        screen_exit_time=_get_dt(rel_dict, "SCREEN_EXIT_TIME"),
        collision_probability=_get_float(rel_dict, "COLLISION_PROBABILITY"),
        collision_probability_method=rel_dict.get("COLLISION_PROBABILITY_METHOD"),
        collision_percentile=_get_float(rel_dict, "COLLISION_PERCENTILE"),
        screening_threshold=_get_float(rel_dict, "SCREENING_THRESHOLD"),
    )

    def _build_obj(d: Dict[str, Any], default_name: str) -> CDMObject:
        meta = CDMObjectMetadata(
            object=d.get("OBJECT", default_name),
            object_designator=d.get("OBJECT_DESIGNATOR", ""),
            catalog_name=d.get("CATALOG_NAME", "SATCAT"),
            object_name=d.get("OBJECT_NAME"),
            international_designator=d.get("INTERNATIONAL_DESIGNATOR"),
            object_type=d.get("OBJECT_TYPE", "PAYLOAD"),
            operator_contact_position=d.get("OPERATOR_CONTACT_POSITION"),
            operator_organization=d.get("OPERATOR_ORGANIZATION"),
            operator_phone=d.get("OPERATOR_PHONE"),
            operator_email=d.get("OPERATOR_EMAIL"),
            ephemeris_name=d.get("EPHEMERIS_NAME"),
            covariance_method=d.get("COVARIANCE_METHOD", "CALCULATED"),
            maneuverable=d.get("MANEUVERABLE", "YES"),
            orbit_center=d.get("ORBIT_CENTER", "EARTH"),
            ref_frame=d.get("REF_FRAME", "EME2000"),
            gravity_model=d.get("GRAVITY_MODEL"),
            atmospheric_model=d.get("ATMOSPHERIC_MODEL"),
            n_body_perturbations=d.get("N_BODY_PERTURBATIONS"),
            solar_rad_pressure=d.get("SOLAR_RAD_PRESSURE"),
            earth_tides=d.get("EARTH_TIDES"),
            intrack_thrust=d.get("INTRACK_THRUST"),
        )

        sv = None
        if any(k in d for k in ("X", "Y", "Z")):
            sv = CDMStateVector(
                epoch=_get_dt(d, "EPOCH"),
                x=_get_float(d, "X"),
                y=_get_float(d, "Y"),
                z=_get_float(d, "Z"),
                x_dot=_get_float(d, "X_DOT"),
                y_dot=_get_float(d, "Y_DOT"),
                z_dot=_get_float(d, "Z_DOT"),
            )

        cov = None
        if any(k in d for k in ("CR_R", "CT_T", "CN_N")):
            cov = CDMCovariance(
                ref_frame=d.get("REF_FRAME", "RTN"),
                cr_r=_get_float(d, "CR_R"),
                ct_r=_get_float(d, "CT_R"),
                ct_t=_get_float(d, "CT_T"),
                cn_r=_get_float(d, "CN_R"),
                cn_t=_get_float(d, "CN_T"),
                cn_n=_get_float(d, "CN_N"),
                crdot_r=_get_float(d, "CRDOT_R"),
                crdot_t=_get_float(d, "CRDOT_T"),
                crdot_n=_get_float(d, "CRDOT_N"),
                crdot_rdot=_get_float(d, "CRDOT_RDOT"),
                ctdot_r=_get_float(d, "CTDOT_R"),
                ctdot_t=_get_float(d, "CTDOT_T"),
                ctdot_n=_get_float(d, "CTDOT_N"),
                ctdot_rdot=_get_float(d, "CTDOT_RDOT"),
                ctdot_tdot=_get_float(d, "CTDOT_TDOT"),
                cndot_r=_get_float(d, "CNDOT_R"),
                cndot_t=_get_float(d, "CNDOT_T"),
                cndot_n=_get_float(d, "CNDOT_N"),
                cndot_rdot=_get_float(d, "CNDOT_RDOT"),
                cndot_tdot=_get_float(d, "CNDOT_TDOT"),
                cndot_ndot=_get_float(d, "CNDOT_NDOT"),
            )

        data = CDMObjectData(
            time_lastob_start=_get_dt(d, "TIME_LASTOB_START"),
            time_lastob_end=_get_dt(d, "TIME_LASTOB_END"),
            recommended_od_span=_get_float(d, "RECOMMENDED_OD_SPAN"),
            actual_od_span=_get_float(d, "ACTUAL_OD_SPAN"),
            obs_available=_get_int(d, "OBS_AVAILABLE"),
            obs_used=_get_int(d, "OBS_USED"),
            tracks_available=_get_int(d, "TRACKS_AVAILABLE"),
            tracks_used=_get_int(d, "TRACKS_USED"),
            residuals_accepted=_get_float(d, "RESIDUALS_ACCEPTED"),
            weighted_rms=_get_float(d, "WEIGHTED_RMS"),
            spherical_coefficients=_get_float(d, "SPHERICAL_COEFFICIENTS"),
            cd_area_over_mass=_get_float(d, "CD_AREA_OVER_MASS"),
            cr_area_over_mass=_get_float(d, "CR_AREA_OVER_MASS"),
            thrust_acceleration=_get_float(d, "THRUST_ACCELERATION"),
            sedr=_get_float(d, "SEDR"),
            state_vector=sv,
            covariance=cov,
        )
        return CDMObject(metadata=meta, data=data)

    return CDMMessage(
        header=header,
        relative_metadata=relative_metadata,
        object1=_build_obj(obj_dicts[0], "OBJECT1"),
        object2=_build_obj(obj_dicts[1], "OBJECT2"),
    )


def export_cdm_kvn(cdm: CDMMessage) -> str:
    """Export CDMMessage into official CCSDS 508.0-B-1 KVN text."""
    lines: List[str] = [
        "CCSDS_CDM_VERS              = " + cdm.header.ccsds_cdm_vers,
    ]
    for c in cdm.header.comment:
        lines.append(f"COMMENT {c}")
    lines.extend([
        f"CREATION_DATE               = {format_ccsds_timestamp(cdm.header.creation_date)}",
        f"ORIGINATOR                  = {cdm.header.originator}",
    ])
    if cdm.header.message_for:
        lines.append(f"MESSAGE_FOR                 = {cdm.header.message_for}")
    lines.append(f"MESSAGE_ID                  = {cdm.header.message_id}")
    lines.append("")

    # Relative Metadata
    lines.append(f"TCA                         = {format_ccsds_timestamp(cdm.relative_metadata.tca)}")
    lines.append(f"MISS_DISTANCE [m]           = {cdm.relative_metadata.miss_distance:.3f}")
    if cdm.relative_metadata.relative_speed is not None:
        lines.append(f"RELATIVE_SPEED [m/s]        = {cdm.relative_metadata.relative_speed:.3f}")

    if cdm.relative_metadata.relative_position_r is not None:
        lines.append(f"RELATIVE_POSITION_R [m]     = {cdm.relative_metadata.relative_position_r:.3f}")
        lines.append(f"RELATIVE_POSITION_T [m]     = {cdm.relative_metadata.relative_position_t:.3f}")
        lines.append(f"RELATIVE_POSITION_N [m]     = {cdm.relative_metadata.relative_position_n:.3f}")

    if cdm.relative_metadata.relative_velocity_r is not None:
        lines.append(f"RELATIVE_VELOCITY_R [m/s]   = {cdm.relative_metadata.relative_velocity_r:.3f}")
        lines.append(f"RELATIVE_VELOCITY_T [m/s]   = {cdm.relative_metadata.relative_velocity_t:.3f}")
        lines.append(f"RELATIVE_VELOCITY_N [m/s]   = {cdm.relative_metadata.relative_velocity_n:.3f}")

    if cdm.relative_metadata.start_screen_volume:
        lines.append(f"START_SCREEN_VOLUME         = {format_ccsds_timestamp(cdm.relative_metadata.start_screen_volume)}")
    if cdm.relative_metadata.stop_screen_volume:
        lines.append(f"STOP_SCREEN_VOLUME          = {format_ccsds_timestamp(cdm.relative_metadata.stop_screen_volume)}")
    if cdm.relative_metadata.screen_volume_frame:
        lines.append(f"SCREEN_VOLUME_FRAME         = {cdm.relative_metadata.screen_volume_frame}")
    if cdm.relative_metadata.screen_volume_shape:
        lines.append(f"SCREEN_VOLUME_SHAPE         = {cdm.relative_metadata.screen_volume_shape}")
    if cdm.relative_metadata.screen_volume_x is not None:
        lines.append(f"SCREEN_VOLUME_X [m]         = {cdm.relative_metadata.screen_volume_x:.3f}")
    if cdm.relative_metadata.screen_volume_y is not None:
        lines.append(f"SCREEN_VOLUME_Y [m]         = {cdm.relative_metadata.screen_volume_y:.3f}")
    if cdm.relative_metadata.screen_volume_z is not None:
        lines.append(f"SCREEN_VOLUME_Z [m]         = {cdm.relative_metadata.screen_volume_z:.3f}")
    if cdm.relative_metadata.screen_entry_time:
        lines.append(f"SCREEN_ENTRY_TIME           = {format_ccsds_timestamp(cdm.relative_metadata.screen_entry_time)}")
    if cdm.relative_metadata.screen_exit_time:
        lines.append(f"SCREEN_EXIT_TIME            = {format_ccsds_timestamp(cdm.relative_metadata.screen_exit_time)}")

    if cdm.relative_metadata.collision_probability is not None:
        lines.append(f"COLLISION_PROBABILITY       = {cdm.relative_metadata.collision_probability:.6e}")
    if cdm.relative_metadata.collision_probability_method:
        lines.append(f"COLLISION_PROBABILITY_METHOD= {cdm.relative_metadata.collision_probability_method}")
    lines.append("")

    # Helper for objects
    def _append_obj(obj: CDMObject):
        lines.append(f"OBJECT                      = {obj.metadata.object}")
        for c in obj.metadata.comment:
            lines.append(f"COMMENT {c}")
        lines.append(f"OBJECT_DESIGNATOR           = {obj.metadata.object_designator}")
        if obj.metadata.catalog_name:
            lines.append(f"CATALOG_NAME                = {obj.metadata.catalog_name}")
        if obj.metadata.object_name:
            lines.append(f"OBJECT_NAME                 = {obj.metadata.object_name}")
        if obj.metadata.international_designator:
            lines.append(f"INTERNATIONAL_DESIGNATOR    = {obj.metadata.international_designator}")
        if obj.metadata.object_type:
            lines.append(f"OBJECT_TYPE                 = {obj.metadata.object_type}")
        if obj.metadata.operator_contact_position:
            lines.append(f"OPERATOR_CONTACT_POSITION   = {obj.metadata.operator_contact_position}")
        if obj.metadata.operator_organization:
            lines.append(f"OPERATOR_ORGANIZATION       = {obj.metadata.operator_organization}")
        if obj.metadata.ephemeris_name:
            lines.append(f"EPHEMERIS_NAME              = {obj.metadata.ephemeris_name}")
        if obj.metadata.covariance_method:
            lines.append(f"COVARIANCE_METHOD           = {obj.metadata.covariance_method}")
        if obj.metadata.maneuverable:
            lines.append(f"MANEUVERABLE                = {obj.metadata.maneuverable}")
        if obj.metadata.ref_frame:
            lines.append(f"REF_FRAME                   = {obj.metadata.ref_frame}")

        # State Vector
        sv = obj.data.state_vector
        if sv is not None:
            if sv.epoch:
                lines.append(f"EPOCH                       = {format_ccsds_timestamp(sv.epoch)}")
            if sv.x is not None:
                lines.append(f"X [km]                      = {sv.x:14.6f}")
                lines.append(f"Y [km]                      = {sv.y:14.6f}")
                lines.append(f"Z [km]                      = {sv.z:14.6f}")
            if sv.x_dot is not None:
                lines.append(f"X_DOT [km/s]                = {sv.x_dot:14.6f}")
                lines.append(f"Y_DOT [km/s]                = {sv.y_dot:14.6f}")
                lines.append(f"Z_DOT [km/s]                = {sv.z_dot:14.6f}")

        # Covariance Matrix
        cov = obj.data.covariance
        if cov is not None:
            lines.append(f"COV_REF_FRAME               = {cov.ref_frame}")
            terms = [
                ("CR_R [m**2]", cov.cr_r),
                ("CT_R [m**2]", cov.ct_r),
                ("CT_T [m**2]", cov.ct_t),
                ("CN_R [m**2]", cov.cn_r),
                ("CN_T [m**2]", cov.cn_t),
                ("CN_N [m**2]", cov.cn_n),
                ("CRDOT_R [m**2/s]", cov.crdot_r),
                ("CRDOT_T [m**2/s]", cov.crdot_t),
                ("CRDOT_N [m**2/s]", cov.crdot_n),
                ("CRDOT_RDOT [m**2/s**2]", cov.crdot_rdot),
                ("CTDOT_R [m**2/s]", cov.ctdot_r),
                ("CTDOT_T [m**2/s]", cov.ctdot_t),
                ("CTDOT_N [m**2/s]", cov.ctdot_n),
                ("CTDOT_RDOT [m**2/s**2]", cov.ctdot_rdot),
                ("CTDOT_TDOT [m**2/s**2]", cov.ctdot_tdot),
                ("CNDOT_R [m**2/s]", cov.cndot_r),
                ("CNDOT_T [m**2/s]", cov.cndot_t),
                ("CNDOT_N [m**2/s]", cov.cndot_n),
                ("CNDOT_RDOT [m**2/s**2]", cov.cndot_rdot),
                ("CNDOT_TDOT [m**2/s**2]", cov.cndot_tdot),
                ("CNDOT_NDOT [m**2/s**2]", cov.cndot_ndot),
            ]
            for label, val in terms:
                if val is not None:
                    lines.append(f"{label:<28}= {val:16.6e}")
        lines.append("")

    _append_obj(cdm.object1)
    _append_obj(cdm.object2)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# General Auto-detect Parser & Validator
# ---------------------------------------------------------------------------

def parse_cdm(content: str) -> CDMMessage:
    """
    Parse CDM content automatically detecting XML or KVN format.
    """
    stripped = content.strip()
    if stripped.startswith("<"):
        return parse_cdm_xml(stripped)
    return parse_cdm_kvn(stripped)


def validate_cdm_xml(xml_content: str) -> ValidationResult:
    """Validate CDM XML string against CCSDS 508.0-B-1 rules."""
    try:
        cdm = parse_cdm_xml(xml_content)
        return cdm.validate()
    except Exception as e:
        return ValidationResult(is_valid=False, errors=[str(e)])


def validate_cdm_kvn(kvn_content: str) -> ValidationResult:
    """Validate CDM KVN string against CCSDS 508.0-B-1 rules."""
    try:
        cdm = parse_cdm_kvn(kvn_content)
        return cdm.validate()
    except Exception as e:
        return ValidationResult(is_valid=False, errors=[str(e)])


# ---------------------------------------------------------------------------
# Builder Helper
# ---------------------------------------------------------------------------

def create_cdm_message(
    primary_sat_id: str,
    primary_sat_name: str,
    secondary_obj_id: str,
    secondary_obj_name: str,
    miss_distance_m: float,
    relative_speed_m_s: float,
    collision_prob: float,
    tca: datetime,
    primary_state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    secondary_state: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    primary_cov_rtn_m2: Optional[Union[np.ndarray, Dict[str, float]]] = None,
    secondary_cov_rtn_m2: Optional[Union[np.ndarray, Dict[str, float]]] = None,
    screening_volume_m: Optional[Tuple[float, float, float]] = (5000.0, 25000.0, 5000.0),
    originator: str = "AEGIS-LEO SSA SHIELD",
) -> CDMMessage:
    """
    Convenience factory to construct a fully populated CDMMessage.
    """
    header = CDMHeader(
        ccsds_cdm_vers="1.0",
        creation_date=datetime.now(timezone.utc),
        originator=originator,
        message_id=f"CDM-AEGIS-{uuid.uuid4().hex[:8].upper()}",
    )

    # Screening volume setup
    sx, sy, sz = screening_volume_m if screening_volume_m else (None, None, None)

    # Relative position and velocity estimation
    rel_pos_r = rel_pos_t = rel_pos_n = None
    rel_vel_r = rel_vel_t = rel_vel_n = None

    if primary_state is not None and secondary_state is not None:
        r1_km, v1_km_s = primary_state
        r2_km, v2_km_s = secondary_state
        dr_m = (r2_km - r1_km) * 1000.0
        dv_m_s = (v2_km_s - v1_km_s) * 1000.0
        # In RTN frame of primary:
        r_unit = r1_km / np.linalg.norm(r1_km)
        h_vec = np.cross(r1_km, v1_km_s)
        n_unit = h_vec / np.linalg.norm(h_vec)
        t_unit = np.cross(n_unit, r_unit)

        rel_pos_r = float(np.dot(dr_m, r_unit))
        rel_pos_t = float(np.dot(dr_m, t_unit))
        rel_pos_n = float(np.dot(dr_m, n_unit))

        rel_vel_r = float(np.dot(dv_m_s, r_unit))
        rel_vel_t = float(np.dot(dv_m_s, t_unit))
        rel_vel_n = float(np.dot(dv_m_s, n_unit))

    rel_meta = CDMRelativeMetadata(
        tca=tca,
        miss_distance=float(miss_distance_m),
        relative_speed=float(relative_speed_m_s),
        relative_position_r=rel_pos_r,
        relative_position_t=rel_pos_t,
        relative_position_n=rel_pos_n,
        relative_velocity_r=rel_vel_r,
        relative_velocity_t=rel_vel_t,
        relative_velocity_n=rel_vel_n,
        start_screen_volume=tca - timedelta(minutes=15),
        stop_screen_volume=tca + timedelta(minutes=15),
        screen_volume_frame="RTN",
        screen_volume_shape="ELLIPSOID",
        screen_volume_x=sx,
        screen_volume_y=sy,
        screen_volume_z=sz,
        collision_probability=collision_prob,
        collision_probability_method="FOSTER-1992",
    )

    def _make_cov(cov_input: Optional[Union[np.ndarray, Dict[str, float]]]) -> Optional[CDMCovariance]:
        if cov_input is None:
            return None
        if isinstance(cov_input, np.ndarray):
            return CDMCovariance.from_matrix(cov_input)
        if isinstance(cov_input, dict):
            return CDMCovariance(
                ref_frame="RTN",
                cr_r=cov_input.get("c_rr", cov_input.get("cr_r")),
                ct_r=cov_input.get("c_rt", cov_input.get("ct_r", 0.0)),
                ct_t=cov_input.get("c_tt", cov_input.get("ct_t")),
                cn_r=cov_input.get("c_rn", cov_input.get("cn_r", 0.0)),
                cn_t=cov_input.get("c_tn", cov_input.get("cn_t", 0.0)),
                cn_n=cov_input.get("c_nn", cov_input.get("cn_n")),
            )
        return None

    def _make_sv(state: Optional[Tuple[np.ndarray, np.ndarray]]) -> Optional[CDMStateVector]:
        if state is None:
            return None
        r, v = state
        return CDMStateVector(
            epoch=tca,
            x=float(r[0]), y=float(r[1]), z=float(r[2]),
            x_dot=float(v[0]), y_dot=float(v[1]), z_dot=float(v[2]),
        )

    obj1 = CDMObject(
        metadata=CDMObjectMetadata(
            object="OBJECT1",
            object_designator=primary_sat_id,
            object_name=primary_sat_name,
            object_type="PAYLOAD",
            maneuverable="YES",
        ),
        data=CDMObjectData(
            state_vector=_make_sv(primary_state),
            covariance=_make_cov(primary_cov_rtn_m2),
        ),
    )

    obj2 = CDMObject(
        metadata=CDMObjectMetadata(
            object="OBJECT2",
            object_designator=secondary_obj_id,
            object_name=secondary_obj_name,
            object_type="DEBRIS",
            maneuverable="NO",
        ),
        data=CDMObjectData(
            state_vector=_make_sv(secondary_state),
            covariance=_make_cov(secondary_cov_rtn_m2),
        ),
    )

    return CDMMessage(
        header=header,
        relative_metadata=rel_meta,
        object1=obj1,
        object2=obj2,
    )
