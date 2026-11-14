"""
AEGIS-LEO: Constellation Slot Stationkeeping & Walker-Delta Phasing Engine
Provides high-fidelity Delta-V budgeting for operational LEO constellations and
intra-plane drift phasing optimization for Walker-Delta slot allocation.

Adheres strictly to physics models, standard astrodynamics formulations
(Vis-Viva, Gauss perturbation, Tsiolkovsky rocket equation, Kuhn-Munkres assignment),
and zero external dependencies beyond NumPy and the Python standard library.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
import math
import numpy as np

from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    EARTH_RADIUS_M,
    EARTH_MU,
    EARTH_MU_SI,
    STANDARD_GRAVITY,
    QUIET_KP,
    QUIET_F10_7,
    MODERATE_KP,
    MODERATE_F10_7,
)
from aegis_leo.physics.atmosphere import compute_atmospheric_density


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class StationkeepingBudget:
    """Detailed annual stationkeeping and disposal Delta-v / propellant budget."""
    satellite_id: str
    altitude_km: float
    inclination_deg: float
    wet_mass_kg: float
    remaining_fuel_kg: float
    isp_seconds: float
    ballistic_coefficient_inv: float  # Cd * A / m (m^2 / kg)

    # Delta-v breakdown (m/s per year)
    drag_delta_v_per_year_m_s: float
    inclination_delta_v_per_year_m_s: float
    eccentricity_delta_v_per_year_m_s: float
    slot_deadband_delta_v_per_year_m_s: float
    collision_avoidance_delta_v_per_year_m_s: float
    subtotal_annual_operational_delta_v_m_s: float

    # Margin and total
    margin_percent: float
    total_annual_operational_delta_v_with_margin_m_s: float

    # End-of-life disposal (one-off)
    deorbit_disposal_delta_v_m_s: float

    # Propellant consumption & mission lifetime
    annual_propellant_consumption_kg: float
    disposal_propellant_kg: float
    fuel_depletion_lifetime_years: float  # Total fuel / annual consumption
    operational_lifetime_with_disposal_years: float  # (Total fuel - disposal) / annual consumption

    # In-plane stationkeeping cycle metrics
    daily_altitude_decay_m: float
    reboost_cycle_days: float
    reboosts_per_year: float
    delta_v_per_reboost_m_s: float
    max_along_track_deadband_km: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert budget to serializable dictionary."""
        return asdict(self)


@dataclass
class PhasingManeuverResult:
    """Astrodynamic maneuver plan for intra-plane slot relocation."""
    maneuver_type: str  # "CIRCULAR_DRIFT_4_BURN" or "RESONANT_ELLIPTIC_2_BURN"
    initial_anomaly_deg: float
    target_anomaly_deg: float
    phase_shift_deg: float  # Signed angular shift required
    delta_altitude_km: float  # Radial separation of drift orbit from nominal
    drift_semi_major_axis_km: float
    drift_rate_deg_per_day: float
    duration_hours: float
    duration_days: float
    total_delta_v_m_s: float
    burn_breakdown_m_s: List[float]
    propellant_consumed_kg: float
    num_revolutions: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert maneuver plan to serializable dictionary."""
        res = asdict(self)
        res["burn_breakdown_m_s"] = [round(b, 4) for b in self.burn_breakdown_m_s]
        return res


@dataclass
class SlotReassignmentResult:
    """Optimal assignment of constellation satellites to intra-plane orbital slots."""
    plane: int
    assignments: List[Dict[str, Any]]
    total_fleet_delta_v_m_s: float
    max_satellite_delta_v_m_s: float
    average_satellite_delta_v_m_s: float
    total_propellant_consumed_kg: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert assignment result to serializable dictionary."""
        return asdict(self)


# ==============================================================================
# OPTIMAL ASSIGNMENT SOLVER (KUHN-MUNKRES / HUNGARIAN ALGORITHM)
# ==============================================================================

def hungarian_algorithm(cost_matrix: Union[np.ndarray, List[List[float]]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the linear sum assignment problem using the Kuhn-Munkres (Hungarian) algorithm.
    Finds row_ind and col_ind that minimize sum(cost_matrix[row_ind, col_ind]).

    Parameters:
        cost_matrix: 2D array of shape (n_rows, n_cols).

    Returns:
        (row_ind, col_ind) arrays of matched pairs.
    """
    cost = np.array(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError("Cost matrix must be 2-dimensional.")

    n_rows, n_cols = cost.shape
    transposed = False
    if n_rows > n_cols:
        cost = cost.T
        n_rows, n_cols = cost.shape
        transposed = True

    u = np.zeros(n_rows + 1, dtype=float)
    v = np.zeros(n_cols + 1, dtype=float)
    p = np.zeros(n_cols + 1, dtype=int)
    way = np.zeros(n_cols + 1, dtype=int)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, np.inf, dtype=float)
        used = np.zeros(n_cols + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0
            for j in range(1, n_cols + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j

            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    row_ind = np.zeros(n_rows, dtype=int)
    col_ind = np.zeros(n_rows, dtype=int)
    for j in range(1, n_cols + 1):
        if p[j] != 0:
            row_ind[p[j] - 1] = p[j] - 1
            col_ind[p[j] - 1] = j - 1

    if transposed:
        return col_ind, row_ind
    return row_ind, col_ind


# ==============================================================================
# STATIONKEEPING DELTA-V BUDGETING ENGINE
# ==============================================================================

def compute_drag_decay(
    altitude_km: float,
    area_m2: float,
    mass_kg: float,
    cd: float = 2.2,
    kp: float = QUIET_KP,
    f10_7: float = QUIET_F10_7,
    density_kg_m3: Optional[float] = None,
) -> Dict[str, float]:
    """
    Calculate atmospheric drag deceleration, semi-major axis daily decay rate,
    and annual tangential velocity loss.

    Physics:
        v = sqrt(mu / r)
        a_drag = 0.5 * rho * v^2 * (Cd * A / m)
        da/dt = - rho * v * (Cd * A / m) * r
        Delta_v_annual = a_drag * (365.25 * 86400)
    """
    r_m = (EARTH_RADIUS_KM + altitude_km) * 1000.0
    v_m_s = math.sqrt(EARTH_MU_SI / r_m)

    if density_kg_m3 is None:
        rho = compute_atmospheric_density(altitude_km=altitude_km, kp=kp, f10_7=f10_7)
    else:
        rho = float(density_kg_m3)

    ballistic_inv = (cd * area_m2) / max(1e-3, mass_kg)  # m^2 / kg
    a_drag_m_s2 = 0.5 * rho * (v_m_s ** 2) * ballistic_inv

    # Daily decay in meters: da/dt = - rho * v * B * r
    da_dt_m_s = rho * v_m_s * ballistic_inv * r_m
    daily_decay_m = da_dt_m_s * 86400.0

    # Annual tangential velocity loss to overcome
    annual_drag_dv_m_s = a_drag_m_s2 * (365.25 * 86400.0)

    return {
        "density_kg_m3": rho,
        "velocity_m_s": v_m_s,
        "drag_accel_m_s2": a_drag_m_s2,
        "daily_decay_m": daily_decay_m,
        "annual_drag_delta_v_m_s": annual_drag_dv_m_s,
    }


def compute_slot_deadband_reboost(
    altitude_km: float,
    daily_decay_m: float,
    altitude_deadband_km: float = 2.0,
) -> Dict[str, float]:
    """
    Compute in-plane slot deadband maintenance cycle parameters.
    Models the classic parabolic drift cycle where drag decays the orbit across
    an allowable radial deadband [-delta_a, +delta_a] followed by a two-impulse
    or tangential re-boost burn back to the upper altitude deadband limit.

    Parameters:
        altitude_km: Nominal orbital altitude (km).
        daily_decay_m: Natural orbital decay rate (meters / day).
        altitude_deadband_km: Half-width of allowed semi-major axis deadband (+/- km).
    """
    r_m = (EARTH_RADIUS_KM + altitude_km) * 1000.0
    v_m_s = math.sqrt(EARTH_MU_SI / r_m)
    n_rad_s = math.sqrt(EARTH_MU_SI / (r_m ** 3))

    total_window_m = 2.0 * altitude_deadband_km * 1000.0
    safe_decay_rate = max(1e-6, daily_decay_m)

    cycle_days = total_window_m / safe_decay_rate
    reboosts_per_year = 365.25 / cycle_days

    # Delta-v per re-boost burn to restore 2 * delta_a
    dv_reboost_m_s = 0.5 * v_m_s * (total_window_m / r_m)

    # Parabolic along-track excursion: delta_lambda = 3/8 * (n / r) * |da/dt| * T_cycle^2
    da_dt_m_s = safe_decay_rate / 86400.0
    t_cycle_sec = cycle_days * 86400.0
    delta_lambda_rad = (3.0 / 8.0) * (n_rad_s / r_m) * da_dt_m_s * (t_cycle_sec ** 2)
    along_track_km = delta_lambda_rad * (r_m / 1000.0)

    return {
        "cycle_days": cycle_days,
        "reboosts_per_year": reboosts_per_year,
        "delta_v_per_reboost_m_s": dv_reboost_m_s,
        "annual_slot_delta_v_m_s": dv_reboost_m_s * reboosts_per_year,
        "max_along_track_deadband_km": along_track_km,
    }


def compute_inclination_stationkeeping_dv(
    altitude_km: float,
    annual_inclination_drift_deg: float = 0.035,
) -> float:
    """
    Calculate cross-track Delta-v (m/s / year) required to cancel out-of-plane
    secular inclination drift caused by Sun/Moon third-body gravitational
    perturbations and solar radiation pressure.

    Formula:
        Delta_v_inc = 2 * v * sin(Delta_i / 2)
    """
    r_m = (EARTH_RADIUS_KM + altitude_km) * 1000.0
    v_m_s = math.sqrt(EARTH_MU_SI / r_m)
    di_rad = math.radians(annual_inclination_drift_deg)
    return float(2.0 * v_m_s * math.sin(di_rad / 2.0))


def compute_eccentricity_stationkeeping_dv(
    altitude_km: float,
    annual_eccentricity_drift: float = 0.0006,
) -> float:
    """
    Calculate in-plane Delta-v (m/s / year) required to control eccentricity vector
    and maintain a circular / frozen orbit against Earth J3 harmonic and SRP.

    Approximation:
        Delta_v_ecc ~= v * Delta_e
    """
    r_m = (EARTH_RADIUS_KM + altitude_km) * 1000.0
    v_m_s = math.sqrt(EARTH_MU_SI / r_m)
    return float(v_m_s * annual_eccentricity_drift)


def compute_deorbit_dv(
    initial_altitude_km: float,
    target_perigee_km: float = 65.0,
) -> float:
    """
    Calculate retrograde Delta-v (m/s) required for end-of-life targeted disposal
    into dense atmosphere per NASA-STD-8719.14B / FCC compliance.
    """
    r1 = (EARTH_RADIUS_KM + initial_altitude_km) * 1000.0
    r2 = (EARTH_RADIUS_KM + target_perigee_km) * 1000.0
    a_trans = (r1 + r2) / 2.0

    v1 = math.sqrt(EARTH_MU_SI / r1)
    v_trans_a = math.sqrt(EARTH_MU_SI * (2.0 / r1 - 1.0 / a_trans))
    return float(v1 - v_trans_a)


def compute_stationkeeping_budget(
    satellite_id: str = "SAT-01",
    altitude_km: float = 550.0,
    inclination_deg: float = 53.0,
    mass_kg: float = 260.0,
    area_m2: float = 2.2,
    cd: float = 2.2,
    remaining_fuel_kg: float = 15.0,
    isp_seconds: float = 250.0,
    kp: float = QUIET_KP,
    f10_7: float = QUIET_F10_7,
    density_kg_m3: Optional[float] = None,
    altitude_deadband_km: float = 2.0,
    annual_inclination_drift_deg: float = 0.035,
    annual_eccentricity_drift: float = 0.0006,
    annual_cam_allocation_m_s: float = 1.5,
    margin_percent: float = 15.0,
    deorbit_target_perigee_km: float = 65.0,
) -> StationkeepingBudget:
    """
    Compute comprehensive annual stationkeeping and disposal Delta-v / propellant budget
    for a satellite in Low Earth Orbit.
    """
    # 1. Drag decay & tangential re-boost
    drag_res = compute_drag_decay(
        altitude_km=altitude_km,
        area_m2=area_m2,
        mass_kg=mass_kg,
        cd=cd,
        kp=kp,
        f10_7=f10_7,
        density_kg_m3=density_kg_m3,
    )
    daily_decay_m = drag_res["daily_decay_m"]
    annual_drag_dv = drag_res["annual_drag_delta_v_m_s"]

    # 2. In-plane slot deadband cycle
    slot_res = compute_slot_deadband_reboost(
        altitude_km=altitude_km,
        daily_decay_m=daily_decay_m,
        altitude_deadband_km=altitude_deadband_km,
    )

    # 3. Cross-track inclination & eccentricity maintenance
    annual_inc_dv = compute_inclination_stationkeeping_dv(
        altitude_km=altitude_km,
        annual_inclination_drift_deg=annual_inclination_drift_deg,
    )
    annual_ecc_dv = compute_eccentricity_stationkeeping_dv(
        altitude_km=altitude_km,
        annual_eccentricity_drift=annual_eccentricity_drift,
    )

    # 4. Operational subtotal & margin
    # Drag compensation is the primary driver of slot maintenance
    subtotal_annual_dv = (
        annual_drag_dv
        + annual_inc_dv
        + annual_ecc_dv
        + annual_cam_allocation_m_s
    )
    margin_factor = 1.0 + (margin_percent / 100.0)
    total_annual_dv_margin = subtotal_annual_dv * margin_factor

    # 5. End-of-life disposal burn
    deorbit_dv = compute_deorbit_dv(
        initial_altitude_km=altitude_km,
        target_perigee_km=deorbit_target_perigee_km,
    )

    # Propellant consumption via Tsiolkovsky equation
    ve = isp_seconds * STANDARD_GRAVITY
    annual_fuel_kg = mass_kg * (1.0 - math.exp(-total_annual_dv_margin / ve))
    disposal_fuel_kg = mass_kg * (1.0 - math.exp(-deorbit_dv / ve))

    # Total depletion lifetime vs operational lifetime with disposal fuel reserved
    if annual_fuel_kg > 1e-6:
        fuel_depletion_lifetime = remaining_fuel_kg / annual_fuel_kg
        usable_fuel_kg = max(0.0, remaining_fuel_kg - disposal_fuel_kg)
        operational_lifetime_with_disposal = usable_fuel_kg / annual_fuel_kg
    else:
        fuel_depletion_lifetime = 999.0
        operational_lifetime_with_disposal = 999.0

    b_inv = (cd * area_m2) / max(1e-3, mass_kg)

    return StationkeepingBudget(
        satellite_id=satellite_id,
        altitude_km=round(float(altitude_km), 2),
        inclination_deg=round(float(inclination_deg), 2),
        wet_mass_kg=round(float(mass_kg), 2),
        remaining_fuel_kg=round(float(remaining_fuel_kg), 3),
        isp_seconds=round(float(isp_seconds), 1),
        ballistic_coefficient_inv=round(float(b_inv), 5),
        drag_delta_v_per_year_m_s=round(float(annual_drag_dv), 3),
        inclination_delta_v_per_year_m_s=round(float(annual_inc_dv), 3),
        eccentricity_delta_v_per_year_m_s=round(float(annual_ecc_dv), 3),
        slot_deadband_delta_v_per_year_m_s=round(float(slot_res["annual_slot_delta_v_m_s"]), 3),
        collision_avoidance_delta_v_per_year_m_s=round(float(annual_cam_allocation_m_s), 3),
        subtotal_annual_operational_delta_v_m_s=round(float(subtotal_annual_dv), 3),
        margin_percent=round(float(margin_percent), 1),
        total_annual_operational_delta_v_with_margin_m_s=round(float(total_annual_dv_margin), 3),
        deorbit_disposal_delta_v_m_s=round(float(deorbit_dv), 3),
        annual_propellant_consumption_kg=round(float(annual_fuel_kg), 4),
        disposal_propellant_kg=round(float(disposal_fuel_kg), 4),
        fuel_depletion_lifetime_years=round(float(fuel_depletion_lifetime), 2),
        operational_lifetime_with_disposal_years=round(float(operational_lifetime_with_disposal), 2),
        daily_altitude_decay_m=round(float(daily_decay_m), 4),
        reboost_cycle_days=round(float(slot_res["cycle_days"]), 2),
        reboosts_per_year=round(float(slot_res["reboosts_per_year"]), 1),
        delta_v_per_reboost_m_s=round(float(slot_res["delta_v_per_reboost_m_s"]), 4),
        max_along_track_deadband_km=round(float(slot_res["max_along_track_deadband_km"]), 3),
    )


def budget_constellation_fleet(
    fleet: List[Dict[str, Any]],
    mission_lifetime_years: float = 5.0,
    kp: float = QUIET_KP,
    f10_7: float = QUIET_F10_7,
    altitude_deadband_km: float = 2.0,
    margin_percent: float = 15.0,
    isp_seconds: float = 250.0,
) -> Dict[str, Any]:
    """
    Perform fleet-wide stationkeeping and propellant analysis across a constellation
    (such as generated by generate_walker_delta_constellation).
    """
    satellite_budgets: List[StationkeepingBudget] = []
    total_fleet_annual_dv = 0.0
    total_fleet_annual_fuel = 0.0
    total_fleet_disposal_fuel = 0.0
    critical_satellites: List[Dict[str, Any]] = []

    for sat in fleet:
        sat_id = sat.get("satellite_id", "SAT")
        alt = float(sat.get("altitude_km", 550.0))
        inc = float(sat.get("inclination_deg", 53.0))
        mass = float(sat.get("mass_kg", 260.0))
        area = float(sat.get("area_m2", 2.2))
        fuel = float(sat.get("remaining_fuel_kg", 15.0))
        isp = float(sat.get("isp_seconds", isp_seconds))

        budget = compute_stationkeeping_budget(
            satellite_id=sat_id,
            altitude_km=alt,
            inclination_deg=inc,
            mass_kg=mass,
            area_m2=area,
            remaining_fuel_kg=fuel,
            isp_seconds=isp,
            kp=kp,
            f10_7=f10_7,
            altitude_deadband_km=altitude_deadband_km,
            margin_percent=margin_percent,
        )
        satellite_budgets.append(budget)

        total_fleet_annual_dv += budget.total_annual_operational_delta_v_with_margin_m_s
        total_fleet_annual_fuel += budget.annual_propellant_consumption_kg
        total_fleet_disposal_fuel += budget.disposal_propellant_kg

        # Check if satellite cannot achieve designated mission lifetime
        required_fuel = (
            budget.annual_propellant_consumption_kg * mission_lifetime_years
            + budget.disposal_propellant_kg
        )
        if fuel < required_fuel:
            critical_satellites.append({
                "satellite_id": sat_id,
                "plane": sat.get("plane", 1),
                "remaining_fuel_kg": fuel,
                "required_fuel_kg": round(required_fuel, 3),
                "projected_lifetime_years": budget.operational_lifetime_with_disposal_years,
                "deficit_kg": round(required_fuel - fuel, 3),
            })

    fleet_size = len(fleet)
    avg_annual_dv = total_fleet_annual_dv / max(1, fleet_size)

    return {
        "fleet_size": fleet_size,
        "mission_lifetime_years": mission_lifetime_years,
        "fleet_total_annual_delta_v_m_s": round(total_fleet_annual_dv, 2),
        "fleet_average_annual_delta_v_m_s": round(avg_annual_dv, 2),
        "fleet_total_annual_fuel_kg": round(total_fleet_annual_fuel, 3),
        "fleet_total_disposal_fuel_kg": round(total_fleet_disposal_fuel, 3),
        "fleet_lifetime_fuel_demand_kg": round(
            total_fleet_annual_fuel * mission_lifetime_years + total_fleet_disposal_fuel, 3
        ),
        "critical_satellite_count": len(critical_satellites),
        "critical_satellites": critical_satellites,
        "budgets": [b.to_dict() for b in satellite_budgets],
    }


# ==============================================================================
# WALKER-DELTA INTRA-PLANE DRIFT PHASING OPTIMIZER
# ==============================================================================

class WalkerPhasingOptimizer:
    """
    Intra-plane drift phasing optimizer for Walker-Delta constellations (T/P/F).

    Formulations:
        1. Circular Drift Phasing (4-Burn Architecture):
           - Injects into circular drift orbit h_drift = h_0 + Delta_h
           - Computes Hohmann entry and exit transfers
           - Accumulates differential drift Delta_phi = (n_drift - n_0) * t_coast
           - Recircularizes at target slot
        2. Resonant Elliptic Phasing (2-Burn Architecture):
           - Single-impulse insertion into resonant eccentric orbit
           - Completes k revolutions while accumulating exact phase drift
           - Single-impulse recircularization at target slot
        3. Kuhn-Munkres Constellation Slot Reassignment:
           - Global minimum Delta-V assignment for multi-satellite relocation.
    """

    def __init__(
        self,
        altitude_km: float = 550.0,
        total_satellites: int = 24,
        planes: int = 4,
        relative_phase_f: int = 1,
        satellite_mass_kg: float = 260.0,
        isp_seconds: float = 250.0,
        max_delta_altitude_km: float = 35.0,
    ):
        self.altitude_km = float(altitude_km)
        self.total_satellites = int(total_satellites)
        self.planes = int(planes)
        self.f_factor = int(relative_phase_f)
        self.mass_kg = float(satellite_mass_kg)
        self.isp_seconds = float(isp_seconds)
        self.max_delta_altitude_km = float(abs(max_delta_altitude_km))

        # Nominal geometry
        self.sats_per_plane = self.total_satellites // self.planes
        self.slot_spacing_deg = 360.0 / self.sats_per_plane
        self.inter_plane_spacing_deg = (self.f_factor * 360.0) / self.total_satellites

        # Nominal orbit parameters
        self.r0_m = (EARTH_RADIUS_KM + self.altitude_km) * 1000.0
        self.v0_m_s = math.sqrt(EARTH_MU_SI / self.r0_m)
        self.n0_rad_s = math.sqrt(EARTH_MU_SI / (self.r0_m ** 3))
        self.orbital_period_s = 2.0 * math.pi / self.n0_rad_s

    def get_slot_nominal_anomaly(self, plane_idx: int, slot_idx: int) -> float:
        """
        Get the nominal true anomaly (degrees) of slot s (0-indexed) in plane p (1-indexed).
        """
        p = (plane_idx - 1) % self.planes
        s = slot_idx % self.sats_per_plane
        nu = (s * self.slot_spacing_deg + p * self.inter_plane_spacing_deg) % 360.0
        return float(round(nu, 4))

    def compute_phase_error(
        self,
        current_anomaly_deg: float,
        target_slot_anomaly_deg: float,
        allow_retrograde: bool = True,
    ) -> float:
        """
        Compute required phase displacement Delta_phi in degrees.
        If allow_retrograde is True, maps Delta_phi into (-180, +180] for shortest angular path.
        Positive Delta_phi means satellite must advance forward along track (drift lower).
        Negative Delta_phi means satellite must slow down / lag behind (drift higher).
        """
        diff = (target_slot_anomaly_deg - current_anomaly_deg) % 360.0
        if allow_retrograde and diff > 180.0:
            diff -= 360.0
        return float(diff)

    def plan_circular_drift_phasing(
        self,
        current_anomaly_deg: float,
        target_anomaly_deg: float,
        delta_altitude_km: Optional[float] = None,
        target_duration_hours: Optional[float] = None,
    ) -> PhasingManeuverResult:
        """
        Plan a 4-burn circular drift phasing maneuver.
        Either delta_altitude_km or target_duration_hours must be specified.
        """
        delta_phi_deg = self.compute_phase_error(current_anomaly_deg, target_anomaly_deg)
        if abs(delta_phi_deg) < 1e-4:
            return PhasingManeuverResult(
                maneuver_type="CIRCULAR_DRIFT_4_BURN",
                initial_anomaly_deg=round(current_anomaly_deg, 3),
                target_anomaly_deg=round(target_anomaly_deg, 3),
                phase_shift_deg=0.0,
                delta_altitude_km=0.0,
                drift_semi_major_axis_km=round(self.altitude_km + EARTH_RADIUS_KM, 3),
                drift_rate_deg_per_day=0.0,
                duration_hours=0.0,
                duration_days=0.0,
                total_delta_v_m_s=0.0,
                burn_breakdown_m_s=[0.0, 0.0, 0.0, 0.0],
                propellant_consumed_kg=0.0,
                num_revolutions=0.0,
            )

        delta_phi_rad = math.radians(delta_phi_deg)

        # If duration is provided, invert for delta_altitude using Newton-Raphson
        if delta_altitude_km is None:
            if target_duration_hours is None or target_duration_hours <= 0:
                raise ValueError("Either delta_altitude_km or target_duration_hours > 0 must be provided.")
            target_t_sec = target_duration_hours * 3600.0

            # Initial linear guess: delta_phi = -1.5 * n0 * (delta_h / r0) * target_t
            dh_m = -(2.0 / 3.0) * (self.r0_m * delta_phi_rad) / (self.n0_rad_s * target_t_sec)

            # Bound within max_delta_altitude_km
            max_dh_m = self.max_delta_altitude_km * 1000.0
            dh_m = max(-max_dh_m, min(max_dh_m, dh_m))

            for _ in range(25):
                rd_m = self.r0_m + dh_m
                nd = math.sqrt(EARTH_MU_SI / (rd_m ** 3))
                a_tr = (self.r0_m + rd_m) / 2.0
                t_tr = math.pi * math.sqrt((a_tr ** 3) / EARTH_MU_SI)
                dphi_tr = 2.0 * (math.pi - self.n0_rad_s * t_tr)

                drift_rate = nd - self.n0_rad_s
                dphi_coast = delta_phi_rad - dphi_tr
                t_coast = dphi_coast / drift_rate
                f = 2.0 * t_tr + t_coast - target_t_sec

                if abs(f) < 1e-4:
                    break

                eps = 1.0  # 1 meter finite difference
                rd_eps = self.r0_m + (dh_m + eps)
                nd_eps = math.sqrt(EARTH_MU_SI / (rd_eps ** 3))
                a_tr_eps = (self.r0_m + rd_eps) / 2.0
                t_tr_eps = math.pi * math.sqrt((a_tr_eps ** 3) / EARTH_MU_SI)
                dphi_tr_eps = 2.0 * (math.pi - self.n0_rad_s * t_tr_eps)
                drift_rate_eps = nd_eps - self.n0_rad_s
                dphi_coast_eps = delta_phi_rad - dphi_tr_eps
                t_coast_eps = dphi_coast_eps / drift_rate_eps
                f_eps = 2.0 * t_tr_eps + t_coast_eps - target_t_sec
                df = (f_eps - f) / eps
                if abs(df) < 1e-12:
                    break
                step = f / df
                dh_m -= step
                dh_m = max(-max_dh_m, min(max_dh_m, dh_m))
        else:
            dh_m = float(delta_altitude_km) * 1000.0
            # Ensure proper direction
            if delta_phi_rad > 0 and dh_m > 0:
                dh_m = -abs(dh_m)
            elif delta_phi_rad < 0 and dh_m < 0:
                dh_m = abs(dh_m)

        rd_m = self.r0_m + dh_m
        vd_m_s = math.sqrt(EARTH_MU_SI / rd_m)
        nd_rad_s = math.sqrt(EARTH_MU_SI / (rd_m ** 3))

        # Hohmann transfer into drift orbit
        a_trans = (self.r0_m + rd_m) / 2.0
        t_trans = math.pi * math.sqrt((a_trans ** 3) / EARTH_MU_SI)

        # Burn 1 & Burn 2
        vt_r0 = math.sqrt(EARTH_MU_SI * (2.0 / self.r0_m - 1.0 / a_trans))
        dv1 = abs(vt_r0 - self.v0_m_s)
        vt_rd = math.sqrt(EARTH_MU_SI * (2.0 / rd_m - 1.0 / a_trans))
        dv2 = abs(vd_m_s - vt_rd)

        # Phase accumulated during transfer legs (entry + exit)
        dphi_transfers = 2.0 * (math.pi - self.n0_rad_s * t_trans)
        dphi_coast = delta_phi_rad - dphi_transfers

        drift_rate_rad_s = nd_rad_s - self.n0_rad_s
        t_coast = max(0.0, dphi_coast / drift_rate_rad_s)
        total_time_s = 2.0 * t_trans + t_coast

        total_dv = 2.0 * (dv1 + dv2)
        burn_breakdown = [dv1, dv2, dv2, dv1]

        # Fuel consumption
        ve = self.isp_seconds * STANDARD_GRAVITY
        fuel_kg = self.mass_kg * (1.0 - math.exp(-total_dv / ve))

        num_revs = total_time_s / self.orbital_period_s
        drift_rate_deg_day = math.degrees(drift_rate_rad_s) * 86400.0

        return PhasingManeuverResult(
            maneuver_type="CIRCULAR_DRIFT_4_BURN",
            initial_anomaly_deg=round(current_anomaly_deg, 3),
            target_anomaly_deg=round(target_anomaly_deg, 3),
            phase_shift_deg=round(delta_phi_deg, 3),
            delta_altitude_km=round(dh_m / 1000.0, 3),
            drift_semi_major_axis_km=round((rd_m) / 1000.0, 3),
            drift_rate_deg_per_day=round(drift_rate_deg_day, 3),
            duration_hours=round(total_time_s / 3600.0, 2),
            duration_days=round(total_time_s / 86400.0, 3),
            total_delta_v_m_s=round(total_dv, 3),
            burn_breakdown_m_s=burn_breakdown,
            propellant_consumed_kg=round(fuel_kg, 4),
            num_revolutions=round(num_revs, 2),
        )

    def plan_resonant_elliptic_phasing(
        self,
        current_anomaly_deg: float,
        target_anomaly_deg: float,
        num_revolutions: int = 10,
    ) -> PhasingManeuverResult:
        """
        Plan a 2-burn resonant eccentric phasing maneuver.
        The satellite burns into an elliptical orbit with perigee/apogee at r0,
        completes k revolutions while accumulating exact relative phase shift,
        and recircularizes back at r0 into the target slot.
        """
        delta_phi_deg = self.compute_phase_error(current_anomaly_deg, target_anomaly_deg)
        if abs(delta_phi_deg) < 1e-4:
            return PhasingManeuverResult(
                maneuver_type="RESONANT_ELLIPTIC_2_BURN",
                initial_anomaly_deg=round(current_anomaly_deg, 3),
                target_anomaly_deg=round(target_anomaly_deg, 3),
                phase_shift_deg=0.0,
                delta_altitude_km=0.0,
                drift_semi_major_axis_km=round(self.altitude_km + EARTH_RADIUS_KM, 3),
                drift_rate_deg_per_day=0.0,
                duration_hours=0.0,
                duration_days=0.0,
                total_delta_v_m_s=0.0,
                burn_breakdown_m_s=[0.0, 0.0],
                propellant_consumed_kg=0.0,
                num_revolutions=0.0,
            )

        k = max(1, int(num_revolutions))
        delta_phi_rad = math.radians(delta_phi_deg)

        # Phasing condition: k * (2 * pi - n0 * T_res) = delta_phi_rad
        # T_res = (2 * pi - delta_phi_rad / k) / n0
        t_res = (2.0 * math.pi - (delta_phi_rad / k)) / self.n0_rad_s
        a_res = (EARTH_MU_SI * ((t_res / (2.0 * math.pi)) ** 2)) ** (1.0 / 3.0)

        # Velocity at r0 on the resonant ellipse
        v_res = math.sqrt(EARTH_MU_SI * (2.0 / self.r0_m - 1.0 / a_res))
        dv_burn = abs(v_res - self.v0_m_s)
        total_dv = 2.0 * dv_burn

        total_time_s = k * t_res
        dh_m = a_res - self.r0_m

        ve = self.isp_seconds * STANDARD_GRAVITY
        fuel_kg = self.mass_kg * (1.0 - math.exp(-total_dv / ve))

        drift_rate_deg_day = (delta_phi_deg / (total_time_s / 86400.0))

        return PhasingManeuverResult(
            maneuver_type="RESONANT_ELLIPTIC_2_BURN",
            initial_anomaly_deg=round(current_anomaly_deg, 3),
            target_anomaly_deg=round(target_anomaly_deg, 3),
            phase_shift_deg=round(delta_phi_deg, 3),
            delta_altitude_km=round(dh_m / 1000.0, 3),
            drift_semi_major_axis_km=round(a_res / 1000.0, 3),
            drift_rate_deg_per_day=round(drift_rate_deg_day, 3),
            duration_hours=round(total_time_s / 3600.0, 2),
            duration_days=round(total_time_s / 86400.0, 3),
            total_delta_v_m_s=round(total_dv, 3),
            burn_breakdown_m_s=[dv_burn, dv_burn],
            propellant_consumed_kg=round(fuel_kg, 4),
            num_revolutions=float(k),
        )

    def optimize_phasing_time_fixed(
        self,
        current_anomaly_deg: float,
        target_anomaly_deg: float,
        max_duration_days: float = 5.0,
    ) -> PhasingManeuverResult:
        """
        Find the Delta-v optimal phasing plan that completes the relocation
        within max_duration_days while respecting maximum altitude deviation.
        """
        target_hours = max_duration_days * 24.0
        return self.plan_circular_drift_phasing(
            current_anomaly_deg=current_anomaly_deg,
            target_anomaly_deg=target_anomaly_deg,
            target_duration_hours=target_hours,
        )

    def optimize_phasing_delta_v_fixed(
        self,
        current_anomaly_deg: float,
        target_anomaly_deg: float,
        max_delta_v_m_s: float = 20.0,
    ) -> PhasingManeuverResult:
        """
        Find the fastest phasing trajectory given a maximum allowable Delta-v budget.
        """
        # Linear approximation: total_dv ~= 2 * v0 * (|dh| / r0)
        # |dh| ~= (max_delta_v_m_s * r0) / (2 * v0)
        dh_est_km = (max_delta_v_m_s * self.r0_m) / (2.0 * self.v0_m_s * 1000.0)
        dh_capped_km = min(dh_est_km, self.max_delta_altitude_km)

        return self.plan_circular_drift_phasing(
            current_anomaly_deg=current_anomaly_deg,
            target_anomaly_deg=target_anomaly_deg,
            delta_altitude_km=dh_capped_km,
        )

    def generate_pareto_front(
        self,
        current_anomaly_deg: float,
        target_anomaly_deg: float,
        min_duration_days: float = 1.0,
        max_duration_days: float = 14.0,
        steps: int = 25,
    ) -> List[Dict[str, float]]:
        """
        Generate Pareto trade-off curve between transfer duration and Delta-v consumption.
        """
        durations = np.linspace(min_duration_days, max_duration_days, steps)
        pareto_points: List[Dict[str, float]] = []

        for d_days in durations:
            try:
                res = self.plan_circular_drift_phasing(
                    current_anomaly_deg=current_anomaly_deg,
                    target_anomaly_deg=target_anomaly_deg,
                    target_duration_hours=float(d_days * 24.0),
                )
                pareto_points.append({
                    "duration_days": res.duration_days,
                    "duration_hours": res.duration_hours,
                    "delta_altitude_km": res.delta_altitude_km,
                    "total_delta_v_m_s": res.total_delta_v_m_s,
                    "propellant_consumed_kg": res.propellant_consumed_kg,
                })
            except Exception:
                continue

        return pareto_points

    def optimize_intra_plane_reassignment(
        self,
        satellites: List[Dict[str, Any]],
        available_slots: Optional[List[float]] = None,
        max_duration_days: float = 7.0,
    ) -> SlotReassignmentResult:
        """
        Solve global optimal assignment of N satellites in a single orbital plane
        to M target slots using the Kuhn-Munkres (Hungarian) algorithm.
        Minimizes total fleet Delta-v.
        """
        n_sats = len(satellites)
        if n_sats == 0:
            raise ValueError("Satellite list cannot be empty.")

        plane_idx = satellites[0].get("plane", 1)

        # Default to uniform Walker-Delta intra-plane slots if not provided
        if available_slots is None:
            available_slots = [
                self.get_slot_nominal_anomaly(plane_idx, s)
                for s in range(self.sats_per_plane)
            ]

        m_slots = len(available_slots)

        # Build Delta-V cost matrix C[i, j]
        cost_matrix = np.zeros((n_sats, m_slots), dtype=float)
        plan_cache: Dict[Tuple[int, int], PhasingManeuverResult] = {}

        for i, sat in enumerate(satellites):
            curr_anom = float(sat.get("true_anomaly_deg", 0.0))
            for j, slot_anom in enumerate(available_slots):
                plan = self.optimize_phasing_time_fixed(
                    current_anomaly_deg=curr_anom,
                    target_anomaly_deg=slot_anom,
                    max_duration_days=max_duration_days,
                )
                cost_matrix[i, j] = plan.total_delta_v_m_s
                plan_cache[(i, j)] = plan

        row_ind, col_ind = hungarian_algorithm(cost_matrix)

        assignments: List[Dict[str, Any]] = []
        total_fleet_dv = 0.0
        total_fuel = 0.0
        max_dv = 0.0

        for r, c in zip(row_ind, col_ind):
            sat = satellites[r]
            sat_id = sat.get("satellite_id", f"SAT-{r+1:02d}")
            plan = plan_cache[(r, c)]
            target_slot = available_slots[c]

            total_fleet_dv += plan.total_delta_v_m_s
            total_fuel += plan.propellant_consumed_kg
            if plan.total_delta_v_m_s > max_dv:
                max_dv = plan.total_delta_v_m_s

            assignments.append({
                "satellite_id": sat_id,
                "current_anomaly_deg": float(sat.get("true_anomaly_deg", 0.0)),
                "assigned_slot_idx": int(c),
                "target_slot_anomaly_deg": round(float(target_slot), 3),
                "phase_shift_deg": plan.phase_shift_deg,
                "delta_altitude_km": plan.delta_altitude_km,
                "duration_days": plan.duration_days,
                "delta_v_m_s": plan.total_delta_v_m_s,
                "propellant_consumed_kg": plan.propellant_consumed_kg,
            })

        avg_dv = total_fleet_dv / max(1, len(assignments))

        return SlotReassignmentResult(
            plane=plane_idx,
            assignments=assignments,
            total_fleet_delta_v_m_s=round(total_fleet_dv, 3),
            max_satellite_delta_v_m_s=round(max_dv, 3),
            average_satellite_delta_v_m_s=round(avg_dv, 3),
            total_propellant_consumed_kg=round(total_fuel, 4),
        )

    def rebalance_plane_after_satellite_loss(
        self,
        remaining_satellites: List[Dict[str, Any]],
        plane_idx: int = 1,
        max_duration_days: float = 10.0,
    ) -> SlotReassignmentResult:
        """
        Rebalance remaining satellites in a plane following a satellite loss or decommissioning.
        Redistributes S-1 satellites uniformly across 360 degrees to close coverage gaps
        with minimal total fleet Delta-v.
        """
        n_remaining = len(remaining_satellites)
        if n_remaining <= 1:
            raise ValueError("Need at least 2 remaining satellites to rebalance a plane.")

        # New uniform slot spacing: 360 / (S - 1)
        new_spacing = 360.0 / n_remaining
        p_offset = (plane_idx - 1) * self.inter_plane_spacing_deg

        new_slots = [
            (s * new_spacing + p_offset) % 360.0
            for s in range(n_remaining)
        ]

        return self.optimize_intra_plane_reassignment(
            satellites=remaining_satellites,
            available_slots=new_slots,
            max_duration_days=max_duration_days,
        )
