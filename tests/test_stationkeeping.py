"""
Unit & Integration Tests for Constellation Slot Stationkeeping & Walker Phasing Optimizer
"""

import pytest
import numpy as np
from aegis_leo.constants import (
    QUIET_KP,
    QUIET_F10_7,
    STORM_G4_KP,
    ACTIVE_F10_7,
)
from aegis_leo.screening.constellation_screen import generate_walker_delta_constellation
from aegis_leo.screening.stationkeeping import (
    hungarian_algorithm,
    compute_drag_decay,
    compute_slot_deadband_reboost,
    compute_inclination_stationkeeping_dv,
    compute_eccentricity_stationkeeping_dv,
    compute_deorbit_dv,
    compute_stationkeeping_budget,
    budget_constellation_fleet,
    WalkerPhasingOptimizer,
    StationkeepingBudget,
    PhasingManeuverResult,
    SlotReassignmentResult,
)


def test_hungarian_algorithm_basic():
    """Verify Kuhn-Munkres algorithm produces exact minimum weight assignment."""
    cost = np.array([
        [4, 1, 3],
        [2, 0, 5],
        [3, 2, 2]
    ])
    r_ind, c_ind = hungarian_algorithm(cost)
    total_cost = cost[r_ind, c_ind].sum()
    assert total_cost == 5  # (0,1)->1, (1,0)->2, (2,2)->2 => 5


def test_hungarian_algorithm_rectangular():
    """Verify rectangular assignment matrices (more columns than rows and vice versa)."""
    # 2 satellites, 4 slots
    cost_wide = np.array([
        [10.0, 2.0, 8.0, 5.0],
        [3.0, 9.0, 4.0, 7.0],
    ])
    r_ind, c_ind = hungarian_algorithm(cost_wide)
    assert len(r_ind) == 2
    assert cost_wide[r_ind, c_ind].sum() == (2.0 + 3.0)

    # 4 satellites, 2 slots (transposed matching)
    cost_tall = cost_wide.T
    r_tall, c_tall = hungarian_algorithm(cost_tall)
    assert len(r_tall) == 2
    assert cost_tall[r_tall, c_tall].sum() == (2.0 + 3.0)


def test_drag_decay_computation():
    """Verify atmospheric drag decay metrics under quiet and storm conditions."""
    quiet = compute_drag_decay(
        altitude_km=550.0,
        area_m2=2.2,
        mass_kg=260.0,
        cd=2.2,
        kp=QUIET_KP,
        f10_7=QUIET_F10_7,
    )
    assert quiet["velocity_m_s"] > 7500.0
    assert quiet["drag_accel_m_s2"] > 0.0
    assert quiet["daily_decay_m"] > 0.0
    assert quiet["annual_drag_delta_v_m_s"] > 0.0

    # Solar storm increases density and drag decay dramatically
    storm = compute_drag_decay(
        altitude_km=550.0,
        area_m2=2.2,
        mass_kg=260.0,
        cd=2.2,
        kp=STORM_G4_KP,
        f10_7=ACTIVE_F10_7,
    )
    assert storm["density_kg_m3"] > quiet["density_kg_m3"]
    assert storm["daily_decay_m"] > quiet["daily_decay_m"]
    assert storm["annual_drag_delta_v_m_s"] > quiet["annual_drag_delta_v_m_s"]


def test_slot_deadband_reboost():
    """Verify along-track slot deadband reboost calculations."""
    daily_decay_m = 10.0  # 10 meters per day decay
    res = compute_slot_deadband_reboost(
        altitude_km=550.0,
        daily_decay_m=daily_decay_m,
        altitude_deadband_km=2.0,
    )
    # Total window = 4000 m. Cycle = 4000 / 10 = 400 days
    assert np.isclose(res["cycle_days"], 400.0)
    assert res["reboosts_per_year"] < 1.0
    assert res["delta_v_per_reboost_m_s"] > 0.0
    assert res["max_along_track_deadband_km"] > 0.0


def test_stationkeeping_budget_synthesis():
    """Verify single-satellite comprehensive stationkeeping budget synthesis."""
    budget = compute_stationkeeping_budget(
        satellite_id="AEGIS-SAT-01",
        altitude_km=550.0,
        inclination_deg=53.0,
        mass_kg=260.0,
        area_m2=2.2,
        remaining_fuel_kg=15.0,
        isp_seconds=250.0,
        margin_percent=15.0,
    )
    assert isinstance(budget, StationkeepingBudget)
    assert budget.satellite_id == "AEGIS-SAT-01"
    assert budget.drag_delta_v_per_year_m_s > 0.0
    assert budget.inclination_delta_v_per_year_m_s > 0.0
    assert budget.eccentricity_delta_v_per_year_m_s > 0.0
    assert budget.collision_avoidance_delta_v_per_year_m_s == 1.5
    assert budget.deorbit_disposal_delta_v_m_s > 100.0  # Typical deorbit dv ~140 m/s
    assert budget.total_annual_operational_delta_v_with_margin_m_s > budget.subtotal_annual_operational_delta_v_m_s
    assert budget.fuel_depletion_lifetime_years > 5.0
    assert budget.annual_propellant_consumption_kg > 0.0

    b_dict = budget.to_dict()
    assert "satellite_id" in b_dict
    assert "total_annual_operational_delta_v_with_margin_m_s" in b_dict


def test_fleet_stationkeeping_budgeting():
    """Verify constellation-wide stationkeeping budget aggregation."""
    fleet = generate_walker_delta_constellation(total_satellites=12, planes=3)
    fleet_res = budget_constellation_fleet(
        fleet=fleet,
        mission_lifetime_years=5.0,
        altitude_deadband_km=2.0,
        margin_percent=15.0,
    )
    assert fleet_res["fleet_size"] == 12
    assert fleet_res["fleet_total_annual_delta_v_m_s"] > 0.0
    assert fleet_res["fleet_total_annual_fuel_kg"] > 0.0
    assert fleet_res["fleet_total_disposal_fuel_kg"] > 0.0
    assert len(fleet_res["budgets"]) == 12


def test_walker_phasing_optimizer_geometry():
    """Verify Walker-Delta slot angle geometry calculations."""
    opt = WalkerPhasingOptimizer(
        altitude_km=550.0,
        total_satellites=24,
        planes=4,
        relative_phase_f=1,
    )
    assert opt.sats_per_plane == 6
    assert opt.slot_spacing_deg == 60.0
    assert opt.inter_plane_spacing_deg == 15.0

    # Plane 1 slots: 0, 60, 120, 180, 240, 300
    s0 = opt.get_slot_nominal_anomaly(plane_idx=1, slot_idx=0)
    s1 = opt.get_slot_nominal_anomaly(plane_idx=1, slot_idx=1)
    s2 = opt.get_slot_nominal_anomaly(plane_idx=1, slot_idx=2)
    assert s0 == 0.0
    assert s1 == 60.0
    assert s2 == 120.0

    # Phase errors
    diff_forward = opt.compute_phase_error(0.0, 60.0)
    assert diff_forward == 60.0
    diff_backward = opt.compute_phase_error(60.0, 0.0)
    assert diff_backward == -60.0


def test_circular_drift_phasing_forward_and_backward():
    """Verify 4-burn circular drift phasing in both directions."""
    opt = WalkerPhasingOptimizer(altitude_km=550.0)

    # 1. Forward drift (+45 deg): lower orbit
    plan_fwd = opt.plan_circular_drift_phasing(
        current_anomaly_deg=0.0,
        target_anomaly_deg=45.0,
        delta_altitude_km=15.0,
    )
    assert plan_fwd.phase_shift_deg == 45.0
    assert plan_fwd.delta_altitude_km < 0.0  # Lowered orbit to speed up
    assert plan_fwd.drift_rate_deg_per_day > 0.0
    assert plan_fwd.total_delta_v_m_s > 0.0
    assert len(plan_fwd.burn_breakdown_m_s) == 4

    # 2. Backward drift (-45 deg): raise orbit
    plan_bwd = opt.plan_circular_drift_phasing(
        current_anomaly_deg=45.0,
        target_anomaly_deg=0.0,
        delta_altitude_km=15.0,
    )
    assert plan_bwd.phase_shift_deg == -45.0
    assert plan_bwd.delta_altitude_km > 0.0  # Raised orbit to slow down
    assert plan_bwd.drift_rate_deg_per_day < 0.0
    assert plan_bwd.total_delta_v_m_s > 0.0

    # 3. Zero phase shift
    plan_zero = opt.plan_circular_drift_phasing(0.0, 0.0)
    assert plan_zero.total_delta_v_m_s == 0.0
    assert plan_zero.duration_hours == 0.0


def test_resonant_elliptic_phasing():
    """Verify 2-burn resonant eccentric phasing maneuver."""
    opt = WalkerPhasingOptimizer(altitude_km=550.0)
    plan = opt.plan_resonant_elliptic_phasing(
        current_anomaly_deg=0.0,
        target_anomaly_deg=60.0,
        num_revolutions=12,
    )
    assert isinstance(plan, PhasingManeuverResult)
    assert plan.maneuver_type == "RESONANT_ELLIPTIC_2_BURN"
    assert plan.num_revolutions == 12
    assert plan.total_delta_v_m_s > 0.0
    assert len(plan.burn_breakdown_m_s) == 2
    assert np.isclose(plan.burn_breakdown_m_s[0], plan.burn_breakdown_m_s[1])


def test_phasing_time_fixed_and_pareto():
    """Verify fixed-time optimization and Pareto front generation."""
    opt = WalkerPhasingOptimizer(altitude_km=550.0)

    # 3-day relocation
    plan_3d = opt.optimize_phasing_time_fixed(
        current_anomaly_deg=0.0,
        target_anomaly_deg=60.0,
        max_duration_days=3.0,
    )
    assert np.isclose(plan_3d.duration_days, 3.0, atol=0.1)
    assert plan_3d.total_delta_v_m_s > 0.0

    # 6-day relocation (should require less delta-v than 3-day)
    plan_6d = opt.optimize_phasing_time_fixed(
        current_anomaly_deg=0.0,
        target_anomaly_deg=60.0,
        max_duration_days=6.0,
    )
    assert plan_6d.total_delta_v_m_s < plan_3d.total_delta_v_m_s

    # Pareto trade-off curve
    pareto = opt.generate_pareto_front(
        current_anomaly_deg=0.0,
        target_anomaly_deg=60.0,
        min_duration_days=2.0,
        max_duration_days=8.0,
        steps=5,
    )
    assert len(pareto) == 5
    # As duration increases, delta-v decreases monotonically
    for i in range(len(pareto) - 1):
        assert pareto[i]["duration_days"] < pareto[i + 1]["duration_days"]
        assert pareto[i]["total_delta_v_m_s"] >= pareto[i + 1]["total_delta_v_m_s"]


def test_intra_plane_slot_reassignment():
    """Verify Kuhn-Munkres multi-satellite slot assignment optimization."""
    opt = WalkerPhasingOptimizer(
        altitude_km=550.0,
        total_satellites=12,
        planes=3,  # 4 satellites per plane
    )

    # Satellites currently displaced slightly from nominal slots (e.g. at 2 deg, 88 deg, 185 deg, 275 deg)
    sats_in_plane = [
        {"satellite_id": "SAT-01", "plane": 1, "true_anomaly_deg": 2.0},
        {"satellite_id": "SAT-02", "plane": 1, "true_anomaly_deg": 88.0},
        {"satellite_id": "SAT-03", "plane": 1, "true_anomaly_deg": 185.0},
        {"satellite_id": "SAT-04", "plane": 1, "true_anomaly_deg": 275.0},
    ]

    res = opt.optimize_intra_plane_reassignment(
        satellites=sats_in_plane,
        max_duration_days=5.0,
    )
    assert isinstance(res, SlotReassignmentResult)
    assert len(res.assignments) == 4
    assert res.total_fleet_delta_v_m_s > 0.0
    assert res.average_satellite_delta_v_m_s > 0.0

    # Check each slot is uniquely assigned
    assigned_slots = set(a["assigned_slot_idx"] for a in res.assignments)
    assert len(assigned_slots) == 4


def test_plane_rebalance_after_satellite_loss():
    """Verify rebalancing a plane from S satellites down to S-1 satellites."""
    opt = WalkerPhasingOptimizer(
        altitude_km=550.0,
        total_satellites=12,
        planes=3,
    )
    # Suppose one satellite failed in Plane 1, leaving 3 satellites at ~0, 90, 180 deg
    remaining_sats = [
        {"satellite_id": "SAT-01", "plane": 1, "true_anomaly_deg": 0.0},
        {"satellite_id": "SAT-02", "plane": 1, "true_anomaly_deg": 90.0},
        {"satellite_id": "SAT-03", "plane": 1, "true_anomaly_deg": 180.0},
    ]

    res = opt.rebalance_plane_after_satellite_loss(
        remaining_satellites=remaining_sats,
        plane_idx=1,
        max_duration_days=7.0,
    )
    assert isinstance(res, SlotReassignmentResult)
    assert len(res.assignments) == 3
    # New slots should be spaced by 360 / 3 = 120 deg
    target_anoms = sorted([a["target_slot_anomaly_deg"] for a in res.assignments])
    spacing1 = (target_anoms[1] - target_anoms[0]) % 360.0
    spacing2 = (target_anoms[2] - target_anoms[1]) % 360.0
    assert np.isclose(spacing1, 120.0)
    assert np.isclose(spacing2, 120.0)
