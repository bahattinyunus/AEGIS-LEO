import pytest
from aegis_leo.ai.pareto_optimizer import ParetoManeuverOptimizer


def test_pareto_optimizer_generation():
    optimizer = ParetoManeuverOptimizer(mass_kg=260.0, fuel_remaining_kg=15.0, isp_s=250.0)
    res = optimizer.generate_pareto_frontier(
        current_altitude_km=550.0,
        inclination_deg=53.0,
        initial_miss_km=0.45,
        num_candidates=8,
    )

    candidates = res["pareto_candidates"]
    assert len(candidates) == 8

    # Verify sorting / monotonic separation growth with Delta-v
    for i in range(len(candidates) - 1):
        assert candidates[i]["delta_v_m_s"] < candidates[i + 1]["delta_v_m_s"]
        assert candidates[i]["resulting_separation_km"] < candidates[i + 1]["resulting_separation_km"]

    recs = res["optimal_recommendations"]
    assert "minimum_propellant_solution" in recs
    assert "balanced_optimal_solution" in recs
    assert "maximum_clearance_solution" in recs
