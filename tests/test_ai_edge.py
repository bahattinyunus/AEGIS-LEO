import pytest
import torch
import numpy as np
from aegis_leo.ai.pinn import ThermosphereDragPINN, get_pretrained_edge_pinn
from aegis_leo.ai.edge_optimizer import EdgeManeuverOptimizer
from aegis_leo.ai.quantizer import benchmark_edge_inference


def test_pinn_inference_shape_and_values():
    model = get_pretrained_edge_pinn()
    res = model.predict_fast(
        altitude_km=550.0,
        kp=8.0,
        f10_7=240.0,
        area_to_mass=0.02,
        v_rel_km_s=7.6,
    )

    assert "density_multiplier" in res
    assert "drag_acceleration_m_s2" in res
    assert "along_track_drift_km_per_day" in res
    assert res["density_multiplier"] >= 1.0
    assert res["drag_acceleration_m_s2"] >= 0.0


def test_pinn_physics_loss_gradient():
    model = ThermosphereDragPINN()
    model.train()

    inputs = torch.randn(8, 5)
    targets = torch.randn(8, 3)
    preds = model(inputs)

    loss = model.physics_loss_function(preds, inputs, targets)
    assert not torch.isnan(loss)
    loss.backward()

    # Verify gradients computed
    for p in model.parameters():
        assert p.grad is not None


def test_maneuver_optimizer_vis_viva_and_savings():
    optimizer = EdgeManeuverOptimizer(
        satellite_mass_kg=260.0,
        fuel_remaining_kg=15.0,
        isp_seconds=250.0,
        thrust_newtons=0.1,
    )

    plan = optimizer.plan_autonomous_avoidance(
        current_altitude_km=550.0,
        inclination_deg=53.0,
        initial_miss_distance_km=0.35,
        storm_kp=8.0,
        storm_f10_7=250.0,
        along_track_disp_km=14.0,
    )

    assert plan["status"] == "AUTONOMOUS_BURN_PLANNED"
    assert plan["proactive_burn"]["delta_v_m_s"] > 0.0
    assert plan["proactive_burn"]["propellant_consumed_kg"] > 0.0
    # Proactive burn should save fuel compared to reactive emergency burn
    assert plan["efficiency_metrics"]["fuel_savings_percentage"] > 10.0
    # Post burn risk should be lower than pre burn risk
    assert plan["proactive_burn"]["post_burn_adri"] < plan["pre_burn_adri"]


def test_edge_benchmarking():
    model = get_pretrained_edge_pinn()
    bench = benchmark_edge_inference(model, num_iterations=100)

    assert bench["latency_microseconds"] > 0.0
    assert bench["int8_quantized_memory_kb"] < 100.0  # Compact footprint
    assert bench["parameter_count"] > 0
