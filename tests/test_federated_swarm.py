"""
Unit and Integration Tests for AEGIS-LEO Federated Swarm Learning
================================================================
Validates:
1. ThermosphericSurrogateMLP forward, backward, numerical gradient checks,
   physics drag consistency loss, activations, parameter serialization and cloning.
2. SatelliteClient local training, FedProx proximal regularization, straggler handling,
   and power state transitions.
3. ISL quantization (INT8, Top-K sparsification), channel packet drops, bandwidth savings.
4. Privacy auditor enforcing zero raw telemetry leakage and DP noise injection.
5. Constellation aggregation strategies: FedAvg, FedProx, Hierarchical, Trimmed Mean
   (Byzantine/SEU resilience), Median, and Norm Bounded.
6. Multi-plane non-IID telemetry data generation.
7. Full end-to-end multi-round Walker constellation federated training.
"""

from __future__ import annotations
import math
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pytest

from aegis_leo.ai.federated_swarm import (
    AggregationStrategy,
    QuantizationMode,
    NodePowerState,
    ModelConfig,
    LocalTrainUpdate,
    RoundSummary,
    TelemetryNormalizer,
    ThermosphericSurrogateMLP,
    ISLQuantizer,
    ISLNetwork,
    SatelliteClient,
    ConstellationDataGenerator,
    TelemetryPrivacyAuditor,
    FederatedSwarmCoordinator,
    build_walker_constellation_federation,
    compare_fedavg_vs_fedprox,
)


# ============================================================================
# 1. Neural Network Architecture & Physics Tests
# ============================================================================

def test_model_initialization_and_forward_pass():
    """Verify layer weight dimensions, parameter count, and forward pass shape."""
    config = ModelConfig(
        input_dim=5,
        hidden_dims=(24, 12),
        output_dim=2,
        activation="silu",
        seed=101,
    )
    mlp = ThermosphericSurrogateMLP(config)

    # Layers: 5 -> 24 -> 12 -> 2
    # W1: (5, 24), b1: (24) -> 120 + 24 = 144
    # W2: (24, 12), b2: (12) -> 288 + 12 = 300
    # W3: (12, 2), b3: (2) -> 24 + 2 = 26
    # Total: 470 parameters
    assert mlp.parameter_count() == 470
    assert "W1" in mlp.parameters and "b1" in mlp.parameters
    assert "W3" in mlp.parameters and "b3" in mlp.parameters

    x_dummy = np.random.randn(8, 5)
    out = mlp.forward(x_dummy)
    assert out.shape == (8, 2)
    assert not np.isnan(out).any()


@pytest.mark.parametrize("activation", ["silu", "relu", "tanh"])
def test_model_activations(activation: str):
    """Verify different supported activation functions compute cleanly without NaNs."""
    config = ModelConfig(hidden_dims=(16,), activation=activation, seed=42)
    mlp = ThermosphericSurrogateMLP(config)
    x = np.linspace(-3.0, 3.0, 100).reshape(20, 5)
    out = mlp.forward(x)
    assert out.shape == (20, 2)
    assert np.isfinite(out).all()


def test_analytical_vs_numerical_gradients():
    """
    Perform finite-difference numerical gradient verification on analytical backpropagation.
    Ensures mathematical correctness of analytical gradients.
    """
    config = ModelConfig(
        input_dim=3,
        hidden_dims=(8,),
        output_dim=1,
        activation="silu",
        physics_loss_weight=0.0,
        weight_decay=1e-3,
        seed=77,
    )
    mlp = ThermosphericSurrogateMLP(config)

    rng = np.random.RandomState(123)
    x = rng.randn(4, 3)
    y = rng.randn(4, 1)

    loss_val, analytical_grads, _ = mlp.compute_loss_and_gradients(
        x_norm=x,
        y_norm=y,
        mu_prox=0.0,
    )

    eps = 1e-6
    for param_name, param_arr in mlp.parameters.items():
        ana_grad = analytical_grads[param_name]
        it = np.nditer(param_arr, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig_val = param_arr[idx]

            # f(w + eps)
            param_arr[idx] = orig_val + eps
            loss_plus, _, _ = mlp.compute_loss_and_gradients(x, y)

            # f(w - eps)
            param_arr[idx] = orig_val - eps
            loss_minus, _, _ = mlp.compute_loss_and_gradients(x, y)

            param_arr[idx] = orig_val

            num_grad = (loss_plus - loss_minus) / (2.0 * eps)
            ana_val = ana_grad[idx]

            diff = abs(num_grad - ana_val)
            denom = max(1e-5, abs(num_grad) + abs(ana_val))
            rel_err = diff / denom

            assert rel_err < 1e-3, (
                f"Gradient mismatch for {param_name}[{idx}]: "
                f"analytical={ana_val:.6f}, numerical={num_grad:.6f}, rel_err={rel_err:.6e}"
            )
            it.iternext()


def test_physics_loss_aerodynamic_consistency():
    """Verify physics loss penalizes deviations from the aerodynamic drag equation."""
    config = ModelConfig(
        physics_loss_weight=0.5,
        hidden_dims=(16, 8),
        seed=42,
    )
    mlp = ThermosphericSurrogateMLP(config)

    # Features: [altitude, Kp, F10.7, A/m, v_rel]
    x_raw = np.array([
        [400.0, 3.0, 120.0, 0.02, 7.6],
        [500.0, 5.0, 180.0, 0.015, 7.5],
    ], dtype=np.float64)

    # Compute exact theoretical drag for these inputs
    # rho = 10^(log_rho)
    # drag_micro = 0.5 * rho * (A/m) * (v_rel*1000)^2 * 1e6
    log_rhos = np.array([-11.8, -12.5])
    rhos = 10.0 ** log_rhos
    v_m_s = x_raw[:, 4] * 1000.0
    drags_exact = 0.5 * rhos * x_raw[:, 3] * (v_m_s ** 2) * 1e6

    y_exact = np.column_stack([log_rhos, drags_exact])
    x_norm = mlp.normalizer.normalize_x(x_raw)
    y_norm = mlp.normalizer.normalize_y(y_exact)

    _, _, metrics_consistent = mlp.compute_loss_and_gradients(
        x_norm=x_norm,
        y_norm=y_norm,
        x_raw=x_raw,
    )

    # Now disturb drag targets significantly
    y_disturbed = y_exact.copy()
    y_disturbed[:, 1] += 50.0  # huge physical violation
    y_disturbed_norm = mlp.normalizer.normalize_y(y_disturbed)

    _, _, metrics_disturbed = mlp.compute_loss_and_gradients(
        x_norm=x_norm,
        y_norm=y_disturbed_norm,
        x_raw=x_raw,
    )

    assert metrics_disturbed["total_loss"] > metrics_consistent["total_loss"]


def test_parameter_flattening_unflattening_and_cloning():
    """Verify parameter vectorization roundtrip and deep cloning independence."""
    mlp = ThermosphericSurrogateMLP()
    flat = mlp.flatten_parameters()
    assert flat.ndim == 1
    assert flat.size == mlp.parameter_count()

    reconstructed = mlp.unflatten_parameters(flat)
    for k in mlp.parameters:
        np.testing.assert_array_equal(mlp.parameters[k], reconstructed[k])

    # Clone test
    clone = mlp.clone()
    assert clone.parameter_count() == mlp.parameter_count()

    # Mutate original and assert clone is unaffected
    mlp.parameters["W1"] += 5.0
    assert not np.allclose(mlp.parameters["W1"], clone.parameters["W1"])


def test_predict_fast_convenience():
    """Verify predict() returns valid physical units."""
    mlp = ThermosphericSurrogateMLP()
    pred = mlp.predict(
        altitude_km=450.0,
        kp=3.5,
        f10_7=130.0,
        area_to_mass=0.018,
        v_rel_km_s=7.65,
    )
    assert "density_kg_m3" in pred
    assert "drag_acceleration_m_s2" in pred
    assert pred["density_kg_m3"] > 0.0
    assert pred["drag_acceleration_m_s2"] >= 0.0


# ============================================================================
# 2. Satellite Client & FedProx Straggler Tests
# ============================================================================

def test_satellite_client_local_training_and_update():
    """Verify SatelliteClient trains locally and produces a valid LocalTrainUpdate."""
    client = SatelliteClient(
        satellite_id="SAT-P01-S01",
        plane_id=1,
        slot_id=1,
        altitude_km=500.0,
        inclination_deg=53.0,
    )

    x_train, y_train = ConstellationDataGenerator.generate_plane_dataset(
        plane_id=1,
        altitude_km=500.0,
        inclination_deg=53.0,
        num_samples=80,
        seed=1,
    )
    client.set_dataset(x_train, y_train)

    global_weights = client.model.get_parameters()
    update = client.train_local(
        global_params=global_weights,
        epochs=4,
        batch_size=16,
    )

    assert update.satellite_id == "SAT-P01-S01"
    assert update.plane_id == 1
    assert update.epochs_completed == 4
    assert update.target_epochs == 4
    assert not update.is_straggler
    assert update.train_loss_final < update.train_loss_initial
    assert update.delta_norm > 0.0
    assert len(update.parameter_deltas) == len(global_weights)


def test_straggler_behavior_under_power_constraint():
    """Verify straggler nodes execute reduced epochs according to work ratio."""
    client_normal = SatelliteClient(
        satellite_id="SAT-SUN-01",
        plane_id=0,
        slot_id=0,
        altitude_km=400.0,
        inclination_deg=45.0,
        is_straggler=False,
    )
    client_straggler = SatelliteClient(
        satellite_id="SAT-ECLIPSE-02",
        plane_id=0,
        slot_id=1,
        altitude_km=400.0,
        inclination_deg=45.0,
        is_straggler=True,
        straggler_work_ratio=0.4,
        power_state=NodePowerState.ECLIPSE_BATTERY_SAVING,
    )

    x, y = ConstellationDataGenerator.generate_plane_dataset(0, 400.0, 45.0, num_samples=60)
    client_normal.set_dataset(x, y)
    client_straggler.set_dataset(x, y)

    gw = client_normal.model.get_parameters()
    u_norm = client_normal.train_local(gw, epochs=10)
    u_strag = client_straggler.train_local(gw, epochs=10)

    assert u_norm.epochs_completed == 10
    assert u_strag.epochs_completed == 4  # ceil(10 * 0.4) = 4
    assert u_strag.is_straggler is True
    assert u_strag.power_state == NodePowerState.ECLIPSE_BATTERY_SAVING


def test_fedprox_proximal_term_restrains_client_drift():
    """
    Verify FedProx with mu > 0 restrains local parameter updates from drifting
    too far away from global reference parameters compared to mu = 0.
    """
    client = SatelliteClient(
        satellite_id="SAT-DRIFT-TEST",
        plane_id=2,
        slot_id=0,
        altitude_km=600.0,
        inclination_deg=70.0,
    )
    x, y = ConstellationDataGenerator.generate_plane_dataset(2, 600.0, 70.0, num_samples=60, seed=12)
    client.set_dataset(x, y)

    gw = client.model.get_parameters()

    # Training with standard FedAvg (mu=0)
    update_fedavg = client.train_local(gw, epochs=8, batch_size=16, mu_prox=0.0)

    # Training with FedProx (mu=0.2)
    update_fedprox = client.train_local(gw, epochs=8, batch_size=16, mu_prox=0.2)

    # FedProx update norm must be strictly smaller because proximal term penalizes distance
    assert update_fedprox.delta_norm < update_fedavg.delta_norm


# ============================================================================
# 3. ISL Quantization, Compression & Network Tests
# ============================================================================

def test_isl_int8_quantization_roundtrip():
    """Verify INT8 quantization achieves significant compression with high fidelity."""
    rng = np.random.RandomState(42)
    deltas = {
        "W1": rng.randn(32, 16).astype(np.float64) * 0.1,
        "b1": rng.randn(16).astype(np.float64) * 0.05,
    }

    raw_bytes = ISLQuantizer.compute_raw_size_bytes(deltas)
    q_dict, meta_dict, tx_bytes = ISLQuantizer.quantize_int8(deltas)

    # Compression check: int8 is ~1 byte vs 8 bytes for float64 -> > 70% savings
    assert tx_bytes < raw_bytes * 0.35

    # Dequantization
    deq_deltas = ISLQuantizer.dequantize_int8(q_dict, meta_dict)

    # Check numerical reconstruction accuracy
    for k in deltas:
        orig = deltas[k]
        recon = deq_deltas[k]
        abs_err = np.max(np.abs(orig - recon))
        rng_span = np.max(orig) - np.min(orig)
        rel_err = abs_err / max(1e-6, rng_span)
        assert rel_err < 0.02, f"INT8 quantization error too high on {k}: {rel_err}"


def test_isl_top_k_sparsification_roundtrip():
    """Verify Top-K sparsification retains largest magnitude updates."""
    rng = np.random.RandomState(99)
    arr = rng.randn(10, 10) * 0.01
    # Insert few large salient updates
    arr[2, 3] = 2.5
    arr[7, 1] = -3.0

    sparse_dict, tx_bytes = ISLQuantizer.sparsify_top_k({"W": arr}, keep_ratio=0.10)
    desparse_dict = ISLQuantizer.desparsify_top_k(sparse_dict)
    recon = desparse_dict["W"]

    # Top elements should be precisely preserved
    assert abs(recon[2, 3] - 2.5) < 1e-6
    assert abs(recon[7, 1] - (-3.0)) < 1e-6
    # 90% of elements should be zeroed
    zero_fraction = float(np.mean(recon == 0.0))
    assert zero_fraction >= 0.85


def test_isl_network_transmission_and_drop_simulation():
    """Verify ISL network models packet loss and tracks bandwidth."""
    net_lossy = ISLNetwork(packet_loss_rate=0.5, seed=42)
    net_lossless = ISLNetwork(packet_loss_rate=0.0, seed=42)

    client = SatelliteClient("SAT-01", 0, 0, 500.0, 53.0)
    x, y = ConstellationDataGenerator.generate_plane_dataset(0, 500.0, 53.0, num_samples=30)
    client.set_dataset(x, y)
    update = client.train_local(client.model.get_parameters(), epochs=1)

    # Lossless network must deliver update
    res_lossless = net_lossless.transmit_update(update, dest_plane_id=0)
    assert res_lossless is not None
    assert net_lossless.total_transmitted_bytes > 0

    # Lossy network must drop some fraction over multiple trials
    drops = 0
    for _ in range(50):
        if net_lossy.transmit_update(update, dest_plane_id=1) is None:
            drops += 1
    assert drops > 0


# ============================================================================
# 4. Privacy Auditor Tests
# ============================================================================

def test_privacy_auditor_zero_raw_telemetry_leakage():
    """Verify privacy auditor permits model weight deltas and rejects telemetry leaks."""
    client = SatelliteClient("SAT-PRIV-01", 0, 0, 450.0, 50.0)
    x, y = ConstellationDataGenerator.generate_plane_dataset(0, 450.0, 50.0, num_samples=40)
    client.set_dataset(x, y)
    valid_update = client.train_local(client.model.get_parameters(), epochs=1)

    # Valid parameter update
    assert TelemetryPrivacyAuditor.audit_update_payload(valid_update) is True

    # Tampered update with invalid non-parameter key
    tampered_update = LocalTrainUpdate(
        satellite_id=valid_update.satellite_id,
        plane_id=valid_update.plane_id,
        slot_id=valid_update.slot_id,
        num_samples=valid_update.num_samples,
        epochs_completed=valid_update.epochs_completed,
        target_epochs=valid_update.target_epochs,
        is_straggler=valid_update.is_straggler,
        train_loss_initial=valid_update.train_loss_initial,
        train_loss_final=valid_update.train_loss_final,
        parameter_deltas={"raw_telemetry_x": x},  # Privacy violation!
        delta_norm=1.0,
        compute_time_ms=10.0,
        transmitted_bytes=100,
    )
    assert TelemetryPrivacyAuditor.audit_update_payload(tampered_update) is False


def test_differential_privacy_noise_addition():
    """Verify DP noise mechanism adds perturbation without modifying tensor shape."""
    deltas = {
        "W1": np.ones((10, 5), dtype=np.float64),
        "b1": np.zeros(5, dtype=np.float64),
    }
    dp_deltas = TelemetryPrivacyAuditor.compute_differential_privacy_noise(
        deltas,
        epsilon=2.0,
        delta_dp=1e-4,
        clip_norm=0.5,
        seed=123,
    )
    assert dp_deltas["W1"].shape == (10, 5)
    assert dp_deltas["b1"].shape == (5,)
    # Noise should cause values to deviate from 1.0
    assert not np.allclose(dp_deltas["W1"], 1.0)


# ============================================================================
# 5. Aggregation Strategies & SEU Resilience Tests
# ============================================================================

def _create_mock_updates(values: List[float]) -> List[LocalTrainUpdate]:
    """Helper to create dummy updates with scalar array parameters."""
    updates = []
    for i, v in enumerate(values):
        updates.append(
            LocalTrainUpdate(
                satellite_id=f"SAT-{i}",
                plane_id=i % 2,
                slot_id=i,
                num_samples=100,
                epochs_completed=5,
                target_epochs=5,
                is_straggler=False,
                train_loss_initial=1.0,
                train_loss_final=0.5,
                parameter_deltas={"W": np.array([v], dtype=np.float64)},
                delta_norm=abs(v),
                compute_time_ms=5.0,
                transmitted_bytes=64,
            )
        )
    return updates


def test_fedavg_aggregation():
    """Verify FedAvg calculates exact weighted mean."""
    updates = _create_mock_updates([1.0, 2.0, 3.0])
    agg = FederatedSwarmCoordinator.aggregate_fed_avg(updates)
    assert np.isclose(agg["W"][0], 2.0)


def test_trimmed_mean_byzantine_seu_resilience():
    """
    Verify Trimmed Mean rejects Single Event Upset (SEU) corrupted weights.
    When a radiation event causes an extreme bit flip on one satellite (e.g. +10,000.0),
    Trimmed Mean successfully eliminates it, whereas FedAvg gets completely ruined!
    """
    # 7 satellites: 6 nominal around 1.0, 1 SEU bit-flipped node with 10000.0
    vals = [0.95, 1.02, 0.98, 1.05, 1.00, 0.97, 10000.0]
    updates = _create_mock_updates(vals)

    coord_trimmed = FederatedSwarmCoordinator(
        aggregation_strategy=AggregationStrategy.TRIMMED_MEAN,
        trimmed_mean_beta=0.15,  # trims 1 top and 1 bottom
    )
    agg_trimmed = coord_trimmed.aggregate_trimmed_mean(updates)
    agg_fedavg = FederatedSwarmCoordinator.aggregate_fed_avg(updates)

    # Trimmed mean remains near ~1.0
    assert abs(agg_trimmed["W"][0] - 1.0) < 0.1
    # FedAvg is ruined by the SEU radiation outlier
    assert agg_fedavg["W"][0] > 1000.0


def test_coordinate_median_aggregation():
    """Verify coordinate-wise median handles skewed distributions."""
    updates = _create_mock_updates([1.0, 2.0, 100.0])
    agg = FederatedSwarmCoordinator.aggregate_coordinate_median(updates)
    assert np.isclose(agg["W"][0], 2.0)


def test_hierarchical_plane_aggregation():
    """Verify hierarchical aggregation aggregates intra-plane then inter-plane."""
    # Plane 0: two satellites with 1.0 and 3.0 -> plane avg 2.0
    # Plane 1: two satellites with 4.0 and 6.0 -> plane avg 5.0
    # Global: average of plane 0 and 1 -> 3.5
    u0 = _create_mock_updates([1.0, 3.0])
    u0[0].plane_id = 0
    u0[1].plane_id = 0

    u1 = _create_mock_updates([4.0, 6.0])
    u1[0].plane_id = 1
    u1[1].plane_id = 1

    coord = FederatedSwarmCoordinator(aggregation_strategy=AggregationStrategy.HIERARCHICAL_FED_AVG)
    agg = coord.aggregate_hierarchical(u0 + u1)
    assert np.isclose(agg["W"][0], 3.5)


def test_norm_bounded_clipping():
    """Verify norm-bounded aggregation clips oversized updates."""
    updates = _create_mock_updates([10.0])
    coord = FederatedSwarmCoordinator(
        aggregation_strategy=AggregationStrategy.NORM_BOUNDED,
        norm_bound_clip=4.0,
    )
    agg = coord.aggregate_norm_bounded(updates)
    assert np.isclose(agg["W"][0], 4.0)


# ============================================================================
# 6. Constellation Data Generator Tests
# ============================================================================

def test_data_generator_physical_bounds():
    """Verify synthetic telemetry generator adheres to atmospheric physics."""
    # Test quiet vs storm at 400 km
    log_rho_quiet, drag_quiet = ConstellationDataGenerator.compute_ground_truth_physics(
        altitude_km=400.0,
        kp=1.5,
        f10_7=70.0,
    )
    log_rho_storm, drag_storm = ConstellationDataGenerator.compute_ground_truth_physics(
        altitude_km=400.0,
        kp=8.0,
        f10_7=220.0,
    )

    # Storm density and drag must be significantly greater than quiet
    assert log_rho_storm > log_rho_quiet
    assert drag_storm > drag_quiet

    # Density at 300 km must be higher than at 700 km
    log_rho_low, _ = ConstellationDataGenerator.compute_ground_truth_physics(300.0, 3.0, 120.0)
    log_rho_high, _ = ConstellationDataGenerator.compute_ground_truth_physics(700.0, 3.0, 120.0)
    assert log_rho_low > log_rho_high


def test_generate_plane_dataset_shapes():
    """Verify dataset generation returns correct dimensions and data types."""
    x, y = ConstellationDataGenerator.generate_plane_dataset(
        plane_id=0,
        altitude_km=500.0,
        inclination_deg=53.0,
        num_samples=50,
    )
    assert x.shape == (50, 5)
    assert y.shape == (50, 2)
    assert x.dtype == np.float64
    assert y.dtype == np.float64
    # All altitudes should hover around 500 km
    assert np.all(x[:, 0] >= 480.0) and np.all(x[:, 0] <= 520.0)


# ============================================================================
# 7. End-to-End Walker Constellation Multi-Round FL Tests
# ============================================================================

def test_walker_constellation_federated_training_end_to_end():
    """
    Complete integration test:
    Builds a 3-plane Walker constellation with stragglers, trains 3 rounds
    using FedProx and INT8 quantization, and verifies loss reduction.
    """
    coordinator, clients, val_dataset = build_walker_constellation_federation(
        num_planes=3,
        sats_per_plane=2,
        base_altitude_km=450.0,
        altitude_spacing_km=80.0,
        straggler_fraction=0.33,
        samples_per_sat=60,
        aggregation_strategy=AggregationStrategy.FED_PROX,
        quantization_mode=QuantizationMode.INT8,
        mu_prox=0.02,
        seed=42,
    )

    assert coordinator.get_registered_count() == 6
    x_val, y_val = val_dataset
    assert len(x_val) == 150

    # Initial evaluation
    initial_eval = coordinator.global_model.evaluate(x_val, y_val)

    # Train 3 federated rounds
    history = coordinator.train_federated(
        num_rounds=3,
        client_fraction=1.0,
        local_epochs=3,
        local_batch_size=16,
        val_dataset=val_dataset,
    )

    assert len(history) == 3
    final_eval = coordinator.global_model.evaluate(x_val, y_val)

    # Global model loss should improve after federated learning
    assert final_eval["mse_norm"] < initial_eval["mse_norm"]
    assert history[-1].total_isl_bytes > 0
    assert history[-1].compression_ratio > 2.0  # INT8 compression ratio


def test_compare_fedavg_vs_fedprox_benchmark():
    """Verify compare_fedavg_vs_fedprox executes without errors and returns comparative metrics."""
    res = compare_fedavg_vs_fedprox(
        num_rounds=2,
        num_planes=2,
        sats_per_plane=2,
        straggler_fraction=0.5,
        mu_prox=0.03,
        seed=10,
    )

    assert "fedavg_final_val_loss" in res
    assert "fedprox_final_val_loss" in res
    assert "fedavg_history" in res
    assert "fedprox_history" in res
    assert len(res["fedavg_history"]) == 2
    assert len(res["fedprox_history"]) == 2
