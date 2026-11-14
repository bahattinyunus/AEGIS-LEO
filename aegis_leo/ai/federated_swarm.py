"""
AEGIS-LEO: Federated Learning for Satellite Swarms & Constellations
===================================================================
Collaborative training of thermospheric density and aerodynamic drag prediction
models across constellation orbital planes via Inter-Satellite Links (ISL)
without transmitting raw orbital telemetry.

Key Capabilities:
1. Pure NumPy Neural Network Surrogate (ThermosphericSurrogateMLP):
   - Multi-layer perceptron for sub-millisecond edge inference.
   - Multi-task output: log10 thermospheric density (kg/m^3) and drag acceleration (micro-m/s^2).
   - Physics-informed loss enforcing aerodynamic drag relationship:
     a_drag = 0.5 * rho * (Cd * A / m) * v_rel^2.
   - Proximal regularization gradient for FedProx optimization.

2. Federated Swarm Algorithms:
   - FedAvg (Federated Averaging): McMahan et al. weighted parameter aggregation.
   - FedProx: Li et al. proximal penalty handling computational stragglers
     (satellites under eclipse power throttling, thermal limits, or partial compute).
   - Hierarchical Intra-Plane / Inter-Plane ISL Aggregation:
     Intra-plane aggregation along high-bandwidth orbital rings, followed by
     inter-plane aggregation across Walker planes.
   - Byzantine-Resilient Aggregation:
     Trimmed Mean, Coordinate-wise Median, and Norm-bounded clipping to filter
     radiation-induced Single Event Upset (SEU) corruptions and telemetry sensor faults.

3. Constellation ISL Mesh & Bandwidth Optimization:
   - Quantized parameter exchange (INT8 uniform quantization and Top-K sparsification).
   - ISL channel simulation (latency, packet loss, Doppler contact windows).
   - Zero raw telemetry leakage: only quantized parameter deltas are shared.

4. Non-IID Orbital Physics Data Generation:
   - Multi-plane Walker constellation simulation across altitude layers (300 - 800 km),
     inclination regimes (equatorial, mid-inclination, polar auroral zones),
     and space weather storm events (Kp 0-9, F10.7 65-300 SFU).

Dependencies:
- numpy
- standard library (math, dataclasses, enum, typing, time, hashlib, copy)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    QUIET_KP,
    STORM_G5_KP,
    QUIET_F10_7,
    ACTIVE_F10_7,
    EXTREME_F10_7,
)


# ============================================================================
# Enums and System Types
# ============================================================================

class AggregationStrategy(str, Enum):
    """Federated aggregation algorithm."""
    FED_AVG = "FED_AVG"
    FED_PROX = "FED_PROX"
    HIERARCHICAL_FED_AVG = "HIERARCHICAL_FED_AVG"
    TRIMMED_MEAN = "TRIMMED_MEAN"
    COORDINATE_MEDIAN = "COORDINATE_MEDIAN"
    NORM_BOUNDED = "NORM_BOUNDED"


class QuantizationMode(str, Enum):
    """ISL parameter compression / quantization method."""
    NONE = "NONE"
    INT8 = "INT8"
    TOP_K_SPARSE = "TOP_K_SPARSE"


class NodePowerState(str, Enum):
    """Satellite power & operational state affecting edge compute."""
    NOMINAL_SUNLIT = "NOMINAL_SUNLIT"
    ECLIPSE_BATTERY_SAVING = "ECLIPSE_BATTERY_SAVING"
    THERMAL_THROTTLED = "THERMAL_THROTTLED"
    MANEUVER_PRIORITY = "MANEUVER_PRIORITY"


class ISLStatus(str, Enum):
    """Inter-satellite link communication state."""
    CONNECTED = "CONNECTED"
    OCCULTED = "OCCULTED"
    HIGH_DOPPLER_DROPOUT = "HIGH_DOPPLER_DROPOUT"
    BANDWIDTH_THROTTLED = "BANDWIDTH_THROTTLED"


# ============================================================================
# Configuration Dataclasses
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for the pure-NumPy thermospheric surrogate network."""
    input_dim: int = 5                  # [alt_km, Kp, F10.7, A/m, v_rel_km_s]
    hidden_dims: Tuple[int, ...] = (32, 16)
    output_dim: int = 2                 # [log10_density_kg_m3, drag_accel_micro_m_s2]
    activation: str = "silu"            # "silu", "relu", "tanh"
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    physics_loss_weight: float = 0.05   # Aerodynamic drag consistency loss weight
    seed: int = 42


@dataclass
class LocalTrainUpdate:
    """Summary of local satellite edge training."""
    satellite_id: str
    plane_id: int
    slot_id: int
    num_samples: int
    epochs_completed: int
    target_epochs: int
    is_straggler: bool
    train_loss_initial: float
    train_loss_final: float
    parameter_deltas: Dict[str, np.ndarray]
    delta_norm: float
    compute_time_ms: float
    transmitted_bytes: int
    power_state: NodePowerState = NodePowerState.NOMINAL_SUNLIT


@dataclass
class RoundSummary:
    """Summary metrics of a constellation federated learning round."""
    round_number: int
    global_train_loss: float
    global_val_loss: float
    participating_nodes: int
    straggler_nodes: int
    strategy_used: AggregationStrategy
    mean_delta_norm: float
    total_isl_bytes: int
    compression_ratio: float
    round_duration_s: float
    plane_losses: Dict[int, float] = field(default_factory=dict)


# ============================================================================
# Feature Normalizer
# ============================================================================

class TelemetryNormalizer:
    """
    Standardization and normalization for thermospheric telemetry.
    Ensures numerical stability across vast LEO dynamic ranges (alt 150-1000 km,
    density spanning 8 orders of magnitude from 1e-9 to 1e-17 kg/m^3).
    """

    def __init__(
        self,
        feature_means: Optional[np.ndarray] = None,
        feature_stds: Optional[np.ndarray] = None,
        target_means: Optional[np.ndarray] = None,
        target_stds: Optional[np.ndarray] = None,
    ):
        # Default physical baselines:
        # Inputs: [altitude_km, Kp, F10.7, Area_to_mass (m^2/kg), v_rel (km/s)]
        if feature_means is None:
            self.x_mean = np.array([500.0, 3.5, 130.0, 0.015, 7.6], dtype=np.float64)
        else:
            self.x_mean = np.asarray(feature_means, dtype=np.float64)

        if feature_stds is None:
            self.x_std = np.array([150.0, 2.5, 55.0, 0.010, 0.3], dtype=np.float64)
        else:
            self.x_std = np.asarray(feature_stds, dtype=np.float64)
        self.x_std = np.where(self.x_std == 0.0, 1.0, self.x_std)

        # Targets: [log10_density_kg_m3, drag_accel_micro_m_s2]
        # At 500 km quiet: rho ~ 5e-13 -> log10(rho) ~ -12.3; drag ~ 0.5 * 5e-13 * 0.015 * (7600)^2 * 1e6 ~ 0.22 um/s^2
        if target_means is None:
            self.y_mean = np.array([-12.0, 1.0], dtype=np.float64)
        else:
            self.y_mean = np.asarray(target_means, dtype=np.float64)

        if target_stds is None:
            self.y_std = np.array([2.5, 2.0], dtype=np.float64)
        else:
            self.y_std = np.asarray(target_stds, dtype=np.float64)
        self.y_std = np.where(self.y_std == 0.0, 1.0, self.y_std)

    def normalize_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def denormalize_x(self, x_norm: np.ndarray) -> np.ndarray:
        return x_norm * self.x_std + self.x_mean

    def normalize_y(self, y: np.ndarray) -> np.ndarray:
        return (y - self.y_mean) / self.y_std

    def denormalize_y(self, y_norm: np.ndarray) -> np.ndarray:
        return y_norm * self.y_std + self.y_mean


# ============================================================================
# Pure NumPy Multilayer Perceptron (Surrogate Model)
# ============================================================================

class ThermosphericSurrogateMLP:
    """
    Sub-millisecond Physics-Informed Edge Surrogate for Thermospheric Drag.
    Implemented exclusively in pure NumPy for deterministic, dependency-free
    onboard execution on spaceborne flight computers (RAD750, LEON4, ARM Cortex-R5).
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.normalizer = TelemetryNormalizer()
        self.rng = np.random.RandomState(self.config.seed)

        self.layer_sizes: List[int] = [
            self.config.input_dim,
            *self.config.hidden_dims,
            self.config.output_dim,
        ]

        # Weights and biases stored in dictionary
        self.parameters: Dict[str, np.ndarray] = {}
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """He / Xavier weight initialization."""
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]

            if self.config.activation == "relu":
                # He initialization
                std = math.sqrt(2.0 / fan_in)
            else:
                # Glorot / Xavier initialization
                std = math.sqrt(2.0 / (fan_in + fan_out))

            w_name = f"W{i + 1}"
            b_name = f"b{i + 1}"
            self.parameters[w_name] = self.rng.randn(fan_in, fan_out).astype(np.float64) * std
            self.parameters[b_name] = np.zeros(fan_out, dtype=np.float64)

    # ------------------------------------------------------------------------
    # Activations & Derivatives
    # ------------------------------------------------------------------------

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        z_clip = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_clip))

    def _activation_forward(self, z: np.ndarray) -> np.ndarray:
        act = self.config.activation.lower()
        if act == "silu":
            return z * self._sigmoid(z)
        elif act == "relu":
            return np.maximum(0.0, z)
        elif act == "tanh":
            return np.tanh(z)
        else:
            return z

    def _activation_backward(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        act = self.config.activation.lower()
        if act == "silu":
            sig = self._sigmoid(z)
            return sig * (1.0 + z * (1.0 - sig))
        elif act == "relu":
            return np.where(z > 0.0, 1.0, 0.0)
        elif act == "tanh":
            return 1.0 - a ** 2
        else:
            return np.ones_like(z)

    # ------------------------------------------------------------------------
    # Forward & Backward Pass
    # ------------------------------------------------------------------------

    def forward(
        self,
        x_norm: np.ndarray,
        return_cache: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Forward pass through the MLP.
        x_norm shape: (N, input_dim)
        Returns: y_norm shape (N, output_dim)
        """
        num_layers = len(self.layer_sizes) - 1
        a = x_norm
        cache = {"a0": a}

        for i in range(1, num_layers):
            w = self.parameters[f"W{i}"]
            b = self.parameters[f"b{i}"]
            z = np.dot(a, w) + b
            a = self._activation_forward(z)
            cache[f"z{i}"] = z
            cache[f"a{i}"] = a

        # Final linear output layer
        w_last = self.parameters[f"W{num_layers}"]
        b_last = self.parameters[f"b{num_layers}"]
        z_last = np.dot(a, w_last) + b_last
        cache[f"z{num_layers}"] = z_last
        cache[f"a{num_layers}"] = z_last

        if return_cache:
            return z_last, cache
        return z_last

    def compute_loss_and_gradients(
        self,
        x_norm: np.ndarray,
        y_norm: np.ndarray,
        x_raw: Optional[np.ndarray] = None,
        mu_prox: float = 0.0,
        global_parameters: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[float, Dict[str, np.ndarray], Dict[str, float]]:
        """
        Compute total loss and analytical backpropagation gradients.
        Incorporates:
        1. Task MSE loss: 0.5 * ||y_pred - y_true||^2
        2. Physics drag consistency penalty
        3. L2 weight decay
        4. FedProx proximal penalty: (mu_prox / 2) * ||w - w_global||^2
        """
        n_samples = x_norm.shape[0]
        if n_samples == 0:
            raise ValueError("Zero samples provided for gradient computation.")

        y_pred, cache = self.forward(x_norm, return_cache=True)

        # 1. Base MSE Loss
        diff = y_pred - y_norm
        mse_loss = float(np.mean(np.sum(diff ** 2, axis=1)))

        # 2. Physics Drag Consistency Term
        # y[:, 0] is normalized log10_density, y[:, 1] is normalized drag_accel
        phys_loss = 0.0
        phys_grad_y = np.zeros_like(y_pred)

        if self.config.physics_loss_weight > 0.0 and x_raw is not None:
            # Physical formula: a_drag_micro = 0.5 * 10^(log10_rho) * (A/m) * (v_rel*1000)^2 * 1e6
            y_pred_denorm = self.normalizer.denormalize_y(y_pred)
            pred_log_rho = y_pred_denorm[:, 0]
            pred_drag_micro = y_pred_denorm[:, 1]

            area_to_mass = x_raw[:, 3]
            v_rel_km_s = x_raw[:, 4]
            v_rel_m_s = v_rel_km_s * 1000.0

            # Expected physical drag
            # Bound log_rho for numerical stability in exponent
            log_rho_clipped = np.clip(pred_log_rho, -18.0, -8.0)
            rho_calc = 10.0 ** log_rho_clipped
            expected_drag_micro = 0.5 * rho_calc * area_to_mass * (v_rel_m_s ** 2) * 1e6

            drag_err_micro = pred_drag_micro - expected_drag_micro
            # Normalized physics error
            drag_err_norm = drag_err_micro / self.normalizer.y_std[1]
            phys_loss = float(np.mean(drag_err_norm ** 2))

            # Gradient contribution to output layer
            phys_grad_y[:, 1] += (2.0 / n_samples) * drag_err_norm

        # Total data + physics loss
        total_loss = mse_loss + self.config.physics_loss_weight * phys_loss

        # 3. L2 Weight Decay Loss
        l2_loss = 0.0
        num_layers = len(self.layer_sizes) - 1
        for i in range(1, num_layers + 1):
            w = self.parameters[f"W{i}"]
            l2_loss += 0.5 * self.config.weight_decay * float(np.sum(w ** 2))
        total_loss += l2_loss

        # 4. FedProx Proximal Loss Term: (mu / 2) * ||w - w_global||^2
        prox_loss = 0.0
        if mu_prox > 0.0 and global_parameters is not None:
            for k, w in self.parameters.items():
                if k in global_parameters:
                    delta = w - global_parameters[k]
                    prox_loss += 0.5 * mu_prox * float(np.sum(delta ** 2))
            total_loss += prox_loss

        # --------------------------------------------------------------------
        # Backpropagation
        # ------------------------------------------------------------------------
        grads: Dict[str, np.ndarray] = {}

        # Delta at output layer
        delta = (2.0 / n_samples) * diff + self.config.physics_loss_weight * phys_grad_y

        for i in range(num_layers, 0, -1):
            a_prev = cache[f"a{i - 1}"]
            w = self.parameters[f"W{i}"]

            # Gradient w.r.t weights and biases
            dw = np.dot(a_prev.T, delta) + self.config.weight_decay * w
            db = np.sum(delta, axis=0)

            # FedProx gradient contribution: mu * (w - w_global)
            if mu_prox > 0.0 and global_parameters is not None:
                w_key = f"W{i}"
                b_key = f"b{i}"
                if w_key in global_parameters:
                    dw += mu_prox * (w - global_parameters[w_key])
                if b_key in global_parameters:
                    db += mu_prox * (self.parameters[b_key] - global_parameters[b_key])

            grads[f"W{i}"] = dw
            grads[f"b{i}"] = db

            if i > 1:
                z_prev = cache[f"z{i - 1}"]
                a_prev_act = cache[f"a{i - 1}"]
                d_act = self._activation_backward(z_prev, a_prev_act)
                delta = np.dot(delta, w.T) * d_act

        metrics = {
            "mse_loss": mse_loss,
            "phys_loss": phys_loss,
            "prox_loss": prox_loss,
            "total_loss": total_loss,
        }
        return total_loss, grads, metrics

    def apply_gradients(self, grads: Dict[str, np.ndarray], lr: Optional[float] = None) -> None:
        """Standard gradient descent step."""
        eta = lr if lr is not None else self.config.learning_rate
        for k in self.parameters:
            if k in grads:
                self.parameters[k] -= eta * grads[k]

    # ------------------------------------------------------------------------
    # Parameter Serialization & Math Helpers
    # ------------------------------------------------------------------------

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Returns deep copy of current parameters."""
        return {k: v.copy() for k, v in self.parameters.items()}

    def set_parameters(self, params: Dict[str, np.ndarray]) -> None:
        """Sets internal parameters from provided dict."""
        for k, v in params.items():
            if k in self.parameters:
                self.parameters[k] = v.copy()

    def flatten_parameters(self) -> np.ndarray:
        """Flattens all network parameters into a single 1D vector."""
        vecs = []
        for k in sorted(self.parameters.keys()):
            vecs.append(self.parameters[k].ravel())
        return np.concatenate(vecs)

    def unflatten_parameters(self, flat_vector: np.ndarray) -> Dict[str, np.ndarray]:
        """Reconstructs parameter dictionary from 1D vector."""
        params: Dict[str, np.ndarray] = {}
        idx = 0
        for k in sorted(self.parameters.keys()):
            shape = self.parameters[k].shape
            size = self.parameters[k].size
            params[k] = flat_vector[idx : idx + size].reshape(shape).copy()
            idx += size
        return params

    def parameter_count(self) -> int:
        """Total scalar parameters in the network."""
        return sum(v.size for v in self.parameters.values())

    def clone(self) -> ThermosphericSurrogateMLP:
        """Create an independent copy with identical architecture and weights."""
        new_mlp = ThermosphericSurrogateMLP(self.config)
        new_mlp.normalizer = copy.deepcopy(self.normalizer)
        new_mlp.set_parameters(self.get_parameters())
        return new_mlp

    def evaluate(
        self,
        x_raw: np.ndarray,
        y_raw: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model on unnormalized test dataset.
        Returns physical metrics (MSE, MAE of density and drag).
        """
        x_norm = self.normalizer.normalize_x(x_raw)
        y_norm = self.normalizer.normalize_y(y_raw)

        y_pred_norm = self.forward(x_norm)
        y_pred_raw = self.normalizer.denormalize_y(y_pred_norm)

        mse_norm = float(np.mean((y_pred_norm - y_norm) ** 2))
        mae_log_rho = float(np.mean(np.abs(y_pred_raw[:, 0] - y_raw[:, 0])))
        mae_drag = float(np.mean(np.abs(y_pred_raw[:, 1] - y_raw[:, 1])))

        # Relative density percentage error
        rho_pred = 10.0 ** np.clip(y_pred_raw[:, 0], -18.0, -8.0)
        rho_true = 10.0 ** np.clip(y_raw[:, 0], -18.0, -8.0)
        rel_rho_err_pct = float(np.mean(np.abs(rho_pred - rho_true) / rho_true) * 100.0)

        return {
            "mse_norm": mse_norm,
            "mae_log_density": mae_log_rho,
            "mae_drag_micro_m_s2": mae_drag,
            "mean_relative_density_error_pct": rel_rho_err_pct,
        }

    def predict(
        self,
        altitude_km: float,
        kp: float,
        f10_7: float,
        area_to_mass: float,
        v_rel_km_s: float = 7.6,
    ) -> Dict[str, float]:
        """
        Convenience edge prediction returning physical units.
        Returns:
            density_kg_m3: Thermospheric density (kg/m^3)
            log10_density: Base-10 logarithm of density
            drag_acceleration_m_s2: Drag acceleration (m/s^2)
        """
        x_raw = np.array([[altitude_km, kp, f10_7, area_to_mass, v_rel_km_s]], dtype=np.float64)
        x_norm = self.normalizer.normalize_x(x_raw)
        y_pred_norm = self.forward(x_norm)
        y_pred_raw = self.normalizer.denormalize_y(y_pred_norm)[0]

        log_rho = float(y_pred_raw[0])
        density = float(10.0 ** np.clip(log_rho, -18.0, -8.0))
        drag_micro = float(max(0.0, y_pred_raw[1]))
        drag_m_s2 = drag_micro * 1e-6

        return {
            "density_kg_m3": density,
            "log10_density": log_rho,
            "drag_acceleration_m_s2": drag_m_s2,
            "drag_acceleration_micro_m_s2": drag_micro,
        }


# ============================================================================
# ISL Quantization and Compression Engine
# ============================================================================

class ISLQuantizer:
    """
    Quantizes parameter deltas for bandwidth-constrained Inter-Satellite Links.
    Reduces transmission volume by 75% (INT8) to 90%+ (Top-K Sparsification).
    """

    @staticmethod
    def quantize_int8(
        delta_dict: Dict[str, np.ndarray],
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[float, float]], int]:
        """
        Uniform INT8 quantization: maps float64 array to int8 with scale and min_val.
        Returns:
            quantized_dict: int8 arrays
            meta_dict: (min_val, scale) per parameter
            total_bytes: total payload size in bytes
        """
        q_dict: Dict[str, np.ndarray] = {}
        meta_dict: Dict[str, Tuple[float, float]] = {}
        total_bytes = 0

        for k, arr in delta_dict.items():
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            rng = max_val - min_val

            if rng <= 1e-12:
                scale = 1.0
                q_arr = np.zeros_like(arr, dtype=np.int8)
            else:
                scale = rng / 254.0
                q_norm = np.clip(np.round((arr - min_val) / scale) - 127, -128, 127)
                q_arr = q_norm.astype(np.int8)

            q_dict[k] = q_arr
            meta_dict[k] = (min_val, scale)
            # 1 byte per int8 + 16 bytes overhead for metadata
            total_bytes += q_arr.nbytes + 16

        return q_dict, meta_dict, total_bytes

    @staticmethod
    def dequantize_int8(
        q_dict: Dict[str, np.ndarray],
        meta_dict: Dict[str, Tuple[float, float]],
    ) -> Dict[str, np.ndarray]:
        """Dequantizes INT8 representation back to floating point."""
        deq_dict: Dict[str, np.ndarray] = {}
        for k, q_arr in q_dict.items():
            min_val, scale = meta_dict[k]
            deq_dict[k] = (q_arr.astype(np.float64) + 127.0) * scale + min_val
        return deq_dict

    @staticmethod
    def sparsify_top_k(
        delta_dict: Dict[str, np.ndarray],
        keep_ratio: float = 0.20,
    ) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]], int]:
        """
        Top-K sparsification: retains only the top keep_ratio fraction of parameters
        by absolute magnitude. Sends non-zero indices and values.
        Returns:
            sparse_dict: {k: (indices_1d, values, original_shape)}
            total_bytes: approximate wire transmission bytes
        """
        ratio = np.clip(keep_ratio, 0.01, 1.0)
        sparse_dict = {}
        total_bytes = 0

        for k, arr in delta_dict.items():
            shape = arr.shape
            flat = arr.ravel()
            n_elements = flat.size
            k_count = max(1, int(math.ceil(n_elements * ratio)))

            abs_vals = np.abs(flat)
            # Find threshold
            if k_count < n_elements:
                top_indices = np.argpartition(abs_vals, -k_count)[-k_count:]
            else:
                top_indices = np.arange(n_elements)

            sparse_vals = flat[top_indices].astype(np.float32)
            sparse_indices = top_indices.astype(np.int32)

            sparse_dict[k] = (sparse_indices, sparse_vals, shape)
            # 4 bytes for float32 value + 4 bytes for int32 index
            total_bytes += (sparse_vals.nbytes + sparse_indices.nbytes)

        return sparse_dict, total_bytes

    @staticmethod
    def desparsify_top_k(
        sparse_dict: Dict[str, Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]],
    ) -> Dict[str, np.ndarray]:
        """Reconstructs dense parameter deltas from Top-K sparse representation."""
        dense_dict: Dict[str, np.ndarray] = {}
        for k, (indices, vals, shape) in sparse_dict.items():
            flat = np.zeros(shape, dtype=np.float64).ravel()
            flat[indices] = vals.astype(np.float64)
            dense_dict[k] = flat.reshape(shape)
        return dense_dict

    @staticmethod
    def compute_raw_size_bytes(delta_dict: Dict[str, np.ndarray]) -> int:
        """Computes raw uncompressed float64 size in bytes."""
        return sum(arr.nbytes for arr in delta_dict.values())


# ============================================================================
# Inter-Satellite Link (ISL) Network Model
# ============================================================================

class ISLNetwork:
    """
    Models Inter-Satellite Link (ISL) communication constraints:
    - Intra-plane links (continuous, low latency, high throughput)
    - Inter-plane links (periodic Walker cross-plane contact, Doppler drift, eclipse occultation)
    - Packet loss and bandwidth budget tracking.
    """

    def __init__(
        self,
        packet_loss_rate: float = 0.0,
        intra_plane_latency_ms: float = 5.0,
        inter_plane_latency_ms: float = 25.0,
        quantization_mode: QuantizationMode = QuantizationMode.NONE,
        top_k_ratio: float = 0.25,
        seed: int = 42,
    ):
        self.packet_loss_rate = packet_loss_rate
        self.intra_plane_latency_ms = intra_plane_latency_ms
        self.inter_plane_latency_ms = inter_plane_latency_ms
        self.quantization_mode = quantization_mode
        self.top_k_ratio = top_k_ratio
        self.rng = np.random.RandomState(seed)

        self.total_transmitted_bytes = 0
        self.total_raw_bytes = 0
        self.dropped_packets_count = 0

    def transmit_update(
        self,
        update: LocalTrainUpdate,
        dest_plane_id: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Simulate transmission of satellite parameter delta across ISL.
        Applies quantization, calculates bandwidth, and models potential link dropouts.
        Returns:
            Received dequantized parameter deltas, or None if packet was lost.
        """
        # Determine link type and drop probability
        is_intra_plane = (update.plane_id == dest_plane_id)
        drop_prob = self.packet_loss_rate if is_intra_plane else (self.packet_loss_rate * 1.5)

        if self.rng.rand() < drop_prob:
            self.dropped_packets_count += 1
            return None

        raw_bytes = ISLQuantizer.compute_raw_size_bytes(update.parameter_deltas)
        self.total_raw_bytes += raw_bytes

        if self.quantization_mode == QuantizationMode.INT8:
            q_dict, meta_dict, tx_bytes = ISLQuantizer.quantize_int8(update.parameter_deltas)
            self.total_transmitted_bytes += tx_bytes
            update.transmitted_bytes = tx_bytes
            return ISLQuantizer.dequantize_int8(q_dict, meta_dict)

        elif self.quantization_mode == QuantizationMode.TOP_K_SPARSE:
            sparse_dict, tx_bytes = ISLQuantizer.sparsify_top_k(
                update.parameter_deltas,
                keep_ratio=self.top_k_ratio,
            )
            self.total_transmitted_bytes += tx_bytes
            update.transmitted_bytes = tx_bytes
            return ISLQuantizer.desparsify_top_k(sparse_dict)

        else:
            self.total_transmitted_bytes += raw_bytes
            update.transmitted_bytes = raw_bytes
            return {k: v.copy() for k, v in update.parameter_deltas.items()}


# ============================================================================
# Satellite Client (Onboard Edge Node)
# ============================================================================

class SatelliteClient:
    """
    Simulates an individual satellite node in a Walker constellation.
    Maintains local private telemetry collected during orbital passes.
    Supports FedAvg and FedProx (with proximal penalty) for local optimization.
    """

    def __init__(
        self,
        satellite_id: str,
        plane_id: int,
        slot_id: int,
        altitude_km: float,
        inclination_deg: float,
        model_config: Optional[ModelConfig] = None,
        is_straggler: bool = False,
        straggler_work_ratio: float = 0.4,
        power_state: NodePowerState = NodePowerState.NOMINAL_SUNLIT,
    ):
        self.satellite_id = satellite_id
        self.plane_id = plane_id
        self.slot_id = slot_id
        self.altitude_km = altitude_km
        self.inclination_deg = inclination_deg

        self.is_straggler = is_straggler
        self.straggler_work_ratio = straggler_work_ratio
        self.power_state = power_state

        self.model = ThermosphericSurrogateMLP(model_config)
        self.local_x: Optional[np.ndarray] = None
        self.local_y: Optional[np.ndarray] = None

    def set_dataset(self, x_raw: np.ndarray, y_raw: np.ndarray) -> None:
        """Ingests local private telemetry without transmitting it anywhere."""
        self.local_x = x_raw.copy()
        self.local_y = y_raw.copy()

    def receive_global_model(self, global_params: Dict[str, np.ndarray]) -> None:
        """Overwrites local parameters with aggregated global parameters."""
        self.model.set_parameters(global_params)

    def train_local(
        self,
        global_params: Dict[str, np.ndarray],
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: Optional[float] = None,
        mu_prox: float = 0.0,
    ) -> LocalTrainUpdate:
        """
        Execute local model training on onboard telemetry.
        If satellite is a straggler (due to eclipse battery saving or thermal throttle),
        it completes fewer local epochs (FedProx gracefully accommodates this).
        """
        if self.local_x is None or self.local_y is None or len(self.local_x) == 0:
            raise ValueError(f"Satellite {self.satellite_id} has no telemetry dataset loaded.")

        # Sync with global model before training
        self.receive_global_model(global_params)
        w_global_copy = {k: v.copy() for k, v in global_params.items()}

        x_norm = self.model.normalizer.normalize_x(self.local_x)
        y_norm = self.model.normalizer.normalize_y(self.local_y)
        n_samples = len(self.local_x)

        # Initial loss
        y_init_pred = self.model.forward(x_norm)
        init_loss = float(np.mean((y_init_pred - y_norm) ** 2))

        # Determine actual epochs completed
        target_epochs = epochs
        if self.is_straggler:
            actual_epochs = max(1, int(math.ceil(epochs * self.straggler_work_ratio)))
        else:
            actual_epochs = epochs

        t_start = time.perf_counter()
        lr = learning_rate if learning_rate is not None else self.model.config.learning_rate

        # Training Loop (Mini-batch SGD with FedProx proximal term)
        rng = np.random.RandomState(42 + self.slot_id)
        indices = np.arange(n_samples)

        for _ in range(actual_epochs):
            rng.shuffle(indices)
            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                x_b_norm = x_norm[batch_idx]
                y_b_norm = y_norm[batch_idx]
                x_b_raw = self.local_x[batch_idx]

                _, grads, _ = self.model.compute_loss_and_gradients(
                    x_norm=x_b_norm,
                    y_norm=y_b_norm,
                    x_raw=x_b_raw,
                    mu_prox=mu_prox,
                    global_parameters=w_global_copy,
                )
                self.model.apply_gradients(grads, lr=lr)

        compute_time_ms = (time.perf_counter() - t_start) * 1000.0

        # Final loss
        y_final_pred = self.model.forward(x_norm)
        final_loss = float(np.mean((y_final_pred - y_norm) ** 2))

        # Parameter Delta: Delta_w = w_local - w_global
        deltas: Dict[str, np.ndarray] = {}
        delta_vecs = []
        for k in self.model.parameters:
            d = self.model.parameters[k] - w_global_copy[k]
            deltas[k] = d
            delta_vecs.append(d.ravel())

        delta_norm = float(np.linalg.norm(np.concatenate(delta_vecs)))

        return LocalTrainUpdate(
            satellite_id=self.satellite_id,
            plane_id=self.plane_id,
            slot_id=self.slot_id,
            num_samples=n_samples,
            epochs_completed=actual_epochs,
            target_epochs=target_epochs,
            is_straggler=self.is_straggler,
            train_loss_initial=init_loss,
            train_loss_final=final_loss,
            parameter_deltas=deltas,
            delta_norm=delta_norm,
            compute_time_ms=compute_time_ms,
            transmitted_bytes=0,
            power_state=self.power_state,
        )


# ============================================================================
# Synthetic Constellation Telemetry & Non-IID Data Generator
# ============================================================================

class ConstellationDataGenerator:
    """
    Generates physically consistent thermospheric telemetry across constellation
    orbital planes. Simulates realistic non-IID conditions:
    - Plane 0: 350 km (high drag, high solar flux sensitivity)
    - Plane 1: 500 km (mid LEO, diurnal day/night bulge)
    - Plane 2: 650 km (moderate drag)
    - Plane 3: 800 km polar plane (85 deg inc: intense auroral heating during storms)
    """

    @staticmethod
    def compute_ground_truth_physics(
        altitude_km: float,
        kp: float,
        f10_7: float,
        area_to_mass: float = 0.015,
        v_rel_km_s: float = 7.6,
        latitude_deg: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Analytical Jacchia-Roberts / NRLMSISE approximation.
        Returns:
            log10_density: log10(rho in kg/m^3)
            drag_micro: drag acceleration in micro-m/s^2 (1e-6 m/s^2)
        """
        alt = np.clip(altitude_km, 150.0, 1000.0)

        # Base exponential density profile with altitude-dependent scale height H
        # H approx 50 km at 300 km, expanding to ~80 km at 600 km
        h_scale = 45.0 + 0.06 * (alt - 200.0)
        # Quiet density baseline at 400 km is approx 3e-12 kg/m^3
        base_rho = 3.0e-12 * math.exp(-(alt - 400.0) / h_scale)

        # Space weather enhancement
        # EUV flux effect (F10.7)
        euv_factor = 1.0 + 0.0035 * max(0.0, f10_7 - 70.0)

        # Geomagnetic storm Joule heating (Kp effect)
        # Auroral amplification at polar latitudes (|lat| > 60 deg)
        lat_weight = 1.0 + 0.5 * (math.sin(math.radians(latitude_deg)) ** 2)
        storm_factor = 1.0 + 0.15 * (max(0.0, kp - 2.0) ** 1.6) * lat_weight

        rho = base_rho * euv_factor * storm_factor
        log_rho = math.log10(max(1e-18, rho))

        # Aerodynamic drag equation: a = 0.5 * rho * (Cd * A / m) * v_rel^2
        # Here area_to_mass is Cd * A / m with default Cd ~ 2.2 folded in
        v_rel_m_s = v_rel_km_s * 1000.0
        drag_m_s2 = 0.5 * rho * area_to_mass * (v_rel_m_s ** 2)
        drag_micro = drag_m_s2 * 1e6

        return log_rho, drag_micro

    @classmethod
    def generate_plane_dataset(
        cls,
        plane_id: int,
        altitude_km: float,
        inclination_deg: float,
        num_samples: int = 150,
        noise_std: float = 0.03,
        seed: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates local non-IID telemetry dataset for an orbital plane.
        Features: [altitude_km, Kp, F10.7, area_to_mass, v_rel_km_s]
        Targets: [log10_density, drag_micro]
        """
        rng = np.random.RandomState(seed + plane_id * 100)

        # Altitude variations around plane nominal altitude (+/- 15 km orbital eccentricity)
        alts = altitude_km + rng.uniform(-15.0, 15.0, size=num_samples)

        # Space weather distribution
        # Some planes encounter storms, others quiet periods
        if plane_id % 2 == 1:
            kps = rng.uniform(4.0, 8.5, size=num_samples)     # Storm active
            f107s = rng.uniform(150.0, 260.0, size=num_samples)
        else:
            kps = rng.uniform(1.0, 4.5, size=num_samples)     # Quiet to unsettled
            f107s = rng.uniform(70.0, 140.0, size=num_samples)

        # Ballistic area-to-mass (m^2/kg) variation across satellites
        b_coeffs = rng.uniform(0.010, 0.025, size=num_samples)

        # Orbital velocities (~7.65 km/s at 350 km to ~7.45 km/s at 800 km)
        v_base = math.sqrt(398600.4418 / (EARTH_RADIUS_KM + altitude_km))
        v_rels = v_base + rng.normal(0.0, 0.05, size=num_samples)

        # Latitude variation along orbital inclination
        lats = rng.uniform(-inclination_deg, inclination_deg, size=num_samples)

        x_list = []
        y_list = []

        for i in range(num_samples):
            log_rho, drag_micro = cls.compute_ground_truth_physics(
                altitude_km=alts[i],
                kp=kps[i],
                f10_7=f107s[i],
                area_to_mass=b_coeffs[i],
                v_rel_km_s=v_rels[i],
                latitude_deg=lats[i],
            )

            # Add sensor noise
            log_rho_obs = log_rho + rng.normal(0.0, noise_std)
            drag_obs = max(0.0, drag_micro + rng.normal(0.0, noise_std * drag_micro))

            x_list.append([alts[i], kps[i], f107s[i], b_coeffs[i], v_rels[i]])
            y_list.append([log_rho_obs, drag_obs])

        return np.array(x_list, dtype=np.float64), np.array(y_list, dtype=np.float64)


# ============================================================================
# Privacy Auditor
# ============================================================================

class TelemetryPrivacyAuditor:
    """
    Enforces and verifies zero-telemetry-leakage constraints.
    Ensures that only model parameter updates, gradients, or quantized deltas
    are transmitted over Inter-Satellite Links, while raw time-series telemetry
    (GPS, accelerometry, state vectors) remains strictly local on each satellite.
    """

    @staticmethod
    def audit_update_payload(update: LocalTrainUpdate) -> bool:
        """
        Verify that update object contains strictly parameter tensors and metadata,
        and no raw orbital state telemetry.
        """
        # Ensure parameter deltas exist and match model layer parameter naming
        if not update.parameter_deltas:
            return False

        for k in update.parameter_deltas:
            if not (k.startswith("W") or k.startswith("b")):
                return False

        # Verify no telemetry arrays are secretly attached
        disallowed_fields = ["raw_telemetry", "gps_pos", "gps_vel", "state_vector"]
        for field_name in disallowed_fields:
            if hasattr(update, field_name):
                return False

        return True

    @staticmethod
    def compute_differential_privacy_noise(
        delta_dict: Dict[str, np.ndarray],
        epsilon: float = 1.0,
        delta_dp: float = 1e-5,
        clip_norm: float = 1.0,
        seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        """
        Applies Differential Privacy (DP) Gaussian Mechanism to parameter deltas:
        sigma = sqrt(2 * ln(1.25 / delta_dp)) * clip_norm / epsilon.
        """
        rng = np.random.RandomState(seed)
        sigma = math.sqrt(2.0 * math.log(1.25 / delta_dp)) * clip_norm / epsilon

        noisy_deltas: Dict[str, np.ndarray] = {}
        for k, arr in delta_dict.items():
            noise = rng.normal(0.0, sigma, size=arr.shape)
            noisy_deltas[k] = arr + noise

        return noisy_deltas


# ============================================================================
# Federated Swarm Aggregator & Coordinator
# ============================================================================

class FederatedSwarmCoordinator:
    """
    Central / Cluster-Head Coordinator for Satellite Swarm Federated Learning.
    Orchestrates federated rounds across orbital planes via Inter-Satellite Links (ISL).

    Supported Aggregation Strategies:
    1. FED_AVG: Standard Federated Averaging (McMahan et al.)
    2. FED_PROX: FedProx with proximal regularization for straggler resilience
    3. HIERARCHICAL_FED_AVG: Plane-level intra-plane aggregation -> Constellation inter-plane aggregation
    4. TRIMMED_MEAN: Outlier / radiation SEU corruption rejection
    5. COORDINATE_MEDIAN: Component-wise median
    6. NORM_BOUNDED: L2 update norm clipping before FedAvg
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.FED_AVG,
        mu_prox: float = 0.01,
        trimmed_mean_beta: float = 0.15,
        norm_bound_clip: float = 5.0,
        isl_network: Optional[ISLNetwork] = None,
        seed: int = 42,
    ):
        self.config = model_config or ModelConfig()
        self.strategy = aggregation_strategy
        self.mu_prox = mu_prox
        self.trimmed_mean_beta = trimmed_mean_beta
        self.norm_bound_clip = norm_bound_clip
        self.isl_network = isl_network or ISLNetwork(seed=seed)
        self.rng = np.random.RandomState(seed)

        # Global model maintained by constellation coordinator
        self.global_model = ThermosphericSurrogateMLP(self.config)
        self.registered_nodes: Dict[str, SatelliteClient] = {}
        self.round_history: List[RoundSummary] = []

    def register_satellite(self, node: SatelliteClient) -> None:
        """Register a constellation satellite node."""
        self.registered_nodes[node.satellite_id] = node

    def get_registered_count(self) -> int:
        return len(self.registered_nodes)

    # ------------------------------------------------------------------------
    # Aggregation Implementations
    # ------------------------------------------------------------------------

    @staticmethod
    def aggregate_fed_avg(
        updates: List[LocalTrainUpdate],
    ) -> Dict[str, np.ndarray]:
        """
        Federated Averaging:
        Delta_global = sum_k (n_k / N_total) * Delta_k
        """
        if not updates:
            raise ValueError("No client updates provided for aggregation.")

        total_samples = sum(u.num_samples for u in updates)
        if total_samples == 0:
            total_samples = len(updates)
            weights = [1.0 / len(updates)] * len(updates)
        else:
            weights = [u.num_samples / total_samples for u in updates]

        agg_deltas: Dict[str, np.ndarray] = {}
        param_keys = updates[0].parameter_deltas.keys()

        for k in param_keys:
            stacked = np.zeros_like(updates[0].parameter_deltas[k], dtype=np.float64)
            for w_k, u in zip(weights, updates):
                stacked += w_k * u.parameter_deltas[k]
            agg_deltas[k] = stacked

        return agg_deltas

    def aggregate_trimmed_mean(
        self,
        updates: List[LocalTrainUpdate],
    ) -> Dict[str, np.ndarray]:
        """
        Coordinate-wise Trimmed Mean:
        Trims top and bottom beta fraction of updates per parameter coordinate.
        Protects against single-event upsets (SEU) corrupting neural weights.
        """
        if not updates:
            raise ValueError("No client updates provided for aggregation.")

        k_clients = len(updates)
        trim_count = int(math.floor(k_clients * self.trimmed_mean_beta))

        agg_deltas: Dict[str, np.ndarray] = {}
        param_keys = updates[0].parameter_deltas.keys()

        for k in param_keys:
            # Stack all client updates along axis 0: shape (K, *param_shape)
            tensors = np.stack([u.parameter_deltas[k] for u in updates], axis=0)

            if trim_count > 0 and 2 * trim_count < k_clients:
                # Sort along client dimension
                sorted_tensors = np.sort(tensors, axis=0)
                # Slice trimmed region
                trimmed = sorted_tensors[trim_count : k_clients - trim_count]
                agg_deltas[k] = np.mean(trimmed, axis=0)
            else:
                agg_deltas[k] = np.mean(tensors, axis=0)

        return agg_deltas

    @staticmethod
    def aggregate_coordinate_median(
        updates: List[LocalTrainUpdate],
    ) -> Dict[str, np.ndarray]:
        """Coordinate-wise median aggregation."""
        if not updates:
            raise ValueError("No client updates provided for aggregation.")

        agg_deltas: Dict[str, np.ndarray] = {}
        param_keys = updates[0].parameter_deltas.keys()

        for k in param_keys:
            tensors = np.stack([u.parameter_deltas[k] for u in updates], axis=0)
            agg_deltas[k] = np.median(tensors, axis=0)

        return agg_deltas

    def aggregate_norm_bounded(
        self,
        updates: List[LocalTrainUpdate],
    ) -> Dict[str, np.ndarray]:
        """Clips L2 norm of parameter deltas to norm_bound_clip before FedAvg."""
        clipped_updates = []
        for u in updates:
            u_copy = copy.deepcopy(u)
            if u_copy.delta_norm > self.norm_bound_clip:
                scale = self.norm_bound_clip / max(1e-12, u_copy.delta_norm)
                for k in u_copy.parameter_deltas:
                    u_copy.parameter_deltas[k] *= scale
                u_copy.delta_norm = self.norm_bound_clip
            clipped_updates.append(u_copy)

        return self.aggregate_fed_avg(clipped_updates)

    def aggregate_hierarchical(
        self,
        updates: List[LocalTrainUpdate],
    ) -> Dict[str, np.ndarray]:
        """
        Hierarchical Constellation Aggregation:
        Phase 1: Intra-plane aggregation along orbital rings via high-bandwidth ISL.
        Phase 2: Inter-plane aggregation across Walker plane cluster heads.
        """
        # Group updates by plane_id
        planes: Dict[int, List[LocalTrainUpdate]] = {}
        for u in updates:
            planes.setdefault(u.plane_id, []).append(u)

        plane_aggregated_deltas: Dict[int, Dict[str, np.ndarray]] = {}
        plane_sample_weights: Dict[int, int] = {}

        for p_id, p_updates in planes.items():
            plane_aggregated_deltas[p_id] = self.aggregate_fed_avg(p_updates)
            plane_sample_weights[p_id] = sum(u.num_samples for u in p_updates)

        # Inter-plane aggregation
        total_constellation_samples = sum(plane_sample_weights.values())
        final_deltas: Dict[str, np.ndarray] = {}
        param_keys = updates[0].parameter_deltas.keys()

        for k in param_keys:
            stacked = np.zeros_like(updates[0].parameter_deltas[k], dtype=np.float64)
            for p_id, p_delta in plane_aggregated_deltas.items():
                w_plane = plane_sample_weights[p_id] / max(1, total_constellation_samples)
                stacked += w_plane * p_delta[k]
            final_deltas[k] = stacked

        return final_deltas

    # ------------------------------------------------------------------------
    # Execution of Federated Rounds
    # ------------------------------------------------------------------------

    def execute_round(
        self,
        round_number: int,
        client_fraction: float = 1.0,
        local_epochs: int = 5,
        local_batch_size: int = 16,
        local_lr: Optional[float] = None,
        val_dataset: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> RoundSummary:
        """
        Execute one complete constellation-wide federated learning round:
        1. Select participating satellites.
        2. Distribute global model weights over ISL.
        3. Train local models on private edge telemetry (with FedProx proximal term).
        4. Transmit updates back over ISL (quantized and privacy audited).
        5. Aggregate updates according to configured strategy.
        6. Apply updates to global model and compute validation metrics.
        """
        t0 = time.perf_counter()
        nodes = list(self.registered_nodes.values())
        if not nodes:
            raise ValueError("No satellites registered with coordinator.")

        # Sample participating satellites
        sample_size = max(1, int(round(len(nodes) * np.clip(client_fraction, 0.05, 1.0))))
        selected_nodes: List[SatelliteClient] = self.rng.choice(
            nodes,
            size=sample_size,
            replace=False,
        ).tolist()

        current_global_params = self.global_model.get_parameters()

        # Step 2 & 3: Local Edge Training
        mu = self.mu_prox if self.strategy == AggregationStrategy.FED_PROX else 0.0
        raw_updates: List[LocalTrainUpdate] = []
        straggler_count = 0

        for node in selected_nodes:
            update = node.train_local(
                global_params=current_global_params,
                epochs=local_epochs,
                batch_size=local_batch_size,
                learning_rate=local_lr,
                mu_prox=mu,
            )
            # Step 4: Privacy Audit
            if not TelemetryPrivacyAuditor.audit_update_payload(update):
                raise RuntimeError(
                    f"Privacy violation: satellite {node.satellite_id} attempted "
                    "to transmit unverified telemetry payload!"
                )

            if update.is_straggler:
                straggler_count += 1
            raw_updates.append(update)

        # Step 4: Transmit updates over ISL (simulating packet drops & quantization)
        coordinator_plane = 0
        transmitted_updates: List[LocalTrainUpdate] = []

        for u in raw_updates:
            received_deltas = self.isl_network.transmit_update(u, dest_plane_id=coordinator_plane)
            if received_deltas is not None:
                # Update with received (possibly quantized/dequantized) deltas
                u.parameter_deltas = received_deltas
                transmitted_updates.append(u)

        if not transmitted_updates:
            # All ISL packets dropped; keep global model unchanged
            return RoundSummary(
                round_number=round_number,
                global_train_loss=0.0,
                global_val_loss=0.0,
                participating_nodes=0,
                straggler_nodes=straggler_count,
                strategy_used=self.strategy,
                mean_delta_norm=0.0,
                total_isl_bytes=0,
                compression_ratio=1.0,
                round_duration_s=time.perf_counter() - t0,
            )

        # Step 5: Aggregation
        if self.strategy == AggregationStrategy.FED_PROX or self.strategy == AggregationStrategy.FED_AVG:
            global_deltas = self.aggregate_fed_avg(transmitted_updates)
        elif self.strategy == AggregationStrategy.HIERARCHICAL_FED_AVG:
            global_deltas = self.aggregate_hierarchical(transmitted_updates)
        elif self.strategy == AggregationStrategy.TRIMMED_MEAN:
            global_deltas = self.aggregate_trimmed_mean(transmitted_updates)
        elif self.strategy == AggregationStrategy.COORDINATE_MEDIAN:
            global_deltas = self.aggregate_coordinate_median(transmitted_updates)
        elif self.strategy == AggregationStrategy.NORM_BOUNDED:
            global_deltas = self.aggregate_norm_bounded(transmitted_updates)
        else:
            global_deltas = self.aggregate_fed_avg(transmitted_updates)

        # Step 6: Apply aggregated updates to Global Model: w^(t+1) = w^t + Delta_global
        new_params = {}
        for k in self.global_model.parameters:
            new_params[k] = current_global_params[k] + global_deltas[k]
        self.global_model.set_parameters(new_params)

        # Step 7: Metrics & Validation
        round_duration = time.perf_counter() - t0
        mean_delta_norm = float(np.mean([u.delta_norm for u in transmitted_updates]))
        mean_train_loss = float(np.mean([u.train_loss_final for u in transmitted_updates]))

        val_loss = 0.0
        plane_losses = {}
        if val_dataset is not None:
            x_val, y_val = val_dataset
            eval_res = self.global_model.evaluate(x_val, y_val)
            val_loss = eval_res["mse_norm"]

        # Calculate compression ratio
        raw_b = max(1, self.isl_network.total_raw_bytes)
        tx_b = max(1, self.isl_network.total_transmitted_bytes)
        comp_ratio = float(raw_b / tx_b)

        summary = RoundSummary(
            round_number=round_number,
            global_train_loss=mean_train_loss,
            global_val_loss=val_loss,
            participating_nodes=len(transmitted_updates),
            straggler_nodes=straggler_count,
            strategy_used=self.strategy,
            mean_delta_norm=mean_delta_norm,
            total_isl_bytes=self.isl_network.total_transmitted_bytes,
            compression_ratio=comp_ratio,
            round_duration_s=round_duration,
            plane_losses=plane_losses,
        )
        self.round_history.append(summary)
        return summary

    def train_federated(
        self,
        num_rounds: int = 5,
        client_fraction: float = 1.0,
        local_epochs: int = 5,
        local_batch_size: int = 16,
        local_lr: Optional[float] = None,
        val_dataset: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> List[RoundSummary]:
        """Execute multiple federated learning rounds."""
        summaries = []
        for r in range(1, num_rounds + 1):
            s = self.execute_round(
                round_number=r,
                client_fraction=client_fraction,
                local_epochs=local_epochs,
                local_batch_size=local_batch_size,
                local_lr=local_lr,
                val_dataset=val_dataset,
            )
            summaries.append(s)
        return summaries


# ============================================================================
# Constellation Factory & Utility Functions
# ============================================================================

def build_walker_constellation_federation(
    num_planes: int = 4,
    sats_per_plane: int = 4,
    base_altitude_km: float = 400.0,
    altitude_spacing_km: float = 100.0,
    straggler_fraction: float = 0.25,
    samples_per_sat: int = 120,
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FED_PROX,
    quantization_mode: QuantizationMode = QuantizationMode.NONE,
    mu_prox: float = 0.02,
    seed: int = 42,
) -> Tuple[FederatedSwarmCoordinator, List[SatelliteClient], Tuple[np.ndarray, np.ndarray]]:
    """
    Factory helper to initialize a complete multi-plane constellation swarm,
    generate non-IID orbital datasets for each satellite, and return the coordinator,
    client list, and constellation-wide validation dataset.
    """
    rng = np.random.RandomState(seed)

    isl_net = ISLNetwork(
        quantization_mode=quantization_mode,
        seed=seed,
    )
    coordinator = FederatedSwarmCoordinator(
        aggregation_strategy=aggregation_strategy,
        mu_prox=mu_prox,
        isl_network=isl_net,
        seed=seed,
    )

    clients: List[SatelliteClient] = []
    val_x_list: List[np.ndarray] = []
    val_y_list: List[np.ndarray] = []

    for p in range(num_planes):
        plane_alt = base_altitude_km + p * altitude_spacing_km
        # Inclination varying: plane 0 (45 deg), plane 1 (53 deg ISS/Starlink), plane 2 (70 deg), plane 3 (85 deg polar)
        plane_inc = 45.0 + p * 13.0

        for s in range(sats_per_plane):
            sat_id = f"SAT-P{p:02d}-S{s:02d}"
            # Randomly designate stragglers based on straggler_fraction
            is_straggler = bool(rng.rand() < straggler_fraction)
            p_state = (
                NodePowerState.ECLIPSE_BATTERY_SAVING
                if is_straggler
                else NodePowerState.NOMINAL_SUNLIT
            )

            client = SatelliteClient(
                satellite_id=sat_id,
                plane_id=p,
                slot_id=s,
                altitude_km=plane_alt,
                inclination_deg=plane_inc,
                is_straggler=is_straggler,
                straggler_work_ratio=0.35 if is_straggler else 1.0,
                power_state=p_state,
            )

            # Generate local non-IID training telemetry
            x_train, y_train = ConstellationDataGenerator.generate_plane_dataset(
                plane_id=p,
                altitude_km=plane_alt,
                inclination_deg=plane_inc,
                num_samples=samples_per_sat,
                seed=seed + p * 10 + s,
            )
            client.set_dataset(x_train, y_train)

            coordinator.register_satellite(client)
            clients.append(client)

        # Also generate plane slice for global validation set
        x_val_p, y_val_p = ConstellationDataGenerator.generate_plane_dataset(
            plane_id=p,
            altitude_km=plane_alt,
            inclination_deg=plane_inc,
            num_samples=50,
            seed=seed + 999 + p,
        )
        val_x_list.append(x_val_p)
        val_y_list.append(y_val_p)

    val_dataset = (
        np.concatenate(val_x_list, axis=0),
        np.concatenate(val_y_list, axis=0),
    )

    return coordinator, clients, val_dataset


def compare_fedavg_vs_fedprox(
    num_rounds: int = 6,
    num_planes: int = 3,
    sats_per_plane: int = 4,
    straggler_fraction: float = 0.50,
    mu_prox: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Empirical benchmark comparing FedAvg vs FedProx under extreme straggler conditions
    (50% of satellites constrained to partial local epochs).
    Demonstrates the stabilizing effect of the proximal regularization term.
    """
    # 1. FedAvg Run
    coord_avg, _, val_data = build_walker_constellation_federation(
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        straggler_fraction=straggler_fraction,
        aggregation_strategy=AggregationStrategy.FED_AVG,
        mu_prox=0.0,
        seed=seed,
    )
    history_avg = coord_avg.train_federated(
        num_rounds=num_rounds,
        local_epochs=6,
        local_batch_size=16,
        val_dataset=val_data,
    )
    final_avg_eval = coord_avg.global_model.evaluate(val_data[0], val_data[1])

    # 2. FedProx Run
    coord_prox, _, _ = build_walker_constellation_federation(
        num_planes=num_planes,
        sats_per_plane=sats_per_plane,
        straggler_fraction=straggler_fraction,
        aggregation_strategy=AggregationStrategy.FED_PROX,
        mu_prox=mu_prox,
        seed=seed,
    )
    history_prox = coord_prox.train_federated(
        num_rounds=num_rounds,
        local_epochs=6,
        local_batch_size=16,
        val_dataset=val_data,
    )
    final_prox_eval = coord_prox.global_model.evaluate(val_data[0], val_data[1])

    return {
        "straggler_fraction": straggler_fraction,
        "fedavg_history": history_avg,
        "fedprox_history": history_prox,
        "fedavg_final_val_loss": final_avg_eval["mse_norm"],
        "fedprox_final_val_loss": final_prox_eval["mse_norm"],
        "fedavg_mae_density": final_avg_eval["mae_log_density"],
        "fedprox_mae_density": final_prox_eval["mae_log_density"],
        "fedavg_mae_drag": final_avg_eval["mae_drag_micro_m_s2"],
        "fedprox_mae_drag": final_prox_eval["mae_drag_micro_m_s2"],
    }
