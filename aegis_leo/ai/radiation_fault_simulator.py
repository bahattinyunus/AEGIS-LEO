"""
Radiation Environment & Single Event Upset (SEU) Fault Injection Simulator
Models space radiation (South Atlantic Anomaly & Solar Particle Events)
and demonstrates Triple Modular Redundancy (TMR) fault masking on edge neural weights.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from aegis_leo.ai.pinn import ThermosphereDragPINN, get_pretrained_edge_pinn


class RadiationFaultSimulator:
    """
    Simulates cosmic ray & solar proton induced bit-flips on neural network weights
    and evaluates Triple Modular Redundancy (TMR) resilience.
    """

    def __init__(self, model: ThermosphereDragPINN):
        self.model = model

    @staticmethod
    def calculate_space_radiation_flux(
        altitude_km: float,
        latitude_deg: float,
        is_solar_particle_event: bool = False,
    ) -> Dict[str, float]:
        """
        Estimate trapped proton and heavy ion flux (particles / cm^2 / s).
        Models South Atlantic Anomaly (SAA) peak flux between -10 and -40 deg latitude.
        """
        alt = np.clip(altitude_km, 200.0, 1000.0)

        # Baseline cosmic background flux
        base_flux = 2.5 * (alt / 500.0)

        # South Atlantic Anomaly (SAA) enhancement (lat: -30 to -15, lon: -60 to 0)
        in_saa = (-45.0 <= latitude_deg <= -10.0)
        saa_boost = 1800.0 * (alt / 500.0)**1.8 if in_saa else 1.0

        # Solar Particle Event (SPE) storm enhancement
        spe_boost = 5000.0 if is_solar_particle_event else 1.0

        total_flux = base_flux * max(1.0, saa_boost * 0.2 + (spe_boost if is_solar_particle_event else 0.0))

        # Expected SEU bit-flip rate per Megabit per day
        seu_rate_per_day = float(total_flux * 1.5e-4)

        return {
            "altitude_km": alt,
            "latitude_deg": latitude_deg,
            "total_flux_p_cm2_s": round(float(total_flux), 1),
            "in_saa_zone": bool(in_saa),
            "spe_active": bool(is_solar_particle_event),
            "estimated_seu_rate_per_mb_day": round(seu_rate_per_day, 3),
        }

    def inject_seu_bit_flips(
        self,
        bit_flip_count: int = 5,
        seed: int = 42,
    ) -> ThermosphereDragPINN:
        """
        Inject random bit-flips / perturbations into network weight tensors.
        """
        np.random.seed(seed)
        corrupted_model = get_pretrained_edge_pinn()
        corrupted_model.load_state_dict(self.model.state_dict())

        # Select weights to perturb
        params = list(corrupted_model.parameters())
        for _ in range(bit_flip_count):
            p = params[np.random.randint(0, len(params))]
            with torch.no_grad():
                flat = p.view(-1)
                idx = np.random.randint(0, flat.numel())
                # Perturb weight value (simulating corrupted exponent/mantissa bit)
                perturbation = float(np.random.choice([-10.0, -5.0, 5.0, 10.0]))
                flat[idx] += perturbation

        return corrupted_model

    def evaluate_tmr_voting(
        self,
        input_data: torch.Tensor,
        bit_flip_count: int = 4,
    ) -> Dict[str, Any]:
        """
        Execute Triple Modular Redundancy (TMR) majority voting.
        Core 1: Nominal
        Core 2: Perturbed with SEU
        Core 3: Nominal
        Compares single corrupted core vs TMR majority voted output.
        """
        nominal_model = self.model
        faulty_model = self.inject_seu_bit_flips(bit_flip_count=bit_flip_count)

        with torch.no_grad():
            out_nominal = nominal_model(input_data).numpy()
            out_faulty = faulty_model(input_data).numpy()
            out_redundant = nominal_model(input_data).numpy()

            # TMR Median / Majority Voting across 3 cores
            stacked = np.stack([out_nominal, out_faulty, out_redundant], axis=0)
            tmr_voted = np.median(stacked, axis=0)

            # Error without TMR vs with TMR
            raw_fault_error = float(np.mean(np.abs(out_faulty - out_nominal)))
            tmr_error = float(np.mean(np.abs(tmr_voted - out_nominal)))

            fault_masked = np.isclose(tmr_error, 0.0, atol=1e-5)

            return {
                "injected_bit_flips": bit_flip_count,
                "single_core_seu_error": round(raw_fault_error, 4),
                "tmr_voted_error": round(tmr_error, 6),
                "fault_masked_successfully": bool(fault_masked),
                "resilience_percentage": 100.0 if fault_masked else 0.0,
                "architecture": "Hardware TMR (Triple Modular Redundancy) Voting",
            }
