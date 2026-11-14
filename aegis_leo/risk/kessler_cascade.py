"""
NASA Standard Breakup Model & Kessler Syndrome Cascade Simulator
Models fragment size distribution N(Lc) and multi-year orbital debris cascade triggers.
"""

from typing import Dict, Any, List
import numpy as np


def nasa_breakup_model(
    target_mass_kg: float,
    projectile_mass_kg: float,
    is_catastrophic: bool = True,
    characteristic_length_min_m: float = 0.10,
) -> Dict[str, Any]:
    """
    NASA Standard Satellite Breakup Model (Johnson, Krisko, Liou, Herbert 2001).
    For catastrophic collisions:
    N(Lc >= L) = 0.1 * (M_total)^0.75 * L^(-1.71)
    """
    m_total = target_mass_kg + projectile_mass_kg
    lc = max(0.01, characteristic_length_min_m)

    if is_catastrophic:
        num_fragments = 0.1 * (m_total ** 0.75) * (lc ** (-1.71))
    else:
        # Non-catastrophic cratering
        num_fragments = 0.1 * ((projectile_mass_kg * 7.0) ** 0.75) * (lc ** (-1.71))

    num_trackable_10cm = int(round(num_fragments))

    # Fragments smaller than 10cm down to 1cm
    num_lethal_1cm = int(round(0.1 * (m_total ** 0.75) * (0.01 ** (-1.71))))

    return {
        "target_mass_kg": target_mass_kg,
        "projectile_mass_kg": projectile_mass_kg,
        "total_mass_kg": m_total,
        "collision_type": "CATASTROPHIC" if is_catastrophic else "NON_CATASTROPHIC",
        "fragments_gt_10cm": num_trackable_10cm,
        "fragments_gt_1cm": num_lethal_1cm,
        "characteristic_length_cutoff_m": lc,
    }


def simulate_cascade_progression(
    initial_active_satellites: int = 5000,
    initial_debris_objects: int = 25000,
    annual_launch_rate: int = 1200,
    years: int = 20,
    mitigation_compliance_pct: float = 85.0,
    solar_cycle_drag_cleansing: bool = True,
) -> Dict[str, Any]:
    """
    Multi-year simplified box model for LEO debris growth and carrying capacity.
    """
    active_history = [initial_active_satellites]
    debris_history = [initial_debris_objects]
    collisions_history = [0]

    curr_active = float(initial_active_satellites)
    curr_debris = float(initial_debris_objects)

    for yr in range(1, years + 1):
        # 11-year solar cycle drag cleansing modulation
        solar_phase = np.sin((yr / 11.0) * 2.0 * np.pi)
        decay_factor = 0.04 + (0.03 * solar_phase if solar_cycle_drag_cleansing else 0.0)

        # Expected collision rate ~ k * N_sat * N_debris
        k_cross = 1.2e-8
        expected_collisions = k_cross * curr_active * curr_debris

        # Breakup fragments added per collision (avg 350 kg sat)
        fragments_per_collision = 450.0

        # Updates
        debris_added = expected_collisions * fragments_per_collision
        debris_decayed = curr_debris * max(0.01, decay_factor)

        curr_active += annual_launch_rate - (curr_active * 0.12)  # 8-year replacement lifecycle
        curr_debris += debris_added - debris_decayed

        active_history.append(int(curr_active))
        debris_history.append(int(curr_debris))
        collisions_history.append(round(float(expected_collisions), 2))

    return {
        "simulation_years": years,
        "initial_debris": initial_debris_objects,
        "final_debris": debris_history[-1],
        "total_estimated_collisions": round(sum(collisions_history), 1),
        "debris_growth_rate_pct": round(((debris_history[-1] - initial_debris_objects) / initial_debris_objects) * 100.0, 1),
        "annual_debris_history": debris_history,
        "annual_collisions_history": collisions_history,
        "kessler_runaway_warning": debris_history[-1] > (initial_debris_objects * 1.5),
    }
