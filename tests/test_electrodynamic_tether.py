"""
Tests for Electrodynamic Tether (EDT) Deorbit Propulsion Physics.
"""

import pytest
from aegis_leo.adr.electrodynamic_tether import (
    ElectrodynamicTetherSpecs,
    compute_geomagnetic_field_b,
    compute_plasma_density,
    simulate_electrodynamic_tether_decay,
    compare_all_deorbit_methods,
    DeorbitMethod,
)


def test_tether_specs_properties():
    specs = ElectrodynamicTetherSpecs(length_m=2000.0, wire_radius_mm=0.5)
    assert specs.tether_mass_kg > 1.0
    assert specs.ohmic_resistance_ohms > 10.0


def test_geomagnetic_field_calculation():
    br, btheta, btot = compute_geomagnetic_field_b(altitude_km=500.0)
    assert 2.0e-5 < btot < 6.0e-5
    assert btot > 0.0


def test_plasma_density_peaks():
    ne_200 = compute_plasma_density(200.0)
    ne_350 = compute_plasma_density(350.0)
    ne_800 = compute_plasma_density(800.0)

    # F2 peak around 350 km should exceed lower/higher regimes
    assert ne_350 > ne_200
    assert ne_350 > ne_800


def test_simulate_electrodynamic_tether_decay():
    res = simulate_electrodynamic_tether_decay(
        satellite_mass_kg=200.0,
        initial_altitude_km=700.0,
        target_altitude_km=300.0,
        dt_days=5.0,
    )
    assert res.final_altitude_km <= 305.0
    assert res.deorbit_time_days < 500.0
    assert res.average_lorentz_drag_force_mn > 0.1
    assert res.peak_induced_emf_volts > 50.0
    assert res.propellant_saved_kg > 5.0

    d = res.to_dict()
    assert d["status"] == "TETHER_DEORBIT_COMPLETE"
    assert "average_current_amps" in d


def test_compare_all_deorbit_methods():
    comp = compare_all_deorbit_methods(satellite_mass_kg=300.0, initial_altitude_km=750.0)
    assert "methods" in comp
    methods = comp["methods"]
    assert DeorbitMethod.ELECTRODYNAMIC_TETHER.value in methods
    assert DeorbitMethod.CHEMICAL_RETRO_PROPULSION.value in methods
    assert DeorbitMethod.DEPLOYABLE_DRAG_SAIL.value in methods

    # Tether should require 0 propellant
    assert methods[DeorbitMethod.ELECTRODYNAMIC_TETHER.value]["propellant_mass_kg"] == 0.0
    # Chemical should require significant propellant
    assert methods[DeorbitMethod.CHEMICAL_RETRO_PROPULSION.value]["propellant_mass_kg"] > 10.0
