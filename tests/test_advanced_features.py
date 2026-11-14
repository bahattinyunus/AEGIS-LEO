import pytest
import numpy as np
import torch
import argparse

from aegis_leo.data.celestrak_live import CelesTrakClient
from aegis_leo.ai.radiation_fault_simulator import RadiationFaultSimulator
from aegis_leo.ai.pinn import get_pretrained_edge_pinn
from aegis_leo.physics.deorbit_simulation import compute_deorbit_burn, simulate_reentry_corridor
from aegis_leo.risk.covariance_visualizer import generate_covariance_ellipsoid_mesh
from aegis_leo.cli import (
    cmd_deorbit,
    cmd_rad_test,
    cmd_celestrak,
    cmd_lifetime,
    cmd_tether,
    cmd_laser,
    cmd_consensus,
)


def test_celestrak_client_fetch():
    client = CelesTrakClient(cache_dir=".test_cache_tle")
    res = client.fetch_tle_by_catnr(25544)

    assert res["norad_catnr"] == 25544
    assert "line1" in res and "line2" in res
    assert len(res["line1"]) > 50


def test_radiation_flux_and_tmr_voting():
    model = get_pretrained_edge_pinn()
    sim = RadiationFaultSimulator(model)

    flux = sim.calculate_space_radiation_flux(altitude_km=550.0, latitude_deg=-25.0, is_solar_particle_event=True)
    assert flux["total_flux_p_cm2_s"] > 0.0
    assert flux["spe_active"] is True

    inp = torch.tensor([[550.0, 8.0, 240.0, 0.02, 7.6]], dtype=torch.float32)
    tmr = sim.evaluate_tmr_voting(inp, bit_flip_count=4)

    assert tmr["fault_masked_successfully"] is True
    assert tmr["resilience_percentage"] == 100.0


def test_deorbit_simulation_and_heating():
    burn = compute_deorbit_burn(initial_altitude_km=550.0, target_perigee_km=60.0)
    assert burn["delta_v_retrograde_m_s"] > 100.0
    assert burn["propellant_required_kg"] > 0.0

    descent = simulate_reentry_corridor(entry_altitude_km=120.0, entry_speed_km_s=7.6)
    assert descent["descent_time_seconds"] > 0.0
    assert descent["peak_stagnation_heat_flux_w_cm2"] >= 0.0
    assert descent["nasa_req_4_7_compliant"] is True


def test_covariance_ellipsoid_generator():
    center = np.array([6800.0, 0.0, 0.0])
    mesh = generate_covariance_ellipsoid_mesh(center, sigma_radial_km=0.2, sigma_intrack_km=0.8, sigma_crosstrack_km=0.3)

    assert len(mesh["x"]) == 25
    assert mesh["uncertainty_volume_km3"] > 0.0
    assert mesh["sigma_level"] == 2.0


def test_advanced_cli_commands(capsys):
    cmd_deorbit(argparse.Namespace(alt=550.0, mass=260.0))
    out_deo = capsys.readouterr().out
    assert "Controlled Deorbit" in out_deo

    cmd_rad_test(argparse.Namespace(alt=550.0, spe=False, flips=2))
    out_rad = capsys.readouterr().out
    assert "Space Radiation" in out_rad

    cmd_celestrak(argparse.Namespace(norad=25544))
    out_cel = capsys.readouterr().out
    assert "CelesTrak Live TLE" in out_cel

    cmd_lifetime(argparse.Namespace(alt=400.0, mass=200.0, area=2.0, cd=2.2))
    out_life = capsys.readouterr().out
    assert "ORBIT LIFETIME" in out_life

    cmd_tether(argparse.Namespace(mass=250.0, alt=750.0))
    out_teth = capsys.readouterr().out
    assert "ELECTRODYNAMIC TETHER" in out_teth

    cmd_laser(argparse.Namespace(mass=1.2, dia=12.0, perigee=450.0, apogee=750.0, power=10.0, duration=45.0, space=False))
    out_las = capsys.readouterr().out
    assert "PULSED LASER" in out_las

    cmd_consensus(argparse.Namespace(sat_a="SAT-WALKER-A1", sat_b="SAT-WALKER-A2", miss=45.0, tca=1800.0, dv=0.45))
    out_con = capsys.readouterr().out
    assert "SWARM MANEUVER QUORUM" in out_con
