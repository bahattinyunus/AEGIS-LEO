import pytest
import numpy as np
from datetime import datetime, timezone

from aegis_leo.data.ccsds_ephemeris import generate_ccsds_oem, generate_ccsds_opm
from aegis_leo.physics.maneuver_splicer import ManeuverSplicer
from aegis_leo.risk.monte_carlo import MonteCarloConjunctionSimulator
from aegis_leo.risk.kessler_cascade import nasa_breakup_model, simulate_cascade_progression
from aegis_leo.cli import cmd_status, cmd_eval, cmd_burn, cmd_monte_carlo, cmd_breakup
import argparse


def test_ccsds_oem_export():
    positions = np.array([[6800.0, 0.0, 0.0], [6790.0, 100.0, 0.0]])
    velocities = np.array([[0.0, 7.6, 0.0], [-0.1, 7.59, 0.0]])
    times = np.array([0.0, 60.0])

    oem_text = generate_ccsds_oem(
        object_name="STARLINK-1007",
        object_id="44713",
        positions_km=positions,
        velocities_km_s=velocities,
        time_seconds=times,
    )

    assert "CCSDS_OEM_VERS = 2.0" in oem_text
    assert "OBJECT_NAME          = STARLINK-1007" in oem_text
    assert "REF_FRAME            = EME2000" in oem_text
    assert len(oem_text.splitlines()) > 10


def test_ccsds_opm_export():
    r = np.array([6878.137, 0.0, 0.0])
    v = np.array([0.0, 7.6, 0.0])
    opm_text = generate_ccsds_opm(
        object_name="AEGIS-SAT",
        object_id="12345",
        epoch=datetime.now(timezone.utc),
        r_eci_km=r,
        v_eci_km_s=v,
        mass_kg=260.0,
        cd=2.2,
        drag_area_m2=2.2,
    )

    assert "CCSDS_OPM_VERS = 2.0" in opm_text
    assert "MASS                 =     260.00" in opm_text


def test_maneuver_splicer():
    splicer = ManeuverSplicer(wet_mass_kg=260.0, thrust_newtons=1.0, isp_seconds=250.0)
    r0 = np.array([6878.137, 0.0, 0.0])
    v0 = np.array([0.0, 7.6, 0.0])
    burn_dir = np.array([0.0, 1.0, 0.0])

    res = splicer.simulate_powered_burn(
        r0=r0, v0=v0, burn_duration_seconds=10.0, burn_direction_unit=burn_dir, dt_seconds=1.0
    )

    assert res["delta_v_delivered_m_s"] > 0.0
    assert res["propellant_spent_kg"] > 0.0
    assert res["final_mass_kg"] < 260.0


def test_monte_carlo_simulator():
    mc = MonteCarloConjunctionSimulator()
    res = mc.run_simulation(nominal_miss_km=0.30, num_trials=500)

    assert res["num_trials"] == 500
    assert "empirical_pc" in res
    assert "analytical_pc" in res
    assert res["min_simulated_miss_km"] <= res["mean_simulated_miss_km"]


def test_nasa_breakup_and_cascade():
    brk = nasa_breakup_model(target_mass_kg=260.0, projectile_mass_kg=5.0)
    assert brk["fragments_gt_10cm"] > 10
    assert brk["fragments_gt_1cm"] > brk["fragments_gt_10cm"]

    cas = simulate_cascade_progression(initial_active_satellites=1000, initial_debris_objects=5000, years=5)
    assert len(cas["annual_debris_history"]) == 6
    assert cas["total_estimated_collisions"] >= 0.0


def test_cli_subcommands(capsys):
    # status
    cmd_status(argparse.Namespace())
    out_status = capsys.readouterr().out
    assert "AEGIS-LEO SSA SHIELD" in out_status

    # eval
    cmd_eval(argparse.Namespace(sat="STARLINK_LEO", miss=0.45, kp=8.0, f107=240.0))
    out_eval = capsys.readouterr().out
    assert "Conjunction Risk Evaluation" in out_eval

    # burn
    cmd_burn(argparse.Namespace(sat="STARLINK_LEO", miss=0.45, kp=8.0))
    out_burn = capsys.readouterr().out
    assert "Autonomous Evasion Plan" in out_burn

    # monte carlo
    cmd_monte_carlo(argparse.Namespace(miss=0.45, trials=100))
    out_mc = capsys.readouterr().out
    assert "Monte Carlo Conjunction Dispersion" in out_mc

    # breakup
    cmd_breakup(argparse.Namespace(target_mass=260.0, proj_mass=5.0))
    out_brk = capsys.readouterr().out
    assert "NASA Standard Breakup Model" in out_brk
