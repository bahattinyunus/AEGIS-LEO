"""
Unit tests for 5-Node Spacecraft Orbital Thermal Control & Radiative Equilibrium Engine.
"""

from aegis_leo.physics.thermal_analyzer import simulate_spacecraft_thermal_balance


def test_nominal_thermal_balance_all_nodes_within_limits():
    rep = simulate_spacecraft_thermal_balance(
        altitude_km=550.0,
        beta_angle_deg=22.0,
        kp_storm=5.0,
        avionics_load_w=85.0,
        burn_active=True,
    )
    assert rep.all_nodes_within_limits is True
    assert rep.thermal_status == "NOMINAL_THERMAL_EQUILIBRIUM"
    assert len(rep.node_summaries) == 5
    assert "EDGE_AI_AVIONICS" in rep.node_summaries
    assert len(rep.timeline) == 72
    assert rep.mean_solar_absorbed_w > 100.0
    assert rep.mean_earth_ir_absorbed_w > 10.0


def test_storm_aero_heating_increases_at_lower_altitude_and_high_kp():
    rep_quiet = simulate_spacecraft_thermal_balance(altitude_km=260.0, kp_storm=1.0, f10_7=75.0)
    rep_storm = simulate_spacecraft_thermal_balance(altitude_km=260.0, kp_storm=8.8, f10_7=270.0)
    assert rep_storm.peak_aero_heating_w > rep_quiet.peak_aero_heating_w


def test_full_sun_high_beta_orbit_thermal():
    rep = simulate_spacecraft_thermal_balance(altitude_km=600.0, beta_angle_deg=78.0, kp_storm=3.0)
    assert all(step["sun_factor"] == 1.0 for step in rep.timeline)
    d = rep.to_dict()
    assert "node_summaries" in d
