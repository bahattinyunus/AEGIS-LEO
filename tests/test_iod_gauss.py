"""
Unit and integration tests for AEGIS-LEO Optical Initial Orbit Determination (IOD).
Tests Gauss IOD, Laplace IOD, Gibbs, Herrick-Gibbs, and Synthetic Observation generation.
"""

import pytest
import numpy as np

from aegis_leo.constants import (
    EARTH_MU,
    EARTH_RADIUS_KM,
    EARTH_ROTATION_RATE_RAD_S,
    SPEED_OF_LIGHT,
)
from aegis_leo.physics.coordinates import kepler_to_cartesian, cartesian_to_kepler, eci_to_geodetic
from aegis_leo.navigation.iod_gauss import (
    OpticalObservation,
    IODResult,
    ra_dec_to_los,
    los_to_ra_dec,
    ground_station_eci,
    apparent_ra_dec,
    angular_separation_deg,
    compute_optical_residuals,
    stumpff_c,
    stumpff_s,
    kepler_universal_propagate,
    gibbs_iod,
    herrick_gibbs_iod,
    three_position_iod,
    gauss_iod,
    laplace_iod,
    initial_orbit_determination,
    generate_synthetic_optical_observations,
    batch_optical_screening,
)


class TestOpticalGeometry:
    """Tests for optical angles, line-of-sight conversions, and ground stations."""

    def test_ra_dec_to_los_and_inverse(self):
        """Verify LOS unit vector conversion and round-trip consistency across quadrants."""
        test_angles = [
            (0.0, 0.0),
            (90.0, 0.0),
            (180.0, 0.0),
            (270.0, 0.0),
            (45.0, 60.0),
            (135.0, -45.0),
            (225.0, 30.0),
            (315.0, -60.0),
            (0.0, 90.0),   # North pole
            (0.0, -90.0),  # South pole
        ]
        for ra, dec in test_angles:
            los = ra_dec_to_los(ra, dec)
            assert np.isclose(np.linalg.norm(los), 1.0, atol=1e-12)
            ra_rec, dec_rec = los_to_ra_dec(los)
            assert np.isclose(dec_rec, dec, atol=1e-6)
            if abs(dec) < 89.9:
                assert np.isclose(ra_rec, ra, atol=1e-6)

    def test_optical_observation_dataclass(self):
        """Verify OpticalObservation validation, properties, and dictionary serialization."""
        obs = OpticalObservation(
            timestamp_s=100.0,
            ra_deg=45.0,
            dec_deg=30.0,
            site_eci_km=[6378.0, 0.0, 0.0],
        )
        assert obs.timestamp_s == 100.0
        assert np.isclose(obs.ra_rad, np.radians(45.0))
        assert np.isclose(obs.dec_rad, np.radians(30.0))
        assert obs.site_eci_km.shape == (3,)
        assert np.isclose(np.linalg.norm(obs.los_vector), 1.0)

        # Invalid declination
        with pytest.raises(ValueError):
            OpticalObservation(0.0, 10.0, 95.0, [0, 0, 0])

        d = obs.to_dict()
        assert d["timestamp_s"] == 100.0
        assert "los_vector" in d

    def test_ground_station_eci_coordinates(self):
        """Verify ground station ECI position at equator and poles with GMST."""
        # Equatorial site at prime meridian with GMST = 0
        r_eq0 = ground_station_eci(lat_deg=0.0, lon_deg=0.0, alt_km=0.0, gmst_rad=0.0)
        assert np.isclose(r_eq0[0], EARTH_RADIUS_KM, atol=1e-3)
        assert np.isclose(r_eq0[1], 0.0, atol=1e-3)
        assert np.isclose(r_eq0[2], 0.0, atol=1e-3)

        # Equatorial site with GMST = 90 deg -> rotated into Y axis
        r_eq90 = ground_station_eci(lat_deg=0.0, lon_deg=0.0, alt_km=0.0, gmst_rad=np.pi / 2.0)
        assert np.isclose(r_eq90[0], 0.0, atol=1e-3)
        assert np.isclose(r_eq90[1], EARTH_RADIUS_KM, atol=1e-3)

        # North pole site (lat=90)
        r_np = ground_station_eci(lat_deg=90.0, lon_deg=0.0, alt_km=10.0, gmst_rad=0.5)
        assert np.isclose(r_np[0], 0.0, atol=1e-3)
        assert np.isclose(r_np[1], 0.0, atol=1e-3)
        assert r_np[2] > 6350.0  # Polar radius + 10 km

    def test_apparent_ra_dec_and_residuals(self):
        """Verify apparent RA/Dec calculation and optical angular residuals."""
        r_site = np.array([EARTH_RADIUS_KM, 0.0, 0.0])
        r_sat = np.array([EARTH_RADIUS_KM + 500.0, 0.0, 0.0])

        ra, dec, slant_range = apparent_ra_dec(r_sat, r_site)
        assert np.isclose(slant_range, 500.0)
        assert np.isclose(ra, 0.0, atol=1e-6)
        assert np.isclose(dec, 0.0, atol=1e-6)

        obs = OpticalObservation(0.0, ra_deg=ra, dec_deg=dec, site_eci_km=r_site)
        d_ra, d_dec, tot_res = compute_optical_residuals(r_sat, obs)
        assert np.isclose(tot_res, 0.0, atol=1e-6)

    def test_angular_separation(self):
        """Verify great-circle angular separation between LOS vectors."""
        u1 = np.array([1.0, 0.0, 0.0])
        u2 = np.array([0.0, 1.0, 0.0])
        u3 = np.array([-1.0, 0.0, 0.0])
        assert np.isclose(angular_separation_deg(u1, u2), 90.0)
        assert np.isclose(angular_separation_deg(u1, u3), 180.0)
        assert np.isclose(angular_separation_deg(u1, u1), 0.0)


class TestUniversalKepler:
    """Tests for universal Kepler variable propagator and Stumpff functions."""

    def test_stumpff_functions(self):
        """Verify Stumpff C(z) and S(z) functions across z=0, positive, and negative."""
        # Near zero
        assert np.isclose(stumpff_c(0.0), 0.5, atol=1e-6)
        assert np.isclose(stumpff_s(0.0), 1.0 / 6.0, atol=1e-6)

        # Positive z (elliptic)
        z_pos = (np.pi / 2.0)**2
        assert np.isclose(stumpff_c(z_pos), 1.0 / z_pos, atol=1e-6)

        # Negative z (hyperbolic)
        z_neg = -1.0
        assert stumpff_c(z_neg) > 0.5
        assert stumpff_s(z_neg) > 1.0 / 6.0

    def test_kepler_universal_propagation_circular(self):
        """Verify circular LEO orbit propagation and conservation of energy."""
        r0, v0 = kepler_to_cartesian(
            a_km=EARTH_RADIUS_KM + 500.0,
            ecc=0.0,
            inc_deg=45.0,
            raan_deg=30.0,
            argp_deg=0.0,
            nu_deg=0.0,
        )
        r0_norm = np.linalg.norm(r0)
        v0_norm = np.linalg.norm(v0)

        # Propagate half orbit: period = 2*pi*sqrt(a^3/mu)
        period = 2.0 * np.pi * np.sqrt(r0_norm**3 / EARTH_MU)
        r_half, v_half, _, _, _, _ = kepler_universal_propagate(r0, v0, period / 2.0)

        # Half orbit position should be diametrically opposite
        assert np.isclose(np.linalg.norm(r_half), r0_norm, atol=1e-6)
        assert np.isclose(np.linalg.norm(v_half), v0_norm, atol=1e-6)
        assert np.isclose(np.dot(r_half, r0) / (r0_norm**2), -1.0, atol=1e-6)

        # Full orbit returns to start
        r_full, v_full, _, _, _, _ = kepler_universal_propagate(r0, v0, period)
        assert np.allclose(r_full, r0, atol=1e-5)
        assert np.allclose(v_full, v0, atol=1e-5)

        # Reverse propagation returns to r0
        r_back, v_back, _, _, _, _ = kepler_universal_propagate(r_half, v_half, -period / 2.0)
        assert np.allclose(r_back, r0, atol=1e-5)
        assert np.allclose(v_back, v0, atol=1e-5)


class TestGibbsAndHerrickGibbs:
    """Tests for 3-position vector IOD: Gibbs and Herrick-Gibbs."""

    @pytest.fixture
    def simulated_leo_triplet(self):
        """Generate three ground-truth position vectors along an eccentric LEO orbit."""
        r0, v0 = kepler_to_cartesian(
            a_km=EARTH_RADIUS_KM + 600.0,
            ecc=0.02,
            inc_deg=51.6,
            raan_deg=45.0,
            argp_deg=30.0,
            nu_deg=10.0,
        )
        dt = 60.0
        r1, v1, _, _, _, _ = kepler_universal_propagate(r0, v0, dt)
        r2, v2, _, _, _, _ = kepler_universal_propagate(r0, v0, 2 * dt)
        return r0, r1, r2, v0, v1, v2, dt

    def test_gibbs_iod_accuracy(self, simulated_leo_triplet):
        """Verify Gibbs method recovers true velocity vector with sub-micron/s precision."""
        r0, r1, r2, v0, v1, v2, _ = simulated_leo_triplet
        v1_gibbs = gibbs_iod(r0, r1, r2, mu=EARTH_MU)
        error_norm = np.linalg.norm(v1_gibbs - v1)
        assert error_norm < 1e-9  # Less than 1 nm/s error on exact 2-body orbit

    def test_herrick_gibbs_iod_accuracy(self, simulated_leo_triplet):
        """Verify Herrick-Gibbs achieves millimeter/s accuracy for close passes."""
        r0, r1, r2, v0, v1, v2, dt = simulated_leo_triplet
        v1_hg = herrick_gibbs_iod(r0, r1, r2, 0.0, dt, 2 * dt, mu=EARTH_MU)
        error_norm = np.linalg.norm(v1_hg - v1)
        assert error_norm < 1e-4  # Less than 0.1 m/s error

    def test_three_position_iod_selector(self, simulated_leo_triplet):
        """Verify three_position_iod auto selector and explicit dispatch."""
        r0, r1, r2, _, v1, _, dt = simulated_leo_triplet
        v_auto = three_position_iod(r0, r1, r2, 0.0, dt, 2 * dt, method="auto")
        assert np.isclose(np.linalg.norm(v_auto - v1), 0.0, atol=1e-3)

        v_gibbs = three_position_iod(r0, r1, r2, 0.0, dt, 2 * dt, method="gibbs")
        assert np.isclose(np.linalg.norm(v_gibbs - v1), 0.0, atol=1e-8)

        v_hg = three_position_iod(r0, r1, r2, 0.0, dt, 2 * dt, method="herrick_gibbs")
        assert np.isclose(np.linalg.norm(v_hg - v1), 0.0, atol=1e-3)

        with pytest.raises(ValueError):
            three_position_iod(r0, r1, r2, 0.0, dt, 2 * dt, method="unsupported_method")

    def test_gibbs_coplanarity_error(self, simulated_leo_triplet):
        """Verify Gibbs raises ValueError when position vectors violate coplanarity."""
        r0, r1, r2, _, _, _, _ = simulated_leo_triplet
        # Perturb r2 heavily out of plane
        normal = np.cross(r0, r2)
        normal_unit = normal / np.linalg.norm(normal)
        r1_perturbed = r1 + 500.0 * normal_unit

        with pytest.raises(ValueError, match="coplanarity"):
            gibbs_iod(r0, r1_perturbed, r2, coplanar_tol_deg=1.0)


class TestGaussIOD:
    """Tests for Gauss Initial Orbit Determination with angles-only measurements."""

    @pytest.fixture
    def optical_tracking_pass(self):
        """Generate synthetic optical tracking pass from a ground station."""
        r0, v0 = kepler_to_cartesian(
            a_km=EARTH_RADIUS_KM + 600.0,
            ecc=0.005,
            inc_deg=45.0,
            raan_deg=0.0,
            argp_deg=0.0,
            nu_deg=0.0,
        )
        lat0, lon0, _ = eci_to_geodetic(r0, gmst_rad=0.0)

        times = [0.0, 45.0, 90.0]
        observations = generate_synthetic_optical_observations(
            r0=r0,
            v0=v0,
            site_lat_deg=lat0,
            site_lon_deg=lon0,
            site_alt_km=0.1,
            observation_times_s=times,
            gmst0_rad=0.0,
            noise_arcsec=0.0,
        )
        r1_true, v1_true, _, _, _, _ = kepler_universal_propagate(r0, v0, 45.0)
        return observations, r1_true, v1_true

    def test_gauss_iod_series_solution(self, optical_tracking_pass):
        """Verify non-iterative Gauss series solution achieves sub-kilometer accuracy."""
        obs, r1_true, v1_true = optical_tracking_pass
        res = gauss_iod(obs[0], obs[1], obs[2], iterate=False)

        assert res.method == "gauss"
        r_err = np.linalg.norm(res.r2_eci_km - r1_true)
        v_err = np.linalg.norm(res.v2_eci_km - v1_true)

        # Pure series approximation without iteration typically achieves < 1 km and < 10 m/s
        assert r_err < 1.0
        assert v_err < 0.02
        assert res.elements["semi_major_axis_km"] > EARTH_RADIUS_KM

    def test_gauss_iod_iterative_refinement(self, optical_tracking_pass):
        """Verify Gauss iterative refinement converges to sub-millimeter precision."""
        obs, r1_true, v1_true = optical_tracking_pass
        res = gauss_iod(obs[0], obs[1], obs[2], iterate=True, tol=1e-8)

        assert res.converged is True
        assert res.method == "gauss_iterative"
        r_err = np.linalg.norm(res.r2_eci_km - r1_true)
        v_err = np.linalg.norm(res.v2_eci_km - v1_true)

        # Iterative solver converges to machine/sub-millimeter precision on 2-body orbit
        assert r_err < 1e-6  # < 1 mm
        assert v_err < 1e-8  # < 0.01 mm/s
        assert res.residual_rms_arcsec < 0.01

        # Summary string check
        summary = res.summary()
        assert "GAUSS_ITERATIVE" in summary
        assert "Converged: True" in summary

    def test_gauss_iod_with_light_time_correction(self):
        """Verify Gauss IOD compensates for finite speed of light travel delay."""
        r0, v0 = kepler_to_cartesian(
            a_km=EARTH_RADIUS_KM + 600.0,
            ecc=0.005,
            inc_deg=45.0,
            raan_deg=0.0,
            argp_deg=0.0,
            nu_deg=0.0,
        )
        lat0, lon0, _ = eci_to_geodetic(r0, gmst_rad=0.0)
        times = [0.0, 45.0, 90.0]

        # Generate observations modeling photon light-travel delay
        obs_lt = generate_synthetic_optical_observations(
            r0=r0,
            v0=v0,
            site_lat_deg=lat0,
            site_lon_deg=lon0,
            site_alt_km=0.1,
            observation_times_s=times,
            gmst0_rad=0.0,
            light_time_delay=True,
        )
        r1_true, v1_true, _, _, _, _ = kepler_universal_propagate(r0, v0, 45.0)

        # Solving without light-time correction has ~1.8 km error due to photon travel lag
        res_uncorrected = gauss_iod(obs_lt[0], obs_lt[1], obs_lt[2], iterate=True, light_time_correction=False)
        err_uncorrected = np.linalg.norm(res_uncorrected.r2_eci_km - r1_true)
        assert err_uncorrected > 1.0  # Photon travel time lag produces ~1.8 km shift

        # Solving WITH light-time correction compensates the photon delay
        res_corrected = gauss_iod(obs_lt[0], obs_lt[1], obs_lt[2], iterate=True, light_time_correction=True)
        assert res_corrected.light_time_corrected is True
        assert res_corrected.converged is True
        err_corrected = np.linalg.norm(res_corrected.r2_eci_km - r1_true)
        assert err_corrected < 0.05  # Reduced by over 98% to < 50 meters!

    def test_gauss_iod_error_handling(self):
        """Verify Gauss IOD error handling for non-chronological or coplanar inputs."""
        site = np.array([6378.0, 0.0, 0.0])
        obs1 = OpticalObservation(10.0, 0.0, 0.0, site)
        obs2 = OpticalObservation(5.0, 10.0, 10.0, site)  # Decreasing time
        obs3 = OpticalObservation(20.0, 20.0, 20.0, site)

        with pytest.raises(ValueError, match="chronological"):
            gauss_iod(obs1, obs2, obs3)

        # Collinear line-of-sight vectors (det(M) near 0)
        obs_c1 = OpticalObservation(0.0, 10.0, 10.0, site)
        obs_c2 = OpticalObservation(10.0, 10.0, 10.0, site)
        obs_c3 = OpticalObservation(20.0, 10.0, 10.0, site)
        with pytest.raises(ValueError, match="near-coplanar"):
            gauss_iod(obs_c1, obs_c2, obs_c3)


class TestLaplaceIOD:
    """Tests for Laplace Initial Orbit Determination."""

    def test_laplace_iod_accuracy(self):
        """Verify Laplace IOD accuracy for a close optical tracking pass."""
        r0, v0 = kepler_to_cartesian(
            a_km=EARTH_RADIUS_KM + 600.0,
            ecc=0.005,
            inc_deg=45.0,
            raan_deg=0.0,
            argp_deg=0.0,
            nu_deg=0.0,
        )
        lat0, lon0, _ = eci_to_geodetic(r0, gmst_rad=0.0)

        dt = 10.0  # 10s intervals
        times = [0.0, dt, 2 * dt]
        observations = generate_synthetic_optical_observations(
            r0=r0,
            v0=v0,
            site_lat_deg=lat0,
            site_lon_deg=lon0,
            site_alt_km=0.1,
            observation_times_s=times,
            gmst0_rad=0.0,
        )
        r1_true, v1_true, _, _, _, _ = kepler_universal_propagate(r0, v0, dt)

        res = laplace_iod(observations[0], observations[1], observations[2])
        assert res.method == "laplace"
        r_err = np.linalg.norm(res.r2_eci_km - r1_true)
        v_err = np.linalg.norm(res.v2_eci_km - v1_true)

        # Laplace error at dt=10s is typically < 5 km and < 0.05 km/s
        assert r_err < 5.0
        assert v_err < 0.05
        assert res.residual_rms_arcsec < 250.0

    def test_laplace_iod_error_handling(self):
        """Verify non-chronological observations trigger ValueError."""
        site = np.array([6378.0, 0.0, 0.0])
        obs1 = OpticalObservation(10.0, 0.0, 0.0, site)
        obs2 = OpticalObservation(10.0, 10.0, 10.0, site)
        obs3 = OpticalObservation(20.0, 20.0, 20.0, site)
        with pytest.raises(ValueError, match="chronological"):
            laplace_iod(obs1, obs2, obs3)


class TestUnifiedDispatcherAndBatch:
    """Tests for initial_orbit_determination dispatcher, noise effects, and batch screening."""

    @pytest.fixture
    def leo_observations(self):
        """Generate 5 observations for batch and dispatcher testing."""
        r0, v0 = kepler_to_cartesian(
            a_km=EARTH_RADIUS_KM + 550.0,
            ecc=0.002,
            inc_deg=51.6,
            raan_deg=20.0,
            argp_deg=10.0,
            nu_deg=0.0,
        )
        lat0, lon0, _ = eci_to_geodetic(r0, gmst_rad=0.0)
        times = [0.0, 30.0, 60.0, 90.0, 120.0]
        return generate_synthetic_optical_observations(
            r0=r0,
            v0=v0,
            site_lat_deg=lat0,
            site_lon_deg=lon0,
            site_alt_km=0.1,
            observation_times_s=times,
            gmst0_rad=0.0,
        )

    def test_unified_dispatcher_modes(self, leo_observations):
        """Verify unified dispatcher dispatches correctly to all methods."""
        obs1, obs2, obs3 = leo_observations[0], leo_observations[1], leo_observations[2]

        res_auto = initial_orbit_determination(obs1, obs2, obs3, method="auto")
        assert res_auto.method == "gauss_iterative"
        assert res_auto.converged is True

        res_gauss = initial_orbit_determination(obs1, obs2, obs3, method="gauss")
        assert res_gauss.method == "gauss"

        res_lap = initial_orbit_determination(obs1, obs2, obs3, method="laplace")
        assert res_lap.method == "laplace"

        with pytest.raises(ValueError, match="Unknown IOD method"):
            initial_orbit_determination(obs1, obs2, obs3, method="invalid_iod")

    def test_batch_optical_screening(self, leo_observations):
        """Verify batch sliding window IOD across 5 observations yields 3 solutions."""
        results = batch_optical_screening(leo_observations, window_size=3, method="gauss_iterative")
        assert len(results) == 3
        for res in results:
            assert res.converged is True
            assert res.elements["semi_major_axis_km"] > EARTH_RADIUS_KM

    def test_noise_robustness(self):
        """Verify IOD solution remains stable under realistic 1 arcsec astrometric noise."""
        r0, v0 = kepler_to_cartesian(
            a_km=EARTH_RADIUS_KM + 600.0,
            ecc=0.003,
            inc_deg=45.0,
            raan_deg=0.0,
            argp_deg=0.0,
            nu_deg=0.0,
        )
        lat0, lon0, _ = eci_to_geodetic(r0, gmst_rad=0.0)
        times = [0.0, 45.0, 90.0]

        # 1.0 arcsecond astrometric noise
        obs_noisy = generate_synthetic_optical_observations(
            r0=r0,
            v0=v0,
            site_lat_deg=lat0,
            site_lon_deg=lon0,
            site_alt_km=0.1,
            observation_times_s=times,
            gmst0_rad=0.0,
            noise_arcsec=1.0,
            seed=42,
        )
        r1_true, v1_true, _, _, _, _ = kepler_universal_propagate(r0, v0, 45.0)

        res = gauss_iod(obs_noisy[0], obs_noisy[1], obs_noisy[2], iterate=True)
        r_err = np.linalg.norm(res.r2_eci_km - r1_true)
        v_err = np.linalg.norm(res.v2_eci_km - v1_true)

        # Under 1 arcsec noise on a short 90s optical baseline, position is recovered within ~50 km
        # and inclination within 0.5 degrees
        assert r_err < 100.0  # < 100 km
        assert v_err < 1.0    # < 1 km/s
        assert abs(res.elements["inclination_deg"] - 45.0) < 0.5
        assert res.elements["semi_major_axis_km"] > EARTH_RADIUS_KM
