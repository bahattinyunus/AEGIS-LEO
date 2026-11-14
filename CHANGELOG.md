# Changelog

All notable changes to the **AEGIS-LEO** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-11-14

### Added
- **CCSDS OEM & OPM Export (`aegis_leo/data/ccsds_ephemeris.py`)**: Interoperability standard for NASA GMAT, ESA DRAMA, and STK.
- **Powered Flight Integrator & Maneuver Splicer (`aegis_leo/physics/maneuver_splicer.py`)**: Continuous finite-duration rocket burn simulation with mass depletion.
- **Monte Carlo Conjunction Simulator (`aegis_leo/risk/monte_carlo.py`)**: 3D covariance dispersion sampling and empirical Pc calculation.
- **NASA Standard Breakup & Kessler Cascade (`aegis_leo/risk/kessler_cascade.py`)**: Power-law fragment size distribution and LEO carrying capacity model.
- **Aegis CLI (`aegis_leo/cli.py`)**: Production terminal tool registered via `pyproject.toml`.
- **Docker & CI/CD Pipelines**: Multi-stage `Dockerfile`, `docker-compose.yml`, and GitHub Actions workflow (`.github/workflows/ci.yml`).
- **Walker Delta Constellation Screening (`aegis_leo/screening/`)**: Multi-satellite all-on-all screening with spatial gating.
- **Pareto Multi-Objective Maneuver Optimizer (`aegis_leo/ai/pareto_optimizer.py`)**: Trade-off frontier for Delta-v vs clearance vs mission downtime.
- **Multi-Model Thermosphere Physics (`aegis_leo/physics/atmosphere_models.py`)**: Jacchia-77, NRLMSISE-00, and Exponential scale models.
- **Ground Track & Blackout Analysis (`aegis_leo/physics/ground_track.py`)**: Sub-satellite 2D tracks and ionospheric communication blackouts.

## [1.0.0] - 2026-11-14

### Added
- Initial release for NASA Space Apps Challenge 2026.
- Core thermospheric drag and Cowell RK4 propagator.
- PyTorch Physics-Informed Neural Network (PINN) surrogate model.
- Adaptive Debris & Drag Risk Index (A-DRI).
- Autonomous Edge-AI Avoidance Optimizer.
- NASA-STD-8719.14 Compliance Auditor.
- FastAPI backend and 3D Streamlit visualizer.
