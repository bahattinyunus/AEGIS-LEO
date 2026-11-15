# Changelog

All notable changes to the **AEGIS-LEO** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-11-15

### Added
- **Ground Station Pass Scheduler & RF/Optical Link Budget (`aegis_leo/physics/pass_scheduler.py`)**:
  - Topocentric SEZ look-angle kinematics (Azimuth, Elevation, Slant Range, Range Rate, Doppler shift).
  - AOS/LOS pass window scheduling across global ground stations (Ankara, Svalbard, Goldstone, Madrid, Canberra).
  - Multi-band RF & Optical downlink budget (`S_BAND`, `X_BAND`, `KA_BAND`, `OPTICAL`) with FSPL, atmospheric/rain fade, Shannon throughput, and pass data volume integration.
- **Orbital Eclipse, Beta Angle & Solar/Battery Power Budget (`aegis_leo/physics/eclipse_power.py`)**:
  - Astronomical Sun vector, orbital Beta angle ($\beta$), and J2 Sun-Synchronous Orbit (SSO) inclination solver.
  - Dual-cone Earth shadow model (Umbra, Penumbra, Sunlit) with solar array power, battery Depth-of-Discharge (DoD %), and thermal swing estimation.
- **End-to-End Integrated Autonomous Mission Pipeline (`aegis_leo/simulation/mission_pipeline.py`)**:
  - Single unified pipeline orchestrating space weather, multi-model thermosphere, eclipse/power, ground comms, Foster B-plane + A-DRI risk, Edge-AI avoidance, 9-agent swarm consensus, and NASA/FCC compliance.
- **Full FastAPI Backend REST Coverage (`backend/main.py`)**:
  - Added `/api/comms/pass-schedule`, `/api/physics/eclipse-power`, `/api/mission/end-to-end`.
  - Exposed Wave 2 subsystems via REST: `/api/risk/foster-bplane`, `/api/compliance/odar-report`, `/api/navigation/ekf-od`, `/api/navigation/iod-gauss`, `/api/sensors/radar-detect`, `/api/ai/mpc-avoidance`, `/api/ai/vleo-drag-makeup`, `/api/ai/federated-swarm`.
- **Streamlit Mission Control Tab 13 (`app.py`)** & **CLI Subcommands (`aegis passes`, `aegis eclipse`, `aegis odar`, `aegis mission`)**.

## [1.4.0] - 2026-11-14

### Added
- **Grand Agent Army Wave 2 Expansion (9 Additional Autonomous Specialist Agents)**:
  - `nav-worker` (`aegis_leo/navigation/sensor_fusion.py`): Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF) for onboard Autonomous Orbit Determination (AOD) fusing Star Tracker LOS vectors, GNSS pseudoranges, and storm-adaptive magnetometers.
  - `sensor-worker` (`aegis_leo/sensors/laser_radar.py`): Spaceborne LIDAR & Radar active sensor emulation with Swerling I-IV fluctuation models, Doppler radar, CFAR detection, and 2D Range-Doppler maps.
  - `iod-worker` (`aegis_leo/navigation/iod_gauss.py`): Initial Orbit Determination (IOD) from optical angles-only tracks (Gauss, Laplace, Gibbs, and Herrick-Gibbs) with light-time travel correction.
  - `mpc-worker` (`aegis_leo/ai/mpc_avoidance.py`): Model Predictive Control (MPC) and Sequential Convex Programming (SCP) with primal-dual interior-point QP solver for non-convex collision avoidance.
  - `vleo-worker` (`aegis_leo/ai/drag_makeup.py`): Adaptive Drag Make-Up (ADMU) controller engine for VLEO (180-300 km) using L1 adaptive control and anti-windup PID against atmospheric density waves.
  - `hcw-worker` (`aegis_leo/astrodynamics/cw_proximity.py`): Hill-Clohessy-Wiltshire (HCW) and Yamanaka-Ankersen elliptical relative motion for Active Debris Removal (ADR) rendezvous and Proximity Operations (RPO).
  - `adr-worker` (`aegis_leo/adr/debris_capture.py`): Active Debris Removal (ADR) capture dynamics: 6-DOF robotic manipulator kinematics, harpoon tethered momentum exchange, and upper stage detumbling.
  - `cdm-worker` (`aegis_leo/data/ccsds_cdm_xml.py`): Complete CCSDS 508.0-B-1 Conjunction Data Message (CDM) XML and KVN parser, schema validator, and astrodynamic verifier.
  - `fed-worker` (`aegis_leo/ai/federated_swarm.py`): Federated Learning across satellite swarms (FedAvg and FedProx) via Inter-Satellite Links (ISL) with zero raw telemetry leakage and DP noise.
- **Repository Test Suite Milestone**: All **329 unit and integration tests** passing with 100% success rate.

## [1.3.0] - 2026-11-14

### Added
- **Multi-Agent Swarm Collaborative Expansion (9 Parallel Specialist Agents)**:
  - `solar-worker` (`aegis_leo/physics/perturbations.py`): Solar Radiation Pressure (SRP) cannonball model with dual-cone shadows & Battin algebraic third-body Sun/Moon gravitational perturbations.
  - `thermo-worker` (`aegis_leo/physics/drag_cross_section.py`): Variable attitude ballistic coefficient dynamics, solar array cross-sectional area tracking (1-DOF/2-DOF SADM), and Sentman free-molecular drag model.
  - `risk-worker` (`aegis_leo/risk/foster_bplane.py`): Foster (1992) numerical B-plane integration via Gauss-Legendre quadrature & Chan (1997) analytical series with Alfano/CARA maximum collision probability.
  - `astro-worker` (`aegis_leo/ai/low_thrust.py`): Continuous low-thrust trajectory planner for Hall effect ion thrusters with Edelbaum analytical equations, RK4 micro-thrust integration, and orbit restoration cycles.
  - `edge-worker` (`aegis_leo/ai/hardware_telemetry.py`): Space-grade flight computer hardware watchdog (SEU SECDED tracking, thermal cycle monitor, DVFS clock throttling, INT8 canary memory guards).
  - `fleet-worker` (`aegis_leo/screening/stationkeeping.py`): Constellation slot stationkeeping delta-v budgeting and Walker-Delta intra-plane drift phasing with Kuhn-Munkres Hungarian optimal assignment.
  - `audit-worker` (`aegis_leo/compliance/odar_generator.py`): NASA-STD-8719.14B Section 4 formal Orbital Debris Assessment Report (ODAR) Markdown and JSON document generator.
  - `disposal-worker` (`aegis_leo/physics/reentry_footprint.py`): Point Nemo (SPOUA) Monte Carlo reentry ground impact dispersion ellipse calculation with 1-3 sigma casualty risk contours.
  - `commander-worker` (`aegis_leo/agents/swarm_telemetry_bus.py`): High-speed pub/sub telemetry streaming bus, circular replay buffer, and SHA-256 tamper-evident structured event logger.
- **Test Suite Expansion**: Repository tests expanded from 53 to **149 passed tests** (100% pass rate).

## [1.2.0] - 2026-11-14

### Added
- **9-Agent Autonomous Swarm Intelligence (`aegis_leo/agents/`)**:
  - `SolarStormAgent-01`: Space weather, CME speed, and Kp/F10.7 storm detection.
  - `ThermosphereAgent-02`: Real-time atmospheric surge and along-track drift tracking.
  - `RiskAssessmentAgent-03`: B-plane covariance projection and A-DRI calculation.
  - `AstrodynamicsAgent-04`: Vis-Viva proactive micro-burn synthesis and fuel minimization.
  - `EdgeInferenceAgent-05`: Sub-millisecond PINN inference with Triple Modular Redundancy (TMR) fault masking.
  - `ConstellationConsensusAgent-06`: Walker-Delta inter-satellite link (ISL) mesh coordination and slot conflict avoidance.
  - `ComplianceWatchdogAgent-07`: Independent watchdog enforcing NASA-STD-8719.14B, IADC guidelines, and passivation reserves with absolute VETO power.
  - `ReentryDisposalAgent-08`: Controlled deorbit trajectory planning toward Point Nemo (SPOUA) and orbital decay monitoring.
  - `MissionCommanderAgent-09`: Byzantine consensus evaluation, weighted voting ledger, and SHA-256 digital signature authorization.
  - `OrbitalSwarmOrchestrator`: Asynchronous blackboard-based lifecycle coordinator for all 9 agents.
- **Interactive Swarm Dashboard (`app.py` Tab 9)**: Real-time 3x3 agent matrix, live voting ledger, inter-agent communication feed, and cryptographic directive display.
- **Swarm CLI & API Endpoints**:
  - `aegis swarm` CLI subcommand with customizable satellite, miss distance, and space weather parameters.
  - `GET /api/swarm/status` and `POST /api/swarm/consensus-cycle` in FastAPI backend.
- **Unit & Integration Test Suite Expansion**: 11 new tests (`tests/test_agents_swarm.py` and `tests/test_api.py`), bringing the total test suite to 53 passing tests.

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
