# Changelog

All notable changes to the **AEGIS-LEO** project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2026-11-16

### Added
- **Mission Analytics Engine (`aegis_leo/analytics/mission_analytics.py`)**:
  - Per-subsystem health scoring (0–100) across 7 subsystems: PROPULSION, COLLISION_RISK, POWER, THERMAL, COMMS, RADIATION, SECURITY.
  - Per-channel anomaly flagging with `AnomalyFlag` severity (INFO / WARNING / CRITICAL) and threshold-aware descriptions.
  - Fuel budget burn-rate projection under nominal and storm conditions using Tsiolkovsky mass-flow model.
  - Radiation life estimation with TID accumulation, annual dose rate, and years-to-limit calculation.
  - Mission lifecycle stage classifier (LAUNCH_AND_EARLY_ORBIT → COMMISSIONING → NOMINAL → END_OF_LIFE → DEORBIT_SEQUENCE).
  - Rolling trend analysis across up to 10 historical snapshots using linear regression + residual-variance direction classifier (IMPROVING / STABLE / DEGRADING / VOLATILE).
  - `MissionAnalyticsEngine` stateful class and `analyze_mission_health` stateless convenience function.
- **Advanced Debris Tracker (`aegis_leo/analytics/debris_tracker.py`)**:
  - `TrackedObject` Keplerian state with auto-computed mean motion and orbital period.
  - All-pairs conjunction screening using analytic RAAN-plane crossing times, J2 secular nodal precession (`_secular_j2_raan_rate`), and altitude shell pre-filter.
  - A-DRI-inspired `_compute_conjunction_risk_score` per pair (miss distance, relative speed, area/mass, altitude band).
  - Secondary conjunction chain detection: post-evasion manoeuvre proxy orbit screened against catalogue.
  - `DebrisFleetRiskMatrix` output with `top_threats(n)` ranking method.
  - `build_synthetic_catalogue` and `build_debris_risk_matrix` convenience builders for rapid testing.
- **Probabilistic Space Weather Forecast Engine (`aegis_leo/physics/space_weather_forecast.py`)**:
  - AR(4) Kp autoregressive predictor seeded from OMNI2-derived autocorrelation coefficients.
  - AR(2) F10.7 solar flux predictor with 27-day rotation noise.
  - CME transit model (Wang et al. 2004 empirical) with probabilistic Kp injection and onset confidence interval.
  - 50-member Monte Carlo ensemble producing hourly P10/P50/P90 Kp and F10.7 traces for up to 72 h.
  - Per-hour thermospheric density factor μ(t) and along-track drift rate accumulation.
  - `StormOnsetAlert` early-warning trigger when P_storm(24h) > 70% or CME onset confidence > 40%.
  - `SpaceWeatherForecaster` class and `run_space_weather_forecast` convenience function.
- **Multi-layer Telemetry Anomaly Detector (`aegis_leo/ai/anomaly_detector.py`)**:
  - Per-channel sliding-window Z-score outlier detection (configurable σ threshold).
  - CUSUM (Cumulative SUM) control chart for change-point detection on each channel.
  - Multivariate Mahalanobis distance anomaly detection across all telemetry channels simultaneously.
  - Physics consistency checks: vis-viva orbital energy, power balance (eclipse vs sunlit), thermal node operational limits, Tsiolkovsky fuel mass budget (±10% tolerance).
  - SEU bit-flip signature detection in PINN output stream using modified Z-score (robust to Gaussian outliers).
  - `TelemetryAnomalyDetector` stateful class and `detect_telemetry_anomalies` batch convenience function.
- **5 New REST API Endpoints (`backend/main.py` — Wave 5)**:
  - `POST /api/analytics/mission-health`: Full subsystem health KPI snapshot.
  - `POST /api/analytics/debris-risk-matrix`: Multi-object debris fleet conjunction matrix.
  - `POST /api/space-weather/forecast`: 24–72 h ensemble Kp/density forecast with CME model.
  - `POST /api/analytics/detect-anomalies`: Batch telemetry frame anomaly detection.
  - `GET /api/analytics/subsystem-trends`: Rolling subsystem trend analysis across synthetic epochs.
- **Streamlit Mission Control Tab 17 (`app.py`)**:
  - Sub-tab 1 — 🏥 Görev Sağlık Panosu: Subsystem radar chart, KPI cards, anomaly list.
  - Sub-tab 2 — 🌦️ Uzay Havası Tahmini: Interactive Kp P10/P50/P90 ensemble, density factor trace, CME alerts.
  - Sub-tab 3 — ☄️ Enkaz Filo Risk Matrisi: TCA scatter plot, top-5 threat table, secondary chain warning.
  - Sub-tab 4 — 🔍 Telemetri Anomali Dedektörü: Health-score time series, anomaly bar overlay, channel Z-score heatmap.
- **Test suite expanded to 468/468 passing tests**.

## [2.1.0] - 2026-11-15

### Added
- **LEO Trapped Radiation, South Atlantic Anomaly & Dose Engine (`aegis_leo/physics/radiation_environment.py`)**:
  - Eccentric tilted geomagnetic dipole (field magnitude, McIlwain L-shell, geomagnetic latitude) along the Earth-fixed ground track.
  - Parametric inner-belt proton (>30 MeV) and outer-belt electron (>1 MeV) flux with an atmospheric loss-cone cutoff that reproduces the SAA, Kp-driven electron enhancement, Stormer cutoff rigidity and storm-time polar-cap SEP access.
  - Orbit-averaged fluence, TID behind aluminium shielding, proton-induced SEU rate and recommended TMR memory scrub interval. Documented as a first-order engineering model, not an AP-9/AE-9 replacement.
- **Impulsive Orbit Transfer Planner (`aegis_leo/astrodynamics/orbit_transfer.py`)**: Hohmann, bi-elliptic, optimally split combined plane change, coplanar phasing and Tsiolkovsky propellant feasibility with strategy trade-off report.
- **TCA Finder & Conjunction Screening (`aegis_leo/risk/tca_finder.py`)**: analytic Kepler + secular J2 propagation, apogee/perigee shell filter, coarse scan with Brent refinement of every range minimum, RIC miss vector, relative speed and synthetic crossing-debris scenario builder.
- **Integration**: mission pipeline sections `radiation_environment`, `tca_screening`, `disposal_transfer`; REST endpoints `POST /api/physics/radiation-environment`, `GET /api/physics/saa-map`, `POST /api/astrodynamics/orbit-transfer`, `POST /api/risk/tca-screening`; CLI subcommands `aegis radiation`, `aegis transfer`, `aegis tca`; Streamlit Mission Control Tab 16.
- `LICENSE` file (MIT), matching the license already declared in `pyproject.toml` and the README.
- Test suite expanded to **413/413 passing tests**.

### Fixed
- API root endpoint and OpenAPI metadata reported stale versions (`1.0.0` / `2.0.0`); both now read `aegis_leo.__version__`.
- Streamlit HUD crashed with `KeyError: 'id'` in Tabs 13 and 15 (catalog entries have no `id` field), which also stopped every later tab from rendering; the selected catalog key is used instead.
- CI now smoke-tests the new CLI subcommands.

## [2.0.0] - 2026-11-15

### Added
- **6-DOF Attitude Determination & Control System (ADCS) Engine (`aegis_leo/astrodynamics/attitude_adcs.py`)**:
  - Full rigid-body Euler rotational dynamics coupled with unit-quaternion kinematics ($\dot{\mathbf{q}} = \frac{1}{2}\boldsymbol{\Omega}(\boldsymbol{\omega})\mathbf{q}$).
  - Gravity-gradient, storm-responsive aerodynamic weathercock, and solar radiation pressure disturbance torques.
  - 4-wheel symmetric pyramid Reaction Wheel Assembly (RWA) Moore-Penrose pseudoinverse allocation ($\mathbf{W}^+$) with wheel speed/momentum saturation guards.
  - Magnetic Torque Rod (MTQ) cross-product momentum dumping ($\mathbf{m}_{dump} = k (\mathbf{h}_{rw} \times \mathbf{B}) / \|\mathbf{B}\|^2$) and pre-burn slew-to-attitude readiness verification.
- **Universal-Variable Lambert Rendezvous Solver & Multi-Debris ADR Tour Optimizer (`aegis_leo/astrodynamics/lambert_tour.py`)**:
  - Universal variable & Stumpff $C(z), S(z)$ two-point orbital boundary value solver ($\mathbf{r}_1, \mathbf{r}_2, \Delta t \to \mathbf{v}_1, \mathbf{v}_2$).
  - Secular $J_2$ nodal precession ($\dot{\Omega}$) drift phasing rate estimator for low-fuel orbital plane alignment.
  - Combinatorial Multi-Target Active Debris Removal (ADR) tour planner sequencing high-risk LEO debris targets within servicer propellant limits.
- **5-Node Spacecraft Orbital Thermal Control & Radiative Equilibrium Engine (`aegis_leo/physics/thermal_analyzer.py`)**:
  - Coupled transient orbital thermal network across `SOLAR_ARRAY`, `BUS_STRUCTURE`, `EDGE_AI_AVIONICS`, `BATTERY_PACK`, and `PROPULSION_UNIT`.
  - Models direct solar flux, Earth albedo, Earth outgoing longwave IR (OLR), VLEO/storm free-molecular aerodynamic heating ($Q_{aero} = \frac{1}{2}\rho v^3 C_h A$), active thermostat survival heaters, and Stefan-Boltzmann $T^4$ deep-space radiative cooling.
- **Post-Quantum Cryptographic (PQC) Space Data Link Security (`aegis_leo/security/pqc_telecommand.py`)**:
  - Module-LWE (ML-KEM / Kyber-inspired) lattice key encapsulation over the negacyclic polynomial ring $\mathcal{R}_q = \mathbb{Z}_{3329}[X]/(X^{64} + 1)$ with 100% exact shared key recovery.
  - CCSDS 355.0-B-2 Space Data Link Security (SDLS) telecommand frame authentication (HMAC-SHA3-256), monotonic anti-replay sliding window sequence counter, tamper-evident Merkle command chain, and adversarial intrusion prevention ledger.
- **Expanded Mission Pipeline, 15-Tab Streamlit HUD, CLI & 45+ FastAPI Endpoints**:
  - Integrated ADCS slew readiness, 5-node thermal safety, and PQC command verification into `run_end_to_end_mission`.
  - Added REST endpoints: `POST /api/astrodynamics/adcs-slew`, `POST /api/physics/thermal-analysis`, `POST /api/security/pqc-verify`, `POST /api/adr/lambert-tour`.
  - Added CLI subcommands: `aegis adcs`, `aegis thermal`, `aegis pqc`, `aegis tour`.
  - Added Streamlit Mission Control Tabs 14 & 15 (**15 Operational Tabs** total) and expanded the automated test suite to **385/385 passing tests**.

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
