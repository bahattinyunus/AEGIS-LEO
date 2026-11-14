"""
AEGIS-LEO: Variable Attitude Ballistic Coefficient & Solar Array Dynamics
Computes dynamic spacecraft cross-sectional area, solar array articulation,
and variable ballistic coefficient as functions of orbital true anomaly and solar beta angle.

Key capabilities:
1. Solar beta angle geometry from orbital elements or Cartesian state vectors.
2. 1-DOF and 2-DOF solar array drive mechanism (SADM) tracking dynamics.
3. Variable attitude modes: LVLH nadir-pointing with solar tracking, aero-feathering
   (drag minimization), broadside aerobraking (drag maximization / deorbit sail), and sun-pointing.
4. Eclipse shadow entry/exit dynamics with optional nightside feathering.
5. Sentman free-molecular flow drag coefficient modeling and effective ballistic coefficient BC = m / (Cd * A).
6. Orbit-averaged drag area and ballistic coefficient sweeps across solar beta angle space.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np

from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    QUIET_KP,
    QUIET_F10_7,
)
from aegis_leo.physics.atmosphere import compute_atmospheric_density
from aegis_leo.physics.coordinates import relative_atmospheric_velocity


class AttitudeMode(str, Enum):
    """Spacecraft attitude control and solar array articulation modes."""
    LVLH_SUN_TRACKING_1AXIS = "lvlh_sun_tracking_1axis"  # 1-DOF SADM rotation around pitch axis (orbit normal)
    LVLH_SUN_TRACKING_2AXIS = "lvlh_sun_tracking_2axis"  # 2-DOF SADM tracking Sun vector in 3D
    FEATHERED = "feathered"                              # Drag minimization: arrays edge-on to ram velocity
    DRAG_MAXIMIZATION = "drag_maximization"              # Deorbit mode: arrays broadside into ram velocity
    SUN_POINTING = "sun_pointing"                        # Spacecraft body & arrays fixed towards Sun
    FIXED_PARKED = "fixed_parked"                        # Fixed solar array angle relative to bus


class TrackingAxis(str, Enum):
    """Primary rotation axis for single-axis solar array drive mechanisms."""
    PITCH = "pitch"  # Aligned with orbit normal (h-vector / cross-track) - standard LEO
    ROLL = "roll"    # Aligned with ram velocity vector


@dataclass
class SpacecraftGeometry:
    """
    Spacecraft geometric, mass, and aerodynamic characteristics.

    Attributes:
        mass_kg: Total spacecraft on-orbit wet/dry mass (kg).
        bus_length_m: Body length along roll / ram axis in nominal LVLH (m).
        bus_width_m: Body width along pitch / cross-track axis in nominal LVLH (m).
        bus_height_m: Body height along yaw / nadir axis in nominal LVLH (m).
        solar_array_area_m2: Total solar panel surface area (both wings combined, m^2).
        solar_array_thickness_m: Panel edge thickness (m).
        solar_array_span_m: Total wing span perpendicular to thickness (m).
        cd_bus: Nominal aerodynamic drag coefficient of spacecraft bus.
        cd_solar_array: Nominal aerodynamic drag coefficient of solar arrays.
    """
    mass_kg: float = 150.0
    bus_length_m: float = 1.0
    bus_width_m: float = 1.0
    bus_height_m: float = 1.5
    solar_array_area_m2: float = 3.0
    solar_array_thickness_m: float = 0.02
    solar_array_span_m: float = 3.0
    cd_bus: float = 2.2
    cd_solar_array: float = 2.2

    @property
    def solar_array_edge_area_m2(self) -> float:
        """Edge-on projected cross section of solar arrays (m^2)."""
        return float(self.solar_array_span_m * self.solar_array_thickness_m)

    @property
    def bus_frontal_area_m2(self) -> float:
        """Nominal ram-facing cross-sectional area of bus in LVLH (width * height, m^2)."""
        return float(self.bus_width_m * self.bus_height_m)

    @property
    def bus_side_area_m2(self) -> float:
        """Side cross-sectional area of bus (length * height, m^2)."""
        return float(self.bus_length_m * self.bus_height_m)

    @property
    def bus_top_bottom_area_m2(self) -> float:
        """Nadir/zenith cross-sectional area of bus (length * width, m^2)."""
        return float(self.bus_length_m * self.bus_width_m)

    @property
    def bus_min_area_m2(self) -> float:
        """Minimum possible projected area of cuboid bus (m^2)."""
        return float(min(self.bus_frontal_area_m2, self.bus_side_area_m2, self.bus_top_bottom_area_m2))

    @property
    def bus_max_area_m2(self) -> float:
        """Maximum possible projected area of cuboid bus (m^2)."""
        return float(max(self.bus_frontal_area_m2, self.bus_side_area_m2, self.bus_top_bottom_area_m2))


def compute_solar_beta_angle(
    inc_rad: float,
    raan_rad: float,
    sun_ra_rad: float,
    sun_dec_rad: float,
) -> float:
    """
    Calculate the solar beta angle (radians) between the orbit plane and the solar vector.

    Formula:
        sin(beta) = cos(delta_sun) * sin(i) * sin(Omega - alpha_sun) + sin(delta_sun) * cos(i)

    Args:
        inc_rad: Orbit inclination (radians).
        raan_rad: Right ascension of ascending node (Omega, radians).
        sun_ra_rad: Solar right ascension (alpha_sun, radians).
        sun_dec_rad: Solar declination (delta_sun, radians).

    Returns:
        Solar beta angle beta in radians in [-pi/2, pi/2].
    """
    sin_beta = (
        np.cos(sun_dec_rad) * np.sin(inc_rad) * np.sin(raan_rad - sun_ra_rad)
        + np.sin(sun_dec_rad) * np.cos(inc_rad)
    )
    sin_beta_clipped = np.clip(sin_beta, -1.0, 1.0)
    return float(np.arcsin(sin_beta_clipped))


def compute_solar_beta_angle_from_vectors(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    sun_eci_km: np.ndarray,
) -> float:
    """
    Calculate the solar beta angle (radians) directly from ECI position, velocity, and Sun vectors.

    Formula:
        h = r x v
        h_hat = h / ||h||
        s_hat = sun / ||sun||
        beta = arcsin(h_hat . s_hat)

    Args:
        r_eci_km: ECI position vector [x, y, z] in km.
        v_eci_km_s: ECI velocity vector [vx, vy, vz] in km/s.
        sun_eci_km: ECI Sun position vector [sx, sy, sz] in km.

    Returns:
        Solar beta angle beta in radians in [-pi/2, pi/2].
    """
    h = np.cross(r_eci_km, v_eci_km_s)
    h_norm = np.linalg.norm(h)
    s_norm = np.linalg.norm(sun_eci_km)

    if h_norm < 1e-9 or s_norm < 1e-9:
        return 0.0

    h_hat = h / h_norm
    s_hat = sun_eci_km / s_norm
    dot_val = np.clip(np.dot(h_hat, s_hat), -1.0, 1.0)
    return float(np.arcsin(dot_val))


def is_in_eclipse(
    true_anomaly_rad: float,
    beta_rad: float,
    altitude_km: float = 500.0,
    nu_noon_rad: float = 0.0,
) -> bool:
    """
    Determine if spacecraft is in Earth's shadow (cylindrical umbra model).

    Args:
        true_anomaly_rad: Orbital true anomaly (radians).
        beta_rad: Solar beta angle (radians).
        altitude_km: Orbital altitude above Earth's surface (km).
        nu_noon_rad: True anomaly corresponding to orbit noon (radians).

    Returns:
        True if in eclipse (nightside shadow), False otherwise.
    """
    r_orbit_km = EARTH_RADIUS_KM + max(100.0, float(altitude_km))
    theta_rel = true_anomaly_rad - nu_noon_rad

    # Nightside condition: satellite is on the hemisphere away from the Sun
    cos_theta = np.cos(theta_rel)
    if cos_theta >= 0.0:
        return False  # Dayside hemisphere

    # Cylindrical shadow condition:
    # Distance from Earth-Sun centerline must be less than Earth's radius
    # d_perp^2 = r^2 * (1 - cos^2(theta_rel) * cos^2(beta))
    cos_beta = np.cos(beta_rad)
    sin2_cone = 1.0 - (cos_theta * cos_beta) ** 2
    d_perp = r_orbit_km * np.sqrt(max(0.0, sin2_cone))

    return bool(d_perp < EARTH_RADIUS_KM)


def compute_eclipse_fraction(
    beta_rad: float,
    altitude_km: float = 500.0,
) -> float:
    """
    Calculate the fraction of orbit spent in eclipse for a given beta angle and altitude.

    Args:
        beta_rad: Solar beta angle (radians).
        altitude_km: Orbital altitude (km).

    Returns:
        Eclipse fraction f in [0.0, 1.0]. Returns 0.0 for full sunlight orbits.
    """
    r_orbit_km = EARTH_RADIUS_KM + max(100.0, float(altitude_km))
    sin_beta_crit = EARTH_RADIUS_KM / r_orbit_km
    beta_crit = np.arcsin(np.clip(sin_beta_crit, 0.0, 1.0))

    if abs(beta_rad) >= beta_crit:
        return 0.0  # Orbit is in continuous sunlight

    cos_beta = np.cos(beta_rad)
    if cos_beta < 1e-6:
        return 0.0

    arg = np.sqrt(max(0.0, 1.0 - (EARTH_RADIUS_KM / r_orbit_km) ** 2)) / cos_beta
    arg_clipped = np.clip(arg, -1.0, 1.0)
    theta_e = np.arccos(arg_clipped)

    return float(theta_e / np.pi)


def compute_solar_array_cross_section(
    true_anomaly_rad: float,
    beta_rad: float,
    geometry: SpacecraftGeometry,
    attitude_mode: AttitudeMode = AttitudeMode.LVLH_SUN_TRACKING_1AXIS,
    nu_noon_rad: float = 0.0,
    eclipse_feathering: bool = False,
    tracking_axis: TrackingAxis = TrackingAxis.PITCH,
    fixed_gimbal_deg: float = 0.0,
) -> float:
    """
    Calculate instantaneous projected cross-sectional area of solar arrays perpendicular to velocity.

    Dynamics:
    - 1-Axis Sun Tracking (pitch axis / orbit normal):
      The array rotates around the orbit normal to track the in-plane Sun vector.
      Projected area along velocity vector:
        A_proj(theta_rel) = A_sa * |sin(theta_rel)| + A_edge * |cos(theta_rel)|
    - 2-Axis Sun Tracking:
      The array articulates in 2-DOF to track the 3D Sun vector.
      Dot product with velocity: v_hat . n_sa = -cos(beta) * sin(theta_rel)
      Projected area:
        A_proj(theta_rel, beta) = A_sa * |cos(beta) * sin(theta_rel)| + A_edge * sqrt(1 - cos^2(beta)*sin^2(theta_rel))
    - Feathered Mode: Arrays turned edge-on to velocity vector (minimum drag).
    - Drag Maximization Mode: Arrays oriented broadside to velocity vector (maximum drag for deorbit).
    - Eclipse Feathering: If active, arrays stow into edge-on orientation during eclipse passages.

    Args:
        true_anomaly_rad: Current orbital true anomaly (radians).
        beta_rad: Solar beta angle (radians).
        geometry: SpacecraftGeometry defining areas and dimensions.
        attitude_mode: Spacecraft attitude mode.
        nu_noon_rad: Orbit noon true anomaly (radians).
        eclipse_feathering: If True, feather arrays edge-on during eclipse to minimize decay.
        tracking_axis: Rotation axis for 1-axis tracking.
        fixed_gimbal_deg: Array cant angle if in FIXED_PARKED mode (degrees).

    Returns:
        Projected cross-sectional area in m^2.
    """
    theta_rel = true_anomaly_rad - nu_noon_rad
    a_face = geometry.solar_array_area_m2
    a_edge = geometry.solar_array_edge_area_m2

    # Check for nightside eclipse feathering
    if eclipse_feathering and is_in_eclipse(true_anomaly_rad, beta_rad, nu_noon_rad=nu_noon_rad):
        return float(a_edge)

    if attitude_mode == AttitudeMode.FEATHERED:
        return float(a_edge)

    if attitude_mode == AttitudeMode.DRAG_MAXIMIZATION:
        return float(a_face)

    if attitude_mode == AttitudeMode.FIXED_PARKED:
        gimbal_rad = np.radians(fixed_gimbal_deg)
        proj = a_face * abs(np.sin(gimbal_rad)) + a_edge * abs(np.cos(gimbal_rad))
        return float(proj)

    if attitude_mode == AttitudeMode.LVLH_SUN_TRACKING_1AXIS:
        if tracking_axis == TrackingAxis.PITCH:
            # Solar array rotates around orbit normal (h) to point towards in-plane Sun
            # Angle relative to velocity: dot(v_hat, n_sa) = -sin(theta_rel)
            sin_rel = abs(np.sin(theta_rel))
            cos_rel = abs(np.cos(theta_rel))
            proj = a_face * sin_rel + a_edge * cos_rel
            return float(proj)
        else:
            # Tracking about roll axis: array normal is perpendicular to velocity vector
            return float(a_edge)

    if attitude_mode in (AttitudeMode.LVLH_SUN_TRACKING_2AXIS, AttitudeMode.SUN_POINTING):
        # 2-DOF tracking points array normal directly at Sun vector
        # v_hat . s_hat = -cos(beta) * sin(theta_rel)
        cos_beta = np.cos(beta_rad)
        sin_rel = np.sin(theta_rel)
        alignment = abs(cos_beta * sin_rel)
        ortho = np.sqrt(max(0.0, 1.0 - alignment**2))
        proj = a_face * alignment + a_edge * ortho
        return float(proj)

    # Fallback default
    return float(a_face * abs(np.sin(theta_rel)) + a_edge * abs(np.cos(theta_rel)))


def compute_bus_cross_section(
    true_anomaly_rad: float,
    beta_rad: float,
    geometry: SpacecraftGeometry,
    attitude_mode: AttitudeMode = AttitudeMode.LVLH_SUN_TRACKING_1AXIS,
    nu_noon_rad: float = 0.0,
    pitch_offset_rad: float = 0.0,
    yaw_offset_rad: float = 0.0,
    roll_offset_rad: float = 0.0,
) -> float:
    """
    Calculate projected cross-sectional area of spacecraft bus perpendicular to velocity.

    Args:
        true_anomaly_rad: Orbital true anomaly (radians).
        beta_rad: Solar beta angle (radians).
        geometry: SpacecraftGeometry defining bus dimensions.
        attitude_mode: Spacecraft attitude mode.
        nu_noon_rad: Orbit noon true anomaly (radians).
        pitch_offset_rad: Body pitch offset angle (radians).
        yaw_offset_rad: Body yaw offset angle (radians).
        roll_offset_rad: Body roll offset angle (radians).

    Returns:
        Bus projected cross-sectional area in m^2.
    """
    if attitude_mode == AttitudeMode.FEATHERED:
        return geometry.bus_min_area_m2

    if attitude_mode == AttitudeMode.DRAG_MAXIMIZATION:
        return geometry.bus_max_area_m2

    if attitude_mode == AttitudeMode.SUN_POINTING:
        # Sun-pointing: body Z or X axis tracks Sun vector
        # Projection of cuboid onto velocity direction varies as angle between velocity and Sun shifts
        theta_rel = true_anomaly_rad - nu_noon_rad
        cos_sun_vel = abs(np.cos(beta_rad) * np.sin(theta_rel))
        sin_sun_vel = np.sqrt(max(0.0, 1.0 - cos_sun_vel**2))
        # Front face area (aligned with Sun) + side face area
        proj = (
            geometry.bus_top_bottom_area_m2 * cos_sun_vel
            + geometry.bus_frontal_area_m2 * sin_sun_vel
        )
        return float(proj)

    # Standard LVLH nadir-pointing attitude with optional attitude offsets:
    # Nominal ram facing area is width * height (frontal area)
    # When pitched or yawed, side and top/bottom faces project into flow
    cos_p = abs(np.cos(pitch_offset_rad))
    sin_p = abs(np.sin(pitch_offset_rad))
    cos_y = abs(np.cos(yaw_offset_rad))
    sin_y = abs(np.sin(yaw_offset_rad))

    proj_bus = (
        geometry.bus_frontal_area_m2 * (cos_p * cos_y)
        + geometry.bus_side_area_m2 * sin_y
        + geometry.bus_top_bottom_area_m2 * (sin_p * cos_y)
    )

    return float(max(1e-4, proj_bus))


def compute_sentman_plate_cd(
    incidence_angle_rad: float,
    speed_ratio: float = 7.0,
    t_wall_k: float = 300.0,
    t_inf_k: float = 1000.0,
) -> float:
    """
    Sentman free-molecular flow drag coefficient for a flat plate.

    Args:
        incidence_angle_rad: Angle between flow velocity and plate outward normal (radians).
        speed_ratio: Molecular speed ratio s = V / sqrt(2*R*T_inf), typically 6-8 in LEO.
        t_wall_k: Spacecraft wall surface temperature (Kelvin).
        t_inf_k: Ambient thermospheric kinetic temperature (Kelvin).

    Returns:
        Aerodynamic drag coefficient Cd (dimensionless, ~2.0 - 2.5).
    """
    cos_gamma = float(np.cos(incidence_angle_rad))
    if cos_gamma <= 0.0:
        return 2.0  # Backside/shielded baseline

    s = max(1.0, float(speed_ratio))
    p = s * cos_gamma

    # Hyperthermal approximation for s >> 1:
    term_diffuse = (np.sqrt(np.pi) / s) * np.sqrt(t_wall_k / t_inf_k) * (cos_gamma**2)
    cd = 2.0 + term_diffuse
    return float(np.clip(cd, 1.8, 3.2))


def compute_ballistic_coefficient(
    mass_kg: float,
    cd: float,
    total_area_m2: float,
) -> float:
    """
    Calculate ballistic coefficient BC = m / (Cd * A) in kg / m^2.

    Higher BC corresponds to higher orbital momentum and reduced atmospheric drag sensitivity.
    """
    if cd <= 0.0 or total_area_m2 <= 0.0:
        return float("inf")
    return float(mass_kg / (cd * total_area_m2))


def compute_instantaneous_drag_properties(
    true_anomaly_rad: float,
    beta_rad: float,
    geometry: SpacecraftGeometry,
    attitude_mode: AttitudeMode = AttitudeMode.LVLH_SUN_TRACKING_1AXIS,
    nu_noon_rad: float = 0.0,
    eclipse_feathering: bool = False,
    pitch_offset_rad: float = 0.0,
    yaw_offset_rad: float = 0.0,
    use_sentman_cd: bool = False,
) -> Dict[str, Any]:
    """
    Compute comprehensive cross-sectional areas, aerodynamic drag coefficient,
    and ballistic coefficient for a specific orbital position and solar geometry.

    Returns:
        Dictionary containing:
            - solar_array_area_m2: Solar array projected area
            - bus_area_m2: Spacecraft bus projected area
            - total_area_m2: Combined projected cross section
            - cd_effective: Area-weighted effective drag coefficient
            - drag_area_m2: Cd * A_total (effective drag product)
            - ballistic_coefficient_kg_m2: BC = m / (Cd * A_total)
            - is_eclipse: Boolean flag indicating if in Earth shadow
            - beta_deg: Solar beta angle in degrees
            - true_anomaly_deg: True anomaly in degrees
    """
    a_sa = compute_solar_array_cross_section(
        true_anomaly_rad=true_anomaly_rad,
        beta_rad=beta_rad,
        geometry=geometry,
        attitude_mode=attitude_mode,
        nu_noon_rad=nu_noon_rad,
        eclipse_feathering=eclipse_feathering,
    )

    a_bus = compute_bus_cross_section(
        true_anomaly_rad=true_anomaly_rad,
        beta_rad=beta_rad,
        geometry=geometry,
        attitude_mode=attitude_mode,
        nu_noon_rad=nu_noon_rad,
        pitch_offset_rad=pitch_offset_rad,
        yaw_offset_rad=yaw_offset_rad,
    )

    total_area = a_sa + a_bus

    if use_sentman_cd:
        theta_rel = true_anomaly_rad - nu_noon_rad
        cd_sa = compute_sentman_plate_cd(incidence_angle_rad=abs(np.sin(theta_rel)))
        cd_bus = compute_sentman_plate_cd(incidence_angle_rad=pitch_offset_rad)
    else:
        cd_sa = geometry.cd_solar_array
        cd_bus = geometry.cd_bus

    # Area-weighted drag coefficient
    if total_area > 0.0:
        cd_eff = (cd_sa * a_sa + cd_bus * a_bus) / total_area
    else:
        cd_eff = geometry.cd_bus

    drag_area = cd_eff * total_area
    bc = compute_ballistic_coefficient(geometry.mass_kg, cd_eff, total_area)
    in_shadow = is_in_eclipse(true_anomaly_rad, beta_rad, nu_noon_rad=nu_noon_rad)

    return {
        "solar_array_area_m2": float(a_sa),
        "bus_area_m2": float(a_bus),
        "total_area_m2": float(total_area),
        "cd_effective": float(cd_eff),
        "drag_area_m2": float(drag_area),
        "ballistic_coefficient_kg_m2": float(bc),
        "is_eclipse": in_shadow,
        "beta_deg": float(np.degrees(beta_rad)),
        "true_anomaly_deg": float(np.degrees(true_anomaly_rad) % 360.0),
    }


def compute_orbit_profile(
    beta_deg: float,
    geometry: Optional[SpacecraftGeometry] = None,
    attitude_mode: AttitudeMode = AttitudeMode.LVLH_SUN_TRACKING_1AXIS,
    num_points: int = 360,
    nu_noon_deg: float = 0.0,
    eclipse_feathering: bool = False,
    use_sentman_cd: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Generate a full one-orbit profile (0 to 360 degrees true anomaly) of drag area and ballistic coefficient.

    Args:
        beta_deg: Solar beta angle (degrees).
        geometry: SpacecraftGeometry (defaults to standard 150kg smallsat).
        attitude_mode: Spacecraft attitude mode.
        num_points: Number of evaluation points along the orbit.
        nu_noon_deg: True anomaly of orbit noon (degrees).
        eclipse_feathering: Enable eclipse feathering.
        use_sentman_cd: Use Sentman free-molecular Cd model.

    Returns:
        Dictionary of numpy arrays for true anomaly, solar array area, bus area,
        total area, Cd, drag area, ballistic coefficient, and eclipse flags.
    """
    if geometry is None:
        geometry = SpacecraftGeometry()

    beta_rad = np.radians(beta_deg)
    nu_noon_rad = np.radians(nu_noon_deg)
    true_anomalies_rad = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    sa_areas = np.zeros(num_points, dtype=float)
    bus_areas = np.zeros(num_points, dtype=float)
    total_areas = np.zeros(num_points, dtype=float)
    cds = np.zeros(num_points, dtype=float)
    drag_areas = np.zeros(num_points, dtype=float)
    bcs = np.zeros(num_points, dtype=float)
    eclipse_mask = np.zeros(num_points, dtype=bool)

    for i, nu in enumerate(true_anomalies_rad):
        props = compute_instantaneous_drag_properties(
            true_anomaly_rad=nu,
            beta_rad=beta_rad,
            geometry=geometry,
            attitude_mode=attitude_mode,
            nu_noon_rad=nu_noon_rad,
            eclipse_feathering=eclipse_feathering,
            use_sentman_cd=use_sentman_cd,
        )
        sa_areas[i] = props["solar_array_area_m2"]
        bus_areas[i] = props["bus_area_m2"]
        total_areas[i] = props["total_area_m2"]
        cds[i] = props["cd_effective"]
        drag_areas[i] = props["drag_area_m2"]
        bcs[i] = props["ballistic_coefficient_kg_m2"]
        eclipse_mask[i] = props["is_eclipse"]

    return {
        "true_anomaly_deg": np.degrees(true_anomalies_rad),
        "solar_array_area_m2": sa_areas,
        "bus_area_m2": bus_areas,
        "total_area_m2": total_areas,
        "cd_effective": cds,
        "drag_area_m2": drag_areas,
        "ballistic_coefficient_kg_m2": bcs,
        "is_eclipse": eclipse_mask,
    }


def compute_orbit_averaged_metrics(
    beta_deg: float,
    geometry: Optional[SpacecraftGeometry] = None,
    attitude_mode: AttitudeMode = AttitudeMode.LVLH_SUN_TRACKING_1AXIS,
    num_points: int = 360,
    eclipse_feathering: bool = False,
    altitude_km: float = 500.0,
) -> Dict[str, float]:
    """
    Calculate orbit-averaged aerodynamic properties and effective ballistic coefficient.

    Formula:
        mean_area = 1/(2*pi) * integral(A(nu) dnu)
        mean_drag_area = 1/(2*pi) * integral(Cd(nu)*A(nu) dnu)
        effective_bc = mass / mean_drag_area

    Returns:
        Dictionary of orbit-averaged metrics.
    """
    if geometry is None:
        geometry = SpacecraftGeometry()

    profile = compute_orbit_profile(
        beta_deg=beta_deg,
        geometry=geometry,
        attitude_mode=attitude_mode,
        num_points=num_points,
        eclipse_feathering=eclipse_feathering,
    )

    mean_total_area = float(np.mean(profile["total_area_m2"]))
    min_total_area = float(np.min(profile["total_area_m2"]))
    max_total_area = float(np.max(profile["total_area_m2"]))
    mean_sa_area = float(np.mean(profile["solar_array_area_m2"]))
    mean_bus_area = float(np.mean(profile["bus_area_m2"]))
    mean_drag_area = float(np.mean(profile["drag_area_m2"]))
    mean_cd = float(np.mean(profile["cd_effective"]))

    effective_bc = float(geometry.mass_kg / mean_drag_area) if mean_drag_area > 0.0 else float("inf")
    mean_bc = float(np.mean(profile["ballistic_coefficient_kg_m2"]))
    eclipse_frac = compute_eclipse_fraction(np.radians(beta_deg), altitude_km=altitude_km)

    return {
        "beta_deg": float(beta_deg),
        "mean_total_area_m2": mean_total_area,
        "min_total_area_m2": min_total_area,
        "max_total_area_m2": max_total_area,
        "mean_solar_array_area_m2": mean_sa_area,
        "mean_bus_area_m2": mean_bus_area,
        "mean_cd": mean_cd,
        "mean_drag_area_m2": mean_drag_area,
        "effective_ballistic_coefficient_kg_m2": effective_bc,
        "mean_ballistic_coefficient_kg_m2": mean_bc,
        "eclipse_fraction": eclipse_frac,
    }


def compute_beta_angle_sweep(
    beta_min_deg: float = -90.0,
    beta_max_deg: float = 90.0,
    num_steps: int = 91,
    geometry: Optional[SpacecraftGeometry] = None,
    attitude_mode: AttitudeMode = AttitudeMode.LVLH_SUN_TRACKING_1AXIS,
    eclipse_feathering: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Perform a parametric sweep over solar beta angles to determine variation in effective ballistic coefficient.

    Args:
        beta_min_deg: Minimum beta angle in degrees (-90.0).
        beta_max_deg: Maximum beta angle in degrees (+90.0).
        num_steps: Number of beta angle points.
        geometry: SpacecraftGeometry.
        attitude_mode: Spacecraft attitude mode.
        eclipse_feathering: Enable eclipse feathering.

    Returns:
        Dictionary containing arrays of beta angles, effective BCs, mean areas, and eclipse fractions.
    """
    if geometry is None:
        geometry = SpacecraftGeometry()

    betas = np.linspace(beta_min_deg, beta_max_deg, num_steps)
    effective_bcs = np.zeros(num_steps, dtype=float)
    mean_areas = np.zeros(num_steps, dtype=float)
    eclipse_fractions = np.zeros(num_steps, dtype=float)

    for i, b in enumerate(betas):
        metrics = compute_orbit_averaged_metrics(
            beta_deg=float(b),
            geometry=geometry,
            attitude_mode=attitude_mode,
            eclipse_feathering=eclipse_feathering,
        )
        effective_bcs[i] = metrics["effective_ballistic_coefficient_kg_m2"]
        mean_areas[i] = metrics["mean_total_area_m2"]
        eclipse_fractions[i] = metrics["eclipse_fraction"]

    return {
        "beta_deg": betas,
        "effective_ballistic_coefficient_kg_m2": effective_bcs,
        "mean_total_area_m2": mean_areas,
        "eclipse_fraction": eclipse_fractions,
    }


def compute_variable_attitude_drag_acceleration(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    sun_eci_km: np.ndarray,
    geometry: Optional[SpacecraftGeometry] = None,
    attitude_mode: AttitudeMode = AttitudeMode.LVLH_SUN_TRACKING_1AXIS,
    kp: float = QUIET_KP,
    f10_7: float = QUIET_F10_7,
    eclipse_feathering: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute high-precision instantaneous aerodynamic drag acceleration vector in ECI (km/s^2)
    accounting for variable attitude cross section, solar beta angle, and orbital true anomaly.

    Formula:
        a_drag = - 0.5 * rho(h, Kp, F10.7) * (Cd * A(nu, beta) / m) * v_rel^2 * v_rel_hat

    Args:
        r_eci_km: Spacecraft position in ECI (km).
        v_eci_km_s: Spacecraft inertial velocity in ECI (km/s).
        sun_eci_km: Sun position in ECI (km).
        geometry: SpacecraftGeometry.
        attitude_mode: AttitudeMode.
        kp: Geomagnetic Kp index.
        f10_7: Solar 10.7 cm radio flux (SFU).
        eclipse_feathering: Feather arrays edge-on during eclipse.

    Returns:
        Tuple of (drag_acceleration_vector_km_s2, telemetry_dict).
    """
    if geometry is None:
        geometry = SpacecraftGeometry()

    r_mag = float(np.linalg.norm(r_eci_km))
    altitude_km = r_mag - EARTH_RADIUS_KM

    if altitude_km > 1200.0 or geometry.mass_kg <= 0.0:
        return np.zeros(3, dtype=float), {"status": "above_atmosphere"}

    # Compute solar beta angle
    beta_rad = compute_solar_beta_angle_from_vectors(r_eci_km, v_eci_km_s, sun_eci_km)

    # Compute orbit normal and in-plane projection of Sun vector to determine relative true anomaly
    h = np.cross(r_eci_km, v_eci_km_s)
    h_norm = np.linalg.norm(h)
    h_hat = h / h_norm if h_norm > 1e-9 else np.array([0.0, 0.0, 1.0])

    s_norm = np.linalg.norm(sun_eci_km)
    s_hat = sun_eci_km / s_norm if s_norm > 1e-9 else np.array([1.0, 0.0, 0.0])

    # In-plane Sun vector
    s_inplane = s_hat - np.dot(s_hat, h_hat) * h_hat
    s_inplane_mag = np.linalg.norm(s_inplane)

    if s_inplane_mag > 1e-6:
        u_sun_inplane = s_inplane / s_inplane_mag
    else:
        # Near beta = +-90 deg: choose arbitrary in-plane reference
        r_hat = r_eci_km / r_mag
        u_sun_inplane = r_hat

    # Satellite in-plane position unit vector
    r_hat = r_eci_km / r_mag
    # Relative angle from orbit noon:
    cos_theta = np.clip(np.dot(r_hat, u_sun_inplane), -1.0, 1.0)
    sin_theta = np.dot(np.cross(u_sun_inplane, r_hat), h_hat)
    theta_rel = float(np.arctan2(sin_theta, cos_theta))

    # Evaluate instantaneous cross section & ballistic coefficient
    drag_props = compute_instantaneous_drag_properties(
        true_anomaly_rad=theta_rel,
        beta_rad=beta_rad,
        geometry=geometry,
        attitude_mode=attitude_mode,
        nu_noon_rad=0.0,
        eclipse_feathering=eclipse_feathering,
    )

    # Relative velocity to rotating atmosphere (km/s)
    v_rel_km_s = relative_atmospheric_velocity(r_eci_km, v_eci_km_s)
    v_rel_mag_km_s = float(np.linalg.norm(v_rel_km_s))

    if v_rel_mag_km_s < 1e-9:
        return np.zeros(3, dtype=float), drag_props

    v_rel_unit = v_rel_km_s / v_rel_mag_km_s
    v_rel_m_s = v_rel_mag_km_s * 1000.0

    # Thermospheric atmospheric density (kg/m^3)
    rho = compute_atmospheric_density(altitude_km, kp, f10_7)

    # Drag acceleration: a_drag = 0.5 * rho * (Cd * A / m) * v_rel^2 in m/s^2
    # drag_props["drag_area_m2"] = Cd * A
    inv_bc = drag_props["drag_area_m2"] / geometry.mass_kg  # m^2 / kg
    a_drag_m_s2 = 0.5 * rho * inv_bc * (v_rel_m_s ** 2)

    # Convert to km/s^2 opposing relative velocity vector
    a_drag_km_s2 = (a_drag_m_s2 / 1000.0) * (-v_rel_unit)

    drag_props["altitude_km"] = altitude_km
    drag_props["atmospheric_density_kg_m3"] = rho
    drag_props["drag_accel_mag_m_s2"] = float(a_drag_m_s2)

    return a_drag_km_s2, drag_props
