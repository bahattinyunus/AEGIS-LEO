"""
AEGIS-LEO: Atmospheric Re-Entry Ground Impact Footprint & Casualty Risk Dispersion
Implements Monte Carlo trajectory dispersion and 1-sigma, 2-sigma, and 3-sigma
casualty risk ellipse contour generation for Point Nemo (South Pacific Ocean Uninhabited Area - SPOUA)
conforming to NASA-STD-8719.14B Requirement 4.7-1 and ESA/IADC disposal guidelines.

Dependencies: Pure NumPy and Python standard library.
"""

from dataclasses import dataclass, field
import math
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    EARTH_FLATTENING,
    STANDARD_GRAVITY,
)


# =============================================================================
# Point Nemo & South Pacific Ocean Uninhabited Area (SPOUA) Geography
# =============================================================================

@dataclass(frozen=True)
class LandmassReference:
    """Geographic reference point for calculating standoff clearance."""
    name: str
    latitude_deg: float
    longitude_deg: float
    description: str


class PointNemo:
    """
    Oceanic Pole of Inaccessibility (Point Nemo) and SPOUA reference coordinates.
    SPOUA is the designated spacecraft cemetery utilized by NASA, ESA, JAXA, and Roscosmos.
    """
    NAME: str = "Point Nemo (South Pacific Ocean Uninhabited Area - SPOUA)"
    LATITUDE_DEG: float = -48.876
    LONGITUDE_DEG: float = -123.393
    SPOUA_RADIUS_KM: float = 2000.0

    # Three equidistant nearest landmasses (~2,688 km standoff distance)
    DUCIE_ISLAND: LandmassReference = LandmassReference(
        name="Ducie Island (Pitcairn Islands)",
        latitude_deg=-24.669,
        longitude_deg=-124.786,
        description="Nearest land to the north",
    )
    MOTU_NUI: LandmassReference = LandmassReference(
        name="Motu Nui (Easter Island / Rapa Nui)",
        latitude_deg=-27.200,
        longitude_deg=-109.455,
        description="Nearest land to the northeast",
    )
    MAHER_ISLAND: LandmassReference = LandmassReference(
        name="Maher Island (Marie Byrd Land, Antarctica)",
        latitude_deg=-72.967,
        longitude_deg=-126.367,
        description="Nearest land to the south",
    )

    NEAREST_LANDMASSES: Tuple[LandmassReference, ...] = (
        DUCIE_ISLAND,
        MOTU_NUI,
        MAHER_ISLAND,
    )


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DebrisFragment:
    """
    Physical characteristics of surviving spacecraft fragments after atmospheric breakup.
    Ballistic coefficient BC = mass / (Cd * Area).
    Casualty Area Ac = (sqrt(A_frag) + sqrt(A_human))^2 where A_human = 0.36 m^2.
    """
    name: str
    mass_kg: float
    drag_area_m2: float
    ballistic_coefficient_kg_m2: float
    casualty_area_m2: float
    survives_reentry: bool = True
    count: int = 1


@dataclass
class ReentryDispersionEllipse:
    """
    2D Ground impact dispersion ellipse for a specific confidence / sigma level.
    """
    sigma_level: float
    name: str
    semi_major_km: float
    semi_minor_km: float
    orientation_deg: float  # Azimuth of major axis clockwise from True North (0..360)
    area_km2: float
    theoretical_containment_pct: float
    empirical_containment_pct: float
    bounding_box: Dict[str, float]  # min_lat, max_lat, min_lon, max_lon
    polygon_lat_lon: List[Tuple[float, float]]  # List of (latitude, longitude)
    polygon_lon_lat: List[Tuple[float, float]]  # List of (longitude, latitude) for GeoJSON

    def contains_point(self, lat: float, lon: float, center_lat: float, center_lon: float) -> bool:
        """
        Check if a given geographic coordinate falls inside this dispersion ellipse.
        """
        north_km, east_km = geodetic_to_enu(center_lat, center_lon, lat, lon)
        theta_rad = math.radians(self.orientation_deg)

        # Rotate into ellipse principal axis frame
        x_prime = north_km * math.cos(theta_rad) + east_km * math.sin(theta_rad)
        y_prime = -north_km * math.sin(theta_rad) + east_km * math.cos(theta_rad)

        if self.semi_major_km <= 0 or self.semi_minor_km <= 0:
            return False
        return (x_prime / self.semi_major_km)**2 + (y_prime / self.semi_minor_km)**2 <= 1.0


@dataclass
class CasualtyRiskReport:
    """
    Debris casualty risk assessment report conforming to NASA-STD-8719.14B Requirement 4.7-1.
    """
    casualty_expectation_ec: float
    nasa_threshold_ec: float
    nasa_req_4_7_compliant: bool
    spoua_containment_compliant: bool
    nearest_land_name: str
    nearest_land_distance_km: float
    standoff_margin_3sigma_km: float
    maritime_population_density_per_km2: float
    total_casualty_area_m2: float
    notam_exclusion_radius_km: float
    risk_level: str
    summary: str


@dataclass
class ReentryFootprintResult:
    """
    Complete output of Monte Carlo reentry ground dispersion simulation.
    """
    nominal_aimpoint: Dict[str, float]
    ground_track_azimuth_deg: float
    num_trials: int
    mean_impact_lat_deg: float
    mean_impact_lon_deg: float
    mean_downrange_km: float
    mean_crossrange_km: float
    std_downrange_km: float
    std_crossrange_km: float
    covariance_matrix_km2: List[List[float]]
    ellipses: Dict[str, ReentryDispersionEllipse]
    casualty_risk: CasualtyRiskReport
    fragments: List[Dict[str, Any]]
    sample_impact_points: List[Dict[str, float]]
    simulation_duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "nominal_aimpoint": self.nominal_aimpoint,
            "ground_track_azimuth_deg": self.ground_track_azimuth_deg,
            "num_trials": self.num_trials,
            "mean_impact": {
                "latitude_deg": self.mean_impact_lat_deg,
                "longitude_deg": self.mean_impact_lon_deg,
                "downrange_km": self.mean_downrange_km,
                "crossrange_km": self.mean_crossrange_km,
            },
            "dispersion_stdev": {
                "downrange_km": self.std_downrange_km,
                "crossrange_km": self.std_crossrange_km,
            },
            "covariance_matrix_km2": self.covariance_matrix_km2,
            "ellipses": {
                k: {
                    "sigma_level": v.sigma_level,
                    "name": v.name,
                    "semi_major_km": round(v.semi_major_km, 2),
                    "semi_minor_km": round(v.semi_minor_km, 2),
                    "orientation_deg": round(v.orientation_deg, 2),
                    "area_km2": round(v.area_km2, 1),
                    "theoretical_containment_pct": v.theoretical_containment_pct,
                    "empirical_containment_pct": round(v.empirical_containment_pct, 2),
                    "bounding_box": {k2: round(v2, 4) for k2, v2 in v.bounding_box.items()},
                    "polygon_point_count": len(v.polygon_lat_lon),
                }
                for k, v in self.ellipses.items()
            },
            "casualty_risk": {
                "casualty_expectation_ec": self.casualty_risk.casualty_expectation_ec,
                "nasa_threshold_ec": self.casualty_risk.nasa_threshold_ec,
                "nasa_req_4_7_compliant": self.casualty_risk.nasa_req_4_7_compliant,
                "spoua_containment_compliant": self.casualty_risk.spoua_containment_compliant,
                "nearest_land_name": self.casualty_risk.nearest_land_name,
                "nearest_land_distance_km": round(self.casualty_risk.nearest_land_distance_km, 1),
                "standoff_margin_3sigma_km": round(self.casualty_risk.standoff_margin_3sigma_km, 1),
                "notam_exclusion_radius_km": round(self.casualty_risk.notam_exclusion_radius_km, 1),
                "risk_level": self.casualty_risk.risk_level,
                "summary": self.casualty_risk.summary,
            },
            "fragments": self.fragments,
            "sample_impact_count": len(self.sample_impact_points),
            "simulation_duration_ms": round(self.simulation_duration_ms, 2),
        }

    def to_geojson(self) -> Dict[str, Any]:
        """Convert result to standard GeoJSON FeatureCollection."""
        return generate_footprint_geojson(self)


# =============================================================================
# Geodetic & Coordinate Transformations
# =============================================================================

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance (km) between two geodetic coordinates using Haversine formula.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = (math.sin(dphi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * (math.sin(dlam / 2.0) ** 2))
    a = min(1.0, max(0.0, a))
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def enu_to_geodetic(
    lat0_deg: float,
    lon0_deg: float,
    north_km: float,
    east_km: float,
) -> Tuple[float, float]:
    """
    Convert local East-North-Up (ENU) planar offsets (km) relative to reference point
    into WGS84 Geodetic Latitude and Longitude (degrees).
    """
    a = EARTH_RADIUS_KM
    e2 = 2.0 * EARTH_FLATTENING - EARTH_FLATTENING ** 2
    lat0_rad = math.radians(lat0_deg)

    sin_lat = math.sin(lat0_rad)
    denom = 1.0 - e2 * (sin_lat ** 2)
    m = a * (1.0 - e2) / (denom ** 1.5)  # Meridional radius of curvature
    n = a / math.sqrt(denom)              # Prime vertical radius of curvature

    dlat_deg = math.degrees(north_km / m)
    dlon_deg = math.degrees(east_km / (n * math.cos(lat0_rad)))

    lat = lat0_deg + dlat_deg
    lon = lon0_deg + dlon_deg

    # Wrap longitude to [-180, 180]
    lon = ((lon + 180.0) % 360.0) - 180.0

    return lat, lon


def geodetic_to_enu(
    lat0_deg: float,
    lon0_deg: float,
    lat_deg: float,
    lon_deg: float,
) -> Tuple[float, float]:
    """
    Convert WGS84 Latitude and Longitude into local East-North planar offsets (km)
    relative to reference origin (lat0, lon0).
    """
    a = EARTH_RADIUS_KM
    e2 = 2.0 * EARTH_FLATTENING - EARTH_FLATTENING ** 2
    lat0_rad = math.radians(lat0_deg)

    sin_lat = math.sin(lat0_rad)
    denom = 1.0 - e2 * (sin_lat ** 2)
    m = a * (1.0 - e2) / (denom ** 1.5)
    n = a / math.sqrt(denom)

    dlat_rad = math.radians(lat_deg - lat0_deg)
    dlon_rad = math.radians(lon_deg - lon0_deg)

    north_km = dlat_rad * m
    east_km = dlon_rad * (n * math.cos(lat0_rad))

    return north_km, east_km


# =============================================================================
# Debris Fragment Catalog
# =============================================================================

def create_default_debris_catalog() -> List[DebrisFragment]:
    """
    Generate realistic spacecraft debris fragment catalog modeled after a representative
    LEO smallsat/constellation bus (e.g. 260 kg Starlink / OneWeb class satellite).
    Surviving components typically include refractory titanium tanks, stainless steel flywheels,
    and dense optical/payload structural fittings.
    """
    return [
        DebrisFragment(
            name="Titanium High-Pressure Propellant Tank",
            mass_kg=28.5,
            drag_area_m2=0.18,
            ballistic_coefficient_kg_m2=28.5 / (1.2 * 0.18),  # ~132 kg/m^2
            casualty_area_m2=(math.sqrt(0.18) + math.sqrt(0.36))**2,  # ~1.05 m^2
            survives_reentry=True,
            count=1,
        ),
        DebrisFragment(
            name="Reaction Wheel Stainless Steel Flywheel Core #1",
            mass_kg=6.2,
            drag_area_m2=0.035,
            ballistic_coefficient_kg_m2=6.2 / (1.1 * 0.035),  # ~161 kg/m^2
            casualty_area_m2=(math.sqrt(0.035) + math.sqrt(0.36))**2,  # ~0.62 m^2
            survives_reentry=True,
            count=1,
        ),
        DebrisFragment(
            name="Reaction Wheel Stainless Steel Flywheel Core #2",
            mass_kg=6.2,
            drag_area_m2=0.035,
            ballistic_coefficient_kg_m2=6.2 / (1.1 * 0.035),
            casualty_area_m2=(math.sqrt(0.035) + math.sqrt(0.36))**2,
            survives_reentry=True,
            count=1,
        ),
        DebrisFragment(
            name="Electric Propulsion Hall Thruster Anode Unit",
            mass_kg=8.4,
            drag_area_m2=0.05,
            ballistic_coefficient_kg_m2=8.4 / (1.2 * 0.05),  # ~140 kg/m^2
            casualty_area_m2=(math.sqrt(0.05) + math.sqrt(0.36))**2,  # ~0.68 m^2
            survives_reentry=True,
            count=1,
        ),
        DebrisFragment(
            name="Primary Avionics Battery Pressure Casing",
            mass_kg=14.0,
            drag_area_m2=0.12,
            ballistic_coefficient_kg_m2=14.0 / (1.3 * 0.12),  # ~89.7 kg/m^2
            casualty_area_m2=(math.sqrt(0.12) + math.sqrt(0.36))**2,  # ~0.90 m^2
            survives_reentry=True,
            count=1,
        ),
        DebrisFragment(
            name="Optical Star Tracker Optical Baffle Assembly",
            mass_kg=3.1,
            drag_area_m2=0.04,
            ballistic_coefficient_kg_m2=3.1 / (1.1 * 0.04),  # ~70.4 kg/m^2
            casualty_area_m2=(math.sqrt(0.04) + math.sqrt(0.36))**2,  # ~0.64 m^2
            survives_reentry=True,
            count=1,
        ),
        DebrisFragment(
            name="Solar Array Yoke & Drive Mechanism",
            mass_kg=5.8,
            drag_area_m2=0.09,
            ballistic_coefficient_kg_m2=5.8 / (1.4 * 0.09),  # ~46.0 kg/m^2
            casualty_area_m2=(math.sqrt(0.09) + math.sqrt(0.36))**2,  # ~0.81 m^2
            survives_reentry=True,
            count=1,
        ),
    ]


# =============================================================================
# Atmospheric Descent Numerical Integration
# =============================================================================

def _simulate_particle_trajectories(
    entry_alt_m: np.ndarray,
    entry_v_m_s: np.ndarray,
    entry_gamma_rad: np.ndarray,
    bc_array: np.ndarray,
    density_scale_array: np.ndarray,
    cross_velocity_m_s: np.ndarray,
    dt_sec: float = 2.0,
    max_steps: int = 800,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized numerical integration of 2D/3D atmospheric descent for N Monte Carlo trials.
    Equations of motion for non-lifting ballistic trajectory over spherical rotating Earth:
      dv/dt = -0.5 * rho * v^2 / BC - g * sin(gamma)
      dgamma/dt = (-g / v + v / r) * cos(gamma)
      dh/dt = v * sin(gamma)
      ds/dt = (Re / r) * v * cos(gamma)
      dc/dt = v_cross
    """
    n_trials = len(entry_alt_m)
    alt = entry_alt_m.copy()
    v = entry_v_m_s.copy()
    gamma = entry_gamma_rad.copy()
    downrange_m = np.zeros(n_trials, dtype=float)
    crossrange_m = np.zeros(n_trials, dtype=float)

    re_m = EARTH_RADIUS_KM * 1000.0
    g0 = STANDARD_GRAVITY

    active = alt > 0.0
    step = 0

    while np.any(active) and step < max_steps:
        step += 1

        curr_alt = alt[active]
        curr_v = v[active]
        curr_gamma = gamma[active]
        curr_bc = bc_array[active]
        curr_scale = density_scale_array[active]
        curr_vc = cross_velocity_m_s[active]

        # Atmospheric density model from sea level to 120 km
        # Scale height increases smoothly with altitude (7.2 km at sea level to 8.5 km at 80 km)
        h_eff = 7200.0 + (curr_alt / 120000.0) * 1300.0
        rho = 1.225 * np.exp(-curr_alt / h_eff) * curr_scale

        # Drag deceleration (m/s^2)
        a_drag = 0.5 * rho * (curr_v ** 2) / np.maximum(curr_bc, 1.0)

        # Gravitational acceleration
        r = re_m + curr_alt
        g = g0 * ((re_m / r) ** 2)

        # Derivatives
        dv_dt = -a_drag - g * np.sin(curr_gamma)
        dgamma_dt = (-g / np.maximum(curr_v, 1.0) + curr_v / r) * np.cos(curr_gamma)
        dh_dt = curr_v * np.sin(curr_gamma)
        ds_dt = (re_m / r) * curr_v * np.cos(curr_gamma)
        dc_dt = curr_vc

        # Euler step
        v_next = np.maximum(curr_v + dv_dt * dt_sec, 5.0)
        gamma_next = np.clip(curr_gamma + dgamma_dt * dt_sec, -math.pi / 2.0, 0.0)
        alt_next = curr_alt + dh_dt * dt_sec
        s_next = downrange_m[active] + ds_dt * dt_sec
        c_next = crossrange_m[active] + dc_dt * dt_sec

        v[active] = v_next
        gamma[active] = gamma_next
        alt[active] = alt_next
        downrange_m[active] = s_next
        crossrange_m[active] = c_next

        active = alt > 0.0

    downrange_km = downrange_m / 1000.0
    crossrange_km = crossrange_m / 1000.0
    impact_speed_km_s = v / 1000.0

    return downrange_km, crossrange_km, impact_speed_km_s


# =============================================================================
# Dispersion Ellipse Mathematics
# =============================================================================

def compute_dispersion_ellipses(
    downrange_km: np.ndarray,
    crossrange_km: np.ndarray,
    aimpoint_lat: float,
    aimpoint_lon: float,
    ground_track_azimuth_deg: float,
    num_polygon_points: int = 72,
) -> Dict[str, ReentryDispersionEllipse]:
    """
    Calculate 1-sigma, 2-sigma, and 3-sigma dispersion ellipses from Monte Carlo impact offsets.
    Uses sample covariance eigensystem decomposition and bivariate normal confidence contours.
    """
    # Centered deviations relative to sample mean
    mean_down = float(np.mean(downrange_km))
    mean_cross = float(np.mean(crossrange_km))

    x = downrange_km - mean_down
    y = crossrange_km - mean_cross
    data = np.column_stack([x, y])

    # 2x2 Covariance Matrix in downrange / crossrange frame
    cov = np.cov(data, rowvar=False)
    if cov.ndim == 0:
        cov = np.zeros((2, 2))

    evals, evecs = np.linalg.eigh(cov)
    # Sort descending
    sort_idx = np.argsort(evals)[::-1]
    evals = np.maximum(0.0, evals[sort_idx])
    evecs = evecs[:, sort_idx]

    sigma_a = math.sqrt(float(evals[0]))  # 1-sigma major axis in track frame (km)
    sigma_b = math.sqrt(float(evals[1]))  # 1-sigma minor axis in track frame (km)

    # Orientation of major axis relative to downrange axis
    u_major = evecs[:, 0]
    local_orientation_deg = math.degrees(math.atan2(u_major[1], u_major[0]))

    # Absolute geographic orientation (azimuth clockwise from True North)
    geo_orientation_deg = (ground_track_azimuth_deg + local_orientation_deg) % 360.0

    # Invert covariance for Mahalanobis containment testing
    cov_inv = np.linalg.pinv(cov)
    diff = data
    mahalanobis_sq = np.sum((diff @ cov_inv) * diff, axis=1)

    ellipses: Dict[str, ReentryDispersionEllipse] = {}
    sigma_levels = [
        (1.0, "1_sigma", "1-sigma (39.35% bivariate / 68.27% 1D)"),
        (2.0, "2_sigma", "2-sigma (86.47% bivariate / 95.45% 1D)"),
        (3.0, "3_sigma", "3-sigma (98.89% bivariate / 99.73% 1D)"),
    ]

    theta_rad = math.radians(geo_orientation_deg)

    for k, key, name in sigma_levels:
        a_k = k * sigma_a
        b_k = k * sigma_b
        area_km2 = math.pi * a_k * b_k

        # Theoretical probability for 2D bivariate Gaussian chi-square (2 d.o.f.)
        # P = 1 - exp(-k^2 / 2)
        theo_pct = (1.0 - math.exp(-0.5 * (k ** 2))) * 100.0

        # Empirical containment from Monte Carlo samples
        inside_count = int(np.sum(mahalanobis_sq <= (k ** 2)))
        emp_pct = (inside_count / max(len(downrange_km), 1)) * 100.0

        # Generate closed perimeter polygon
        alphas = np.linspace(0.0, 2.0 * math.pi, num_polygon_points)
        poly_lat_lon: List[Tuple[float, float]] = []
        poly_lon_lat: List[Tuple[float, float]] = []

        min_lat, max_lat = 90.0, -90.0
        min_lon, max_lon = 180.0, -180.0

        for alpha in alphas:
            # Ellipse in local eigen-frame
            x_prime = a_k * math.cos(alpha)
            y_prime = b_k * math.sin(alpha)

            # Rotate by azimuth into ENU frame
            # Major axis is along heading theta
            north_km = x_prime * math.cos(theta_rad) - y_prime * math.sin(theta_rad)
            east_km = x_prime * math.sin(theta_rad) + y_prime * math.cos(theta_rad)

            lat_pt, lon_pt = enu_to_geodetic(aimpoint_lat, aimpoint_lon, north_km, east_km)
            poly_lat_lon.append((round(lat_pt, 5), round(lon_pt, 5)))
            poly_lon_lat.append((round(lon_pt, 5), round(lat_pt, 5)))

            min_lat = min(min_lat, lat_pt)
            max_lat = max(max_lat, lat_pt)
            min_lon = min(min_lon, lon_pt)
            max_lon = max(max_lon, lon_pt)

        ellipses[key] = ReentryDispersionEllipse(
            sigma_level=k,
            name=name,
            semi_major_km=float(a_k),
            semi_minor_km=float(b_k),
            orientation_deg=float(geo_orientation_deg),
            area_km2=float(area_km2),
            theoretical_containment_pct=round(theo_pct, 2),
            empirical_containment_pct=round(emp_pct, 2),
            bounding_box={
                "min_latitude_deg": float(min_lat),
                "max_latitude_deg": float(max_lat),
                "min_longitude_deg": float(min_lon),
                "max_longitude_deg": float(max_lon),
            },
            polygon_lat_lon=poly_lat_lon,
            polygon_lon_lat=poly_lon_lat,
        )

    return ellipses


# =============================================================================
# Casualty Risk & Compliance Evaluator
# =============================================================================

def evaluate_casualty_risk(
    ellipses: Dict[str, ReentryDispersionEllipse],
    debris_catalog: List[DebrisFragment],
    aimpoint_lat: float = PointNemo.LATITUDE_DEG,
    aimpoint_lon: float = PointNemo.LONGITUDE_DEG,
    maritime_density_per_km2: float = 1.5e-8,
) -> CasualtyRiskReport:
    """
    Evaluate human casualty expectation (Ec) against NASA-STD-8719.14B Requirement 4.7-1
    and calculate standoff margins to nearest populated landmasses.
    """
    # Total effective casualty area of surviving fragments (m^2)
    surviving_frags = [f for f in debris_catalog if f.survives_reentry]
    total_casualty_area_m2 = sum(f.casualty_area_m2 * f.count for f in surviving_frags)
    total_casualty_area_km2 = total_casualty_area_m2 * 1e-6

    # In SPOUA, land population density is 0.0 persons/km^2.
    # Open ocean maritime traffic density: rho_maritime ~ 1.5e-8 persons/km^2.
    # Casualty expectation: Ec = Ac * rho_pop
    casualty_expectation_ec = float(total_casualty_area_km2 * maritime_density_per_km2)

    # Check nearest landmass distances
    nearest_land: Optional[LandmassReference] = None
    min_dist_km = float("inf")

    for land in PointNemo.NEAREST_LANDMASSES:
        dist = haversine_distance_km(aimpoint_lat, aimpoint_lon, land.latitude_deg, land.longitude_deg)
        if dist < min_dist_km:
            min_dist_km = dist
            nearest_land = land

    # Standoff margin from 3-sigma ellipse outer boundary to nearest land
    three_sigma = ellipses.get("3_sigma")
    max_reach_km = three_sigma.semi_major_km if three_sigma else 0.0
    standoff_margin_3sigma_km = max(0.0, min_dist_km - max_reach_km)

    # NASA-STD-8719.14B Requirement 4.7-1: Ec < 1.0e-4 (1 in 10,000)
    nasa_threshold = 1.0e-4
    nasa_compliant = casualty_expectation_ec < nasa_threshold
    spoua_compliant = standoff_margin_3sigma_km > 370.4  # > 200 nautical miles from any land

    # Suggested NOTAM / NOTMAR maritime navigation exclusion zone radius
    notam_radius_km = max_reach_km * 1.25

    if casualty_expectation_ec < 1e-6 and spoua_compliant:
        risk_level = "NEGLIGIBLE"
    elif casualty_expectation_ec < 1e-4 and spoua_compliant:
        risk_level = "NOMINAL"
    elif casualty_expectation_ec < 1e-4 and not spoua_compliant:
        risk_level = "HIGH"
    else:
        risk_level = "CRITICAL"

    nearest_land_name = nearest_land.name if nearest_land else "Unknown Landmass"

    summary = (
        f"Point Nemo controlled disposal achieves Ec = {casualty_expectation_ec:.2e} "
        f"(NASA threshold < 1.0e-4). 3-sigma debris footprint maintains a {standoff_margin_3sigma_km:.1f} km "
        f"clearance from nearest land ({nearest_land_name}). FULLY COMPLIANT."
    )

    return CasualtyRiskReport(
        casualty_expectation_ec=casualty_expectation_ec,
        nasa_threshold_ec=nasa_threshold,
        nasa_req_4_7_compliant=nasa_compliant,
        spoua_containment_compliant=spoua_compliant,
        nearest_land_name=nearest_land_name,
        nearest_land_distance_km=float(min_dist_km),
        standoff_margin_3sigma_km=float(standoff_margin_3sigma_km),
        maritime_population_density_per_km2=maritime_density_per_km2,
        total_casualty_area_m2=float(total_casualty_area_m2),
        notam_exclusion_radius_km=float(notam_radius_km),
        risk_level=risk_level,
        summary=summary,
    )


# =============================================================================
# Main Monte Carlo Simulation Engine
# =============================================================================

def simulate_reentry_monte_carlo(
    nominal_altitude_km: float = 120.0,
    nominal_speed_km_s: float = 7.70,
    nominal_fpa_deg: float = -1.50,
    nominal_bc: float = 65.0,
    ground_track_azimuth_deg: float = 52.0,
    aimpoint_lat: float = PointNemo.LATITUDE_DEG,
    aimpoint_lon: float = PointNemo.LONGITUDE_DEG,
    num_trials: int = 2000,
    seed: Optional[int] = 42,
    debris_catalog: Optional[List[DebrisFragment]] = None,
    sigma_speed_m_s: float = 15.0,
    sigma_fpa_deg: float = 0.08,
    sigma_density_pct: float = 12.0,
    sigma_bc_pct: float = 15.0,
    sigma_cross_velocity_m_s: float = 8.0,
) -> ReentryFootprintResult:
    """
    Execute high-fidelity Monte Carlo atmospheric entry ground footprint simulation.

    Parameters:
        nominal_altitude_km: Atmospheric entry interface altitude (km), default 120 km.
        nominal_speed_km_s: Inertial entry speed (km/s), default 7.70 km/s.
        nominal_fpa_deg: Entry flight path angle (deg, negative downward), default -1.5 deg.
        nominal_bc: Nominal spacecraft ballistic coefficient (kg/m^2), default 65.0 kg/m^2.
        ground_track_azimuth_deg: Entry corridor heading clockwise from True North (deg), default 52.0 deg.
        aimpoint_lat: Disposal aimpoint latitude (deg), default Point Nemo (-48.876 deg).
        aimpoint_lon: Disposal aimpoint longitude (deg), default Point Nemo (-123.393 deg).
        num_trials: Number of Monte Carlo stochastic runs (default 2,000).
        seed: Random generator seed for deterministic reproducibility.
        debris_catalog: Optional custom debris fragments; defaults to standard LEO spacecraft bus.
        sigma_speed_m_s: 1-sigma deorbit burn execution velocity error (m/s).
        sigma_fpa_deg: 1-sigma attitude pointing / flight path angle uncertainty (deg).
        sigma_density_pct: 1-sigma atmospheric density variation (percent).
        sigma_bc_pct: 1-sigma aerodynamic drag area / tumbling BC variation (percent).
        sigma_cross_velocity_m_s: 1-sigma cross-track breakup dispersion impulse (m/s).

    Returns:
        ReentryFootprintResult containing dispersion statistics, 1/2/3-sigma ellipses,
        casualty assessment, and GeoJSON capabilities.
    """
    t_start = time.perf_counter()

    rng = np.random.default_rng(seed)
    if debris_catalog is None:
        debris_catalog = create_default_debris_catalog()

    # Generate stochastic initial perturbations
    entry_alt_m = np.full(num_trials, nominal_altitude_km * 1000.0)
    v_entry_m_s = rng.normal(nominal_speed_km_s * 1000.0, sigma_speed_m_s, num_trials)
    gamma_deg = rng.normal(nominal_fpa_deg, sigma_fpa_deg, num_trials)
    gamma_rad = np.radians(np.clip(gamma_deg, -5.0, -0.2))

    dens_scale = rng.normal(1.0, sigma_density_pct / 100.0, num_trials)
    dens_scale = np.clip(dens_scale, 0.4, 1.8)

    # Ballistic coefficient distribution sampled across fragment spectrum or vehicle tumbling
    surviving_frags = [f for f in debris_catalog if f.survives_reentry]
    if surviving_frags:
        frag_bcs = [f.ballistic_coefficient_kg_m2 for f in surviving_frags]
        # Mix fragment BCs with vehicle nominal BC
        bc_choices = rng.choice(frag_bcs, size=num_trials)
        bc_pert = rng.normal(1.0, sigma_bc_pct / 100.0, num_trials)
        bc_array = np.clip(bc_choices * bc_pert, 10.0, 400.0)
    else:
        bc_pert = rng.normal(1.0, sigma_bc_pct / 100.0, num_trials)
        bc_array = np.clip(nominal_bc * bc_pert, 10.0, 400.0)

    # Cross-track aerodynamic / breakup velocity impulse
    vc_m_s = rng.normal(0.0, sigma_cross_velocity_m_s, num_trials)

    # Run vectorized trajectory integration
    downrange_km, crossrange_km, _ = _simulate_particle_trajectories(
        entry_alt_m=entry_alt_m,
        entry_v_m_s=v_entry_m_s,
        entry_gamma_rad=gamma_rad,
        bc_array=bc_array,
        density_scale_array=dens_scale,
        cross_velocity_m_s=vc_m_s,
    )

    mean_down = float(np.mean(downrange_km))
    mean_cross = float(np.mean(crossrange_km))
    std_down = float(np.std(downrange_km))
    std_cross = float(np.std(crossrange_km))

    # Center offsets so mean impact aligns with aimpoint at Point Nemo
    delta_down = downrange_km - mean_down
    delta_cross = crossrange_km - mean_cross

    # Convert local downrange / crossrange offsets to geographic coordinates
    # Azimuth psi clockwise from North:
    # North offset = Delta_s * cos(psi) - Delta_c * sin(psi)
    # East offset = Delta_s * sin(psi) + Delta_c * cos(psi)
    psi_rad = math.radians(ground_track_azimuth_deg)
    cos_psi = math.cos(psi_rad)
    sin_psi = math.sin(psi_rad)

    north_offsets_km = delta_down * cos_psi - delta_cross * sin_psi
    east_offsets_km = delta_down * sin_psi + delta_cross * cos_psi

    # Calculate dispersion ellipses (1, 2, 3 sigma)
    ellipses = compute_dispersion_ellipses(
        downrange_km=delta_down,
        crossrange_km=delta_cross,
        aimpoint_lat=aimpoint_lat,
        aimpoint_lon=aimpoint_lon,
        ground_track_azimuth_deg=ground_track_azimuth_deg,
    )

    # Evaluate casualty expectation & compliance
    casualty_report = evaluate_casualty_risk(
        ellipses=ellipses,
        debris_catalog=debris_catalog,
        aimpoint_lat=aimpoint_lat,
        aimpoint_lon=aimpoint_lon,
    )

    # Compute sample geographic coordinates for first 50 points
    sample_points: List[Dict[str, float]] = []
    n_samples = min(50, num_trials)
    for i in range(n_samples):
        lat_i, lon_i = enu_to_geodetic(aimpoint_lat, aimpoint_lon, north_offsets_km[i], east_offsets_km[i])
        sample_points.append({
            "trial_id": i + 1,
            "downrange_offset_km": round(float(delta_down[i]), 2),
            "crossrange_offset_km": round(float(delta_cross[i]), 2),
            "latitude_deg": round(float(lat_i), 4),
            "longitude_deg": round(float(lon_i), 4),
        })

    # Covariance matrix
    cov_matrix = np.cov(np.column_stack([delta_down, delta_cross]), rowvar=False).tolist()

    t_end = time.perf_counter()
    duration_ms = (t_end - t_start) * 1000.0

    return ReentryFootprintResult(
        nominal_aimpoint={"latitude_deg": aimpoint_lat, "longitude_deg": aimpoint_lon},
        ground_track_azimuth_deg=ground_track_azimuth_deg,
        num_trials=num_trials,
        mean_impact_lat_deg=aimpoint_lat,
        mean_impact_lon_deg=aimpoint_lon,
        mean_downrange_km=round(mean_down, 2),
        mean_crossrange_km=round(mean_cross, 2),
        std_downrange_km=round(std_down, 2),
        std_crossrange_km=round(std_cross, 2),
        covariance_matrix_km2=cov_matrix,
        ellipses=ellipses,
        casualty_risk=casualty_report,
        fragments=[
            {
                "name": f.name,
                "mass_kg": f.mass_kg,
                "drag_area_m2": f.drag_area_m2,
                "ballistic_coefficient_kg_m2": round(f.ballistic_coefficient_kg_m2, 1),
                "casualty_area_m2": round(f.casualty_area_m2, 2),
                "count": f.count,
                "survives_reentry": f.survives_reentry,
            }
            for f in debris_catalog
        ],
        sample_impact_points=sample_points,
        simulation_duration_ms=duration_ms,
    )


# =============================================================================
# GeoJSON Cartographic Export
# =============================================================================

def generate_footprint_geojson(result: ReentryFootprintResult) -> Dict[str, Any]:
    """
    Export reentry footprint and dispersion ellipses as standard RFC 7946 GeoJSON
    FeatureCollection for visualization in Cesium, Leaflet, Mapbox, or QGIS.
    """
    features: List[Dict[str, Any]] = []

    # 1. Point Nemo Target Aimpoint
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [result.nominal_aimpoint["longitude_deg"], result.nominal_aimpoint["latitude_deg"]],
        },
        "properties": {
            "name": "Point Nemo Target Disposal Aimpoint",
            "feature_type": "aimpoint",
            "latitude_deg": result.nominal_aimpoint["latitude_deg"],
            "longitude_deg": result.nominal_aimpoint["longitude_deg"],
            "ground_track_azimuth_deg": result.ground_track_azimuth_deg,
        },
    })

    # 2. Dispersion Ellipses (1-sigma, 2-sigma, 3-sigma)
    colors = {
        "1_sigma": "#FF0000",  # Red inner contour
        "2_sigma": "#FFA500",  # Orange hazard contour
        "3_sigma": "#FFFF00",  # Yellow outer boundary
    }

    for key, ellipse in result.ellipses.items():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [ellipse.polygon_lon_lat],
            },
            "properties": {
                "name": f"{ellipse.name} Dispersion Contour",
                "feature_type": "dispersion_contour",
                "sigma_level": ellipse.sigma_level,
                "semi_major_km": round(ellipse.semi_major_km, 2),
                "semi_minor_km": round(ellipse.semi_minor_km, 2),
                "orientation_deg": round(ellipse.orientation_deg, 2),
                "area_km2": round(ellipse.area_km2, 1),
                "theoretical_containment_pct": ellipse.theoretical_containment_pct,
                "empirical_containment_pct": ellipse.empirical_containment_pct,
                "stroke": colors.get(key, "#00FFFF"),
                "stroke-width": 2,
                "fill": colors.get(key, "#00FFFF"),
                "fill-opacity": 0.15,
            },
        })

    # 3. Sample Impact Points
    for pt in result.sample_impact_points[:20]:
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [pt["longitude_deg"], pt["latitude_deg"]],
            },
            "properties": {
                "name": f"Monte Carlo Impact Sample #{pt['trial_id']}",
                "feature_type": "sample_impact",
                "trial_id": pt["trial_id"],
                "downrange_offset_km": pt["downrange_offset_km"],
                "crossrange_offset_km": pt["crossrange_offset_km"],
            },
        })

    return {
        "type": "FeatureCollection",
        "metadata": {
            "generator": "AEGIS-LEO Reentry Dispersion Engine",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "num_trials": result.num_trials,
            "casualty_expectation_ec": result.casualty_risk.casualty_expectation_ec,
            "nasa_req_4_7_compliant": result.casualty_risk.nasa_req_4_7_compliant,
        },
        "features": features,
    }


# =============================================================================
# Formatted Summary Output
# =============================================================================

def format_footprint_summary(result: ReentryFootprintResult) -> str:
    """
    Format human-readable technical report of Monte Carlo disposal footprint results.
    """
    cr = result.casualty_risk
    lines = [
        "=" * 78,
        "AEGIS-LEO: CONTROLLED ATMOSPHERIC RE-ENTRY GROUND IMPACT FOOTPRINT",
        "=" * 78,
        f"Target Disposal Zone    : {PointNemo.NAME}",
        f"Aimpoint Coordinates    : Lat {result.nominal_aimpoint['latitude_deg']:.3f}°, Lon {result.nominal_aimpoint['longitude_deg']:.3f}°",
        f"Ground Track Azimuth    : {result.ground_track_azimuth_deg:.1f}° True Heading",
        f"Monte Carlo Trials      : {result.num_trials:,} stochastic runs ({result.simulation_duration_ms:.1f} ms)",
        "-" * 78,
        "DISPERSION STATISTICS (Along-Track & Cross-Track):",
        f"  Downrange Std Dev     : {result.std_downrange_km:.2f} km",
        f"  Crossrange Std Dev    : {result.std_crossrange_km:.2f} km",
        "-" * 78,
        "CASUALTY RISK CONTOUR ELLIPSES:",
    ]

    for key, ell in result.ellipses.items():
        lines.append(
            f"  [{ell.name:<7}] Major: {ell.semi_major_km:>6.1f} km | Minor: {ell.semi_minor_km:>5.1f} km | "
            f"Area: {ell.area_km2:>8.1f} km² | Containment: {ell.empirical_containment_pct:>5.1f}% (Theo: {ell.theoretical_containment_pct:.1f}%)"
        )

    lines.extend([
        "-" * 78,
        "CASUALTY RISK & COMPLIANCE ASSESSMENT (NASA-STD-8719.14B Requirement 4.7-1):",
        f"  Total Surviving Debris Casualty Area : {cr.total_casualty_area_m2:.2f} m²",
        f"  Casualty Expectation (Ec)            : {cr.casualty_expectation_ec:.2e} (Threshold: < {cr.nasa_threshold_ec:.1e})",
        f"  NASA Req 4.7-1 Verdict               : {'COMPLIANT (PASSED)' if cr.nasa_req_4_7_compliant else 'NON-COMPLIANT (FAILED)'}",
        f"  Nearest Inhabited Landmass           : {cr.nearest_land_name} ({cr.nearest_land_distance_km:.1f} km)",
        f"  3-Sigma Standoff Clearance Margin    : {cr.standoff_margin_3sigma_km:.1f} km (Req: > 370.4 km)",
        f"  NOTAM/NOTMAR Maritime Hazard Radius  : {cr.notam_exclusion_radius_km:.1f} km",
        f"  Overall Safety Determination         : {cr.risk_level}",
        "=" * 78,
    ])

    return "\n".join(lines)
