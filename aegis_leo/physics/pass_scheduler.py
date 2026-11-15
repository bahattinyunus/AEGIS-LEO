"""
Ground Station Pass Scheduler & RF/Optical Link Budget Engine
==============================================================

Computes:
1. Topocentric South-East-Zenith (SEZ) look angles (Azimuth, Elevation, Slant Range,
   Range Rate, and Doppler shift) from ECI/Keplerian satellite trajectories.
2. Acquisition of Signal (AOS) / Loss of Signal (LOS) visibility pass windows
   across global ground stations.
3. Multi-band RF & Optical Downlink Link Budget (S-Band, X-Band, Ka-Band, Optical OISL):
   - Free-Space Path Loss (FSPL)
   - Elevation-dependent atmospheric & rain attenuation
   - Received power, Noise spectral density, C/N0, Eb/N0, Link Margin (dB)
   - Shannon-Hartley & adaptive modulation data throughput (Mbps) and pass data volume (MB).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aegis_leo.constants import (
    EARTH_FLATTENING,
    MU_EARTH_KM3_S2,
    EARTH_RADIUS_KM,
    EARTH_ROTATION_RATE_RAD_S,
)
from aegis_leo.physics.ground_track import GROUND_STATIONS

SPEED_OF_LIGHT_M_S = 299_792_458.0
BOLTZMANN_K_DBW_K_HZ = -228.6  # 10*log10(1.380649e-23)


class LinkBand(str, Enum):
    S_BAND = "S_BAND"      # 2.25 GHz TT&C
    X_BAND = "X_BAND"      # 8.20 GHz Payload Downlink
    KA_BAND = "KA_BAND"    # 26.5 GHz High-Rate Downlink
    OPTICAL = "OPTICAL"    # 1550 nm Laser Downlink


@dataclass(frozen=True)
class LinkBandSpec:
    band: LinkBand
    frequency_hz: float
    tx_power_dbw: float
    tx_antenna_gain_dbi: float
    rx_g_over_t_db_k: float
    bandwidth_hz: float
    required_eb_n0_db: float
    zenith_atm_loss_db: float
    rain_fade_coeff_db: float
    implementation_loss_db: float = 2.0


DEFAULT_BAND_SPECS: Dict[LinkBand, LinkBandSpec] = {
    LinkBand.S_BAND: LinkBandSpec(
        band=LinkBand.S_BAND,
        frequency_hz=2.25e9,
        tx_power_dbw=7.0,           # ~5 W
        tx_antenna_gain_dbi=6.0,
        rx_g_over_t_db_k=22.0,
        bandwidth_hz=2.0e6,
        required_eb_n0_db=9.6,
        zenith_atm_loss_db=0.15,
        rain_fade_coeff_db=0.05,
    ),
    LinkBand.X_BAND: LinkBandSpec(
        band=LinkBand.X_BAND,
        frequency_hz=8.20e9,
        tx_power_dbw=10.0,          # 10 W
        tx_antenna_gain_dbi=16.0,
        rx_g_over_t_db_k=31.0,
        bandwidth_hz=150.0e6,
        required_eb_n0_db=10.5,
        zenith_atm_loss_db=0.35,
        rain_fade_coeff_db=0.45,
    ),
    LinkBand.KA_BAND: LinkBandSpec(
        band=LinkBand.KA_BAND,
        frequency_hz=26.5e9,
        tx_power_dbw=12.0,          # ~16 W
        tx_antenna_gain_dbi=28.0,
        rx_g_over_t_db_k=36.0,
        bandwidth_hz=500.0e6,
        required_eb_n0_db=11.0,
        zenith_atm_loss_db=0.80,
        rain_fade_coeff_db=2.80,
    ),
    LinkBand.OPTICAL: LinkBandSpec(
        band=LinkBand.OPTICAL,
        frequency_hz=1.934e14,      # 1550 nm
        tx_power_dbw=3.0,           # 2 W laser
        tx_antenna_gain_dbi=95.0,   # diffraction-limited 10 cm aperture
        rx_g_over_t_db_k=48.0,
        bandwidth_hz=1.5e9,
        required_eb_n0_db=12.0,
        zenith_atm_loss_db=1.20,
        rain_fade_coeff_db=0.50,
    ),
}


@dataclass
class PassWindow:
    station_name: str
    country: str
    aos_time_s: float
    los_time_s: float
    duration_s: float
    max_elevation_deg: float
    aos_azimuth_deg: float
    los_azimuth_deg: float
    min_slant_range_km: float
    max_doppler_khz: float
    mean_link_margin_db: float
    peak_throughput_mbps: float
    total_data_volume_mb: float
    band: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PassScheduleResult:
    simulation_duration_s: float
    altitude_km: float
    inclination_deg: float
    band: str
    total_passes: int
    total_contact_time_s: float
    total_data_downlinked_mb: float
    passes: List[PassWindow] = field(default_factory=list)
    best_station: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_duration_s": round(self.simulation_duration_s, 1),
            "altitude_km": round(self.altitude_km, 2),
            "inclination_deg": round(self.inclination_deg, 2),
            "band": self.band,
            "total_passes": self.total_passes,
            "total_contact_time_s": round(self.total_contact_time_s, 1),
            "total_data_downlinked_mb": round(self.total_data_downlinked_mb, 2),
            "best_station": self.best_station,
            "passes": [p.to_dict() for p in self.passes],
        }


def geodetic_to_ecef_km(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
    """Convert geodetic latitude, longitude, and altitude to ECEF position (km)."""
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    e2 = EARTH_FLATTENING * (2.0 - EARTH_FLATTENING)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n_val = EARTH_RADIUS_KM / math.sqrt(1.0 - e2 * sin_lat * sin_lat)

    x = (n_val + alt_km) * cos_lat * math.cos(lon)
    y = (n_val + alt_km) * cos_lat * math.sin(lon)
    z = (n_val * (1.0 - e2) + alt_km) * sin_lat
    return np.array([x, y, z], dtype=np.float64)


def eci_to_ecef(vec_eci: np.ndarray, gmst_rad: float) -> np.ndarray:
    """Rotate ECI vector into ECEF frame by Greenwich Mean Sidereal Time."""
    c = math.cos(gmst_rad)
    s = math.sin(gmst_rad)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot @ vec_eci


def compute_topocentric_look_angles(
    sat_pos_eci_km: np.ndarray,
    sat_vel_eci_km_s: np.ndarray,
    station_lat_deg: float,
    station_lon_deg: float,
    gmst_rad: float,
    carrier_freq_hz: float = 8.2e9,
) -> Dict[str, float]:
    """
    Compute topocentric Azimuth, Elevation, Slant Range, Range Rate, and Doppler Shift
    in the station's South-East-Zenith (SEZ) frame.
    """
    sat_ecef = eci_to_ecef(sat_pos_eci_km, gmst_rad)
    omega_earth = np.array([0.0, 0.0, EARTH_ROTATION_RATE_RAD_S], dtype=np.float64)
    sat_vel_ecef = eci_to_ecef(sat_vel_eci_km_s, gmst_rad) - np.cross(omega_earth, sat_ecef)

    gs_ecef = geodetic_to_ecef_km(station_lat_deg, station_lon_deg, 0.0)
    rho_ecef = sat_ecef - gs_ecef
    slant_range_km = float(np.linalg.norm(rho_ecef))

    lat = math.radians(station_lat_deg)
    lon = math.radians(station_lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)

    # ECEF -> SEZ rotation matrix
    rot_sez = np.array([
        [sin_lat * cos_lon, sin_lat * sin_lon, -cos_lat],
        [-sin_lon, cos_lon, 0.0],
        [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
    ], dtype=np.float64)

    rho_sez = rot_sez @ rho_ecef
    s_comp, e_comp, z_comp = float(rho_sez[0]), float(rho_sez[1]), float(rho_sez[2])

    horiz = math.hypot(s_comp, e_comp)
    elevation_deg = math.degrees(math.atan2(z_comp, horiz))
    azimuth_deg = math.degrees(math.atan2(e_comp, -s_comp)) % 360.0

    # Range rate (km/s): positive = moving away -> negative Doppler
    range_rate_km_s = float(np.dot(rho_ecef, sat_vel_ecef) / max(slant_range_km, 1e-6))
    doppler_hz = -(range_rate_km_s * 1000.0 / SPEED_OF_LIGHT_M_S) * carrier_freq_hz

    return {
        "azimuth_deg": azimuth_deg,
        "elevation_deg": elevation_deg,
        "slant_range_km": slant_range_km,
        "range_rate_km_s": range_rate_km_s,
        "doppler_khz": doppler_hz / 1000.0,
    }


def compute_link_budget(
    slant_range_km: float,
    elevation_deg: float,
    band: LinkBand = LinkBand.X_BAND,
    rain_rate_mm_hr: float = 0.0,
) -> Dict[str, float]:
    """
    Compute RF/Optical downlink budget at a given slant range and elevation angle.
    """
    spec = DEFAULT_BAND_SPECS[band]
    dist_m = max(slant_range_km * 1000.0, 1.0)
    wavelength_m = SPEED_OF_LIGHT_M_S / spec.frequency_hz

    # Free-Space Path Loss (dB)
    fspl_db = 20.0 * math.log10(4.0 * math.pi * dist_m / wavelength_m)

    # Atmospheric & rain attenuation scaled by cosecant of elevation (capped at 5 deg)
    elev_eff_rad = math.radians(max(elevation_deg, 5.0))
    airmass = 1.0 / math.sin(elev_eff_rad)
    atm_loss_db = spec.zenith_atm_loss_db * airmass
    rain_loss_db = spec.rain_fade_coeff_db * (rain_rate_mm_hr / 10.0) * airmass

    eirp_dbw = spec.tx_power_dbw + spec.tx_antenna_gain_dbi
    c_over_n0_db_hz = (
        eirp_dbw
        - fspl_db
        - atm_loss_db
        - rain_loss_db
        - spec.implementation_loss_db
        + spec.rx_g_over_t_db_k
        - BOLTZMANN_K_DBW_K_HZ
    )

    # SNR in channel bandwidth
    snr_db = c_over_n0_db_hz - 10.0 * math.log10(spec.bandwidth_hz)
    snr_linear = max(10.0 ** (snr_db / 10.0), 1e-6)

    # Shannon capacity with 0.70 coding/framing efficiency factor
    shannon_mbps = (spec.bandwidth_hz * math.log2(1.0 + snr_linear) * 0.70) / 1.0e6
    # Eb/N0 at achievable data rate
    data_rate_hz = max(shannon_mbps * 1.0e6, 1.0)
    eb_n0_db = c_over_n0_db_hz - 10.0 * math.log10(data_rate_hz)
    link_margin_db = snr_db - spec.required_eb_n0_db

    # If link margin is negative, practical throughput drops
    achievable_mbps = shannon_mbps if link_margin_db >= 0.0 else max(0.0, shannon_mbps * 0.1)

    return {
        "band": band.value,
        "fspl_db": round(fspl_db, 2),
        "atmospheric_loss_db": round(atm_loss_db + rain_loss_db, 2),
        "eirp_dbw": round(eirp_dbw, 2),
        "c_over_n0_db_hz": round(c_over_n0_db_hz, 2),
        "snr_db": round(snr_db, 2),
        "eb_n0_db": round(eb_n0_db, 2),
        "link_margin_db": round(link_margin_db, 2),
        "throughput_mbps": round(achievable_mbps, 2),
    }


def _generate_circular_eci_orbit(
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    duration_s: float,
    dt_s: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ECI positions and velocities for a circular LEO orbit."""
    r_km = EARTH_RADIUS_KM + altitude_km
    v_km_s = math.sqrt(MU_EARTH_KM3_S2 / r_km)
    n_rad_s = v_km_s / r_km

    inc = math.radians(inclination_deg)
    raan = math.radians(raan_deg)

    times = np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)
    positions = np.zeros((len(times), 3), dtype=np.float64)
    velocities = np.zeros((len(times), 3), dtype=np.float64)

    c_o, s_o = math.cos(raan), math.sin(raan)
    c_i, s_i = math.cos(inc), math.sin(inc)

    for idx, t in enumerate(times):
        u = n_rad_s * t
        c_u, s_u = math.cos(u), math.sin(u)

        # PQW -> ECI (omega=0, nu=u)
        x = r_km * (c_o * c_u - s_o * s_u * c_i)
        y = r_km * (s_o * c_u + c_o * s_u * c_i)
        z = r_km * (s_u * s_i)

        vx = v_km_s * (-c_o * s_u - s_o * c_u * c_i)
        vy = v_km_s * (-s_o * s_u + c_o * c_u * c_i)
        vz = v_km_s * (c_u * s_i)

        positions[idx] = [x, y, z]
        velocities[idx] = [vx, vy, vz]

    return times, positions, velocities


def schedule_ground_passes(
    altitude_km: float = 550.0,
    inclination_deg: float = 53.0,
    raan_deg: float = 15.0,
    duration_hours: float = 12.0,
    dt_s: float = 20.0,
    min_elevation_deg: float = 5.0,
    band: LinkBand = LinkBand.X_BAND,
    rain_rate_mm_hr: float = 0.0,
    stations: Optional[List[Dict[str, Any]]] = None,
) -> PassScheduleResult:
    """
    Simulate LEO satellite passes over global ground stations and compute RF/Optical
    link budgets and total downlinked data volume.
    """
    duration_s = duration_hours * 3600.0
    times, positions, velocities = _generate_circular_eci_orbit(
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        raan_deg=raan_deg,
        duration_s=duration_s,
        dt_s=dt_s,
    )

    station_list = stations if stations is not None else GROUND_STATIONS
    carrier_hz = DEFAULT_BAND_SPECS[band].frequency_hz
    passes: List[PassWindow] = []

    for st in station_list:
        st_min_el = max(min_elevation_deg, float(st.get("min_elevation_deg", 5.0)))
        in_pass = False
        aos_t = 0.0
        aos_az = 0.0
        max_el = -90.0
        min_range = 1e9
        max_doppler = 0.0
        last_az = 0.0
        margins: List[float] = []
        throughputs: List[float] = []
        data_mb = 0.0

        for idx, t in enumerate(times):
            gmst = EARTH_ROTATION_RATE_RAD_S * float(t)
            look = compute_topocentric_look_angles(
                positions[idx],
                velocities[idx],
                st["lat"],
                st["lon"],
                gmst_rad=gmst,
                carrier_freq_hz=carrier_hz,
            )
            el = look["elevation_deg"]
            if el >= st_min_el:
                lb = compute_link_budget(
                    slant_range_km=look["slant_range_km"],
                    elevation_deg=el,
                    band=band,
                    rain_rate_mm_hr=rain_rate_mm_hr,
                )
                if not in_pass:
                    in_pass = True
                    aos_t = float(t)
                    aos_az = look["azimuth_deg"]
                    max_el = el
                    min_range = look["slant_range_km"]
                    max_doppler = abs(look["doppler_khz"])
                    margins = [lb["link_margin_db"]]
                    throughputs = [lb["throughput_mbps"]]
                    data_mb = (lb["throughput_mbps"] * dt_s) / 8.0
                else:
                    max_el = max(max_el, el)
                    min_range = min(min_range, look["slant_range_km"])
                    max_doppler = max(max_doppler, abs(look["doppler_khz"]))
                    margins.append(lb["link_margin_db"])
                    throughputs.append(lb["throughput_mbps"])
                    data_mb += (lb["throughput_mbps"] * dt_s) / 8.0
                last_az = look["azimuth_deg"]
            elif in_pass:
                in_pass = False
                los_t = float(t)
                passes.append(
                    PassWindow(
                        station_name=st["name"],
                        country=st["country"],
                        aos_time_s=round(aos_t, 1),
                        los_time_s=round(los_t, 1),
                        duration_s=round(los_t - aos_t, 1),
                        max_elevation_deg=round(max_el, 2),
                        aos_azimuth_deg=round(aos_az, 1),
                        los_azimuth_deg=round(last_az, 1),
                        min_slant_range_km=round(min_range, 2),
                        max_doppler_khz=round(max_doppler, 2),
                        mean_link_margin_db=round(float(np.mean(margins)), 2),
                        peak_throughput_mbps=round(max(throughputs), 2),
                        total_data_volume_mb=round(data_mb, 2),
                        band=band.value,
                    )
                )

        if in_pass:
            los_t = float(times[-1])
            passes.append(
                PassWindow(
                    station_name=st["name"],
                    country=st["country"],
                    aos_time_s=round(aos_t, 1),
                    los_time_s=round(los_t, 1),
                    duration_s=round(max(dt_s, los_t - aos_t), 1),
                    max_elevation_deg=round(max_el, 2),
                    aos_azimuth_deg=round(aos_az, 1),
                    los_azimuth_deg=round(last_az, 1),
                    min_slant_range_km=round(min_range, 2),
                    max_doppler_khz=round(max_doppler, 2),
                    mean_link_margin_db=round(float(np.mean(margins)), 2),
                    peak_throughput_mbps=round(max(throughputs), 2),
                    total_data_volume_mb=round(data_mb, 2),
                    band=band.value,
                )
            )

    passes.sort(key=lambda p: p.aos_time_s)
    total_contact = sum(p.duration_s for p in passes)
    total_data = sum(p.total_data_volume_mb for p in passes)

    best_station = None
    if passes:
        by_station: Dict[str, float] = {}
        for p in passes:
            by_station[p.station_name] = by_station.get(p.station_name, 0.0) + p.total_data_volume_mb
        best_station = max(by_station, key=by_station.get)  # type: ignore[arg-type]

    return PassScheduleResult(
        simulation_duration_s=duration_s,
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        band=band.value,
        total_passes=len(passes),
        total_contact_time_s=total_contact,
        total_data_downlinked_mb=total_data,
        passes=passes,
        best_station=best_station,
    )
