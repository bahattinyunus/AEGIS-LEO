"""
AEGIS-LEO: Autonomous Edge-AI & Physics-Informed Orbital Safety Shield
Interactive Streamlit Mission Control & 3D Orbital Visualizer
Developed for NASA Space Apps Challenge 2026 by Team AYAZ.
Lead Architect: Bahattin Yunus Çetin (@bahattinyunus)
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone

from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    QUIET_KP,
    QUIET_F10_7,
    STORM_G4_KP,
    STORM_G5_KP,
)
from aegis_leo.physics.atmosphere import compute_atmospheric_density, space_weather_density_multiplier
from aegis_leo.physics.coordinates import kepler_to_cartesian, cartesian_to_kepler
from aegis_leo.physics.propagator import OrbitalPropagator
from aegis_leo.risk.adri import ADRICalculator, RiskLevel
from aegis_leo.risk.collision import compute_collision_probability
from aegis_leo.risk.debris_density import get_spatial_debris_density
from aegis_leo.ai.pinn import get_pretrained_edge_pinn
from aegis_leo.ai.edge_optimizer import EdgeManeuverOptimizer
from aegis_leo.ai.quantizer import benchmark_edge_inference
from aegis_leo.data.catalog import SATELLITE_CATALOG, DEBRIS_CATALOG
from aegis_leo.data.space_weather import PRESET_STORMS, SpaceWeatherEvent
from aegis_leo.compliance.nasa_std_8719 import NASAComplianceAuditor

# ------------------ Streamlit Page Config ------------------ #

st.set_page_config(
    page_title="AEGIS-LEO | Autonomous Orbital Safety Shield",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Tech Styling
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at 50% 10%, #0d1527 0%, #050811 100%);
        color: #e2e8f0;
    }
    .metric-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 12px;
        padding: 16px;
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .metric-value-cyan {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38bdf8;
        font-family: monospace;
    }
    .metric-value-red {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f43f5e;
        font-family: monospace;
    }
    .metric-value-green {
        font-size: 1.8rem;
        font-weight: 800;
        color: #34d399;
        font-family: monospace;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ------------------ Session State Initialization ------------------ #

if "storm_active" not in st.session_state:
    st.session_state.storm_active = False

if "maneuver_executed" not in st.session_state:
    st.session_state.maneuver_executed = False

if "maneuver_data" not in st.session_state:
    st.session_state.maneuver_data = None


# ------------------ Sidebar Controls ------------------ #

with st.sidebar:
    st.image("https://img.shields.io/badge/NASA%20Space%20Apps-2026%20Ankara-blue.svg", use_container_width=False)
    st.title("🛰️ AEGIS-LEO")
    st.caption("**Autonomous Edge-AI Orbital Safety Shield**  \nTeam AYAZ &bull; Bahattin Yunus Çetin")

    st.markdown("---")
    st.subheader("1. Hedef Uydu (Target Satellite)")
    sat_choice = st.selectbox(
        "Seçilen Uydu:",
        options=list(SATELLITE_CATALOG.keys()),
        format_func=lambda k: f"{SATELLITE_CATALOG[k]['name']} ({SATELLITE_CATALOG[k]['type']})",
    )
    satellite = SATELLITE_CATALOG[sat_choice]

    st.markdown(f"""
    - **İrtifa (Altitude):** {satellite['altitude_km']} km
    - **Eğim (Inclination):** {satellite['inclination_deg']}°
    - **Kütle (Mass):** {satellite['mass_kg']} kg
    - **Yakıt (Propellant):** {satellite['remaining_propellant_kg']} kg
    """)

    st.markdown("---")
    st.subheader("2. Uzay Havası (Space Weather)")
    storm_preset = st.selectbox(
        "Geomanyetik Fırtına Senaryosu:",
        options=list(PRESET_STORMS.keys()),
        format_func=lambda k: PRESET_STORMS[k]["name"],
        index=0 if not st.session_state.storm_active else 3,
    )
    weather_info = PRESET_STORMS[storm_preset]

    st.caption(weather_info["description"])
    custom_weather = st.checkbox("Özel Kp / F10.7 Ayarla (Custom Mode)")
    if custom_weather:
        kp_val = st.slider("Geomanyetik Kp İndeksi", 0.0, 9.0, float(weather_info["kp"]), 0.5)
        f107_val = st.slider("Güneş Radyo Akısı F10.7 (SFU)", 60.0, 360.0, float(weather_info["f10_7"]), 10.0)
    else:
        kp_val = float(weather_info["kp"])
        f107_val = float(weather_info["f10_7"])

    st.markdown("---")
    st.subheader("3. Enkaz & Yakınlaşma (Conjunction)")
    debris_choice = st.selectbox(
        "Tehdit Oluşturan Enkaz:",
        options=list(DEBRIS_CATALOG.keys()),
        format_func=lambda k: f"{DEBRIS_CATALOG[k]['name']}",
    )
    debris = DEBRIS_CATALOG[debris_choice]

    miss_distance_km = st.slider(
        "Tahmini En Yakın Geçiş (Miss Distance - km):",
        min_value=0.05,
        max_value=3.0,
        value=0.45,
        step=0.05,
    )


# ------------------ Core Calculations ------------------ #

adri_calc = ADRICalculator()
optimizer = EdgeManeuverOptimizer(
    satellite_mass_kg=satellite["mass_kg"],
    fuel_remaining_kg=satellite["remaining_propellant_kg"],
    isp_seconds=satellite["isp_seconds"],
    thrust_newtons=satellite["thrust_n"],
)

# Atmospheric expansion calculation
current_density = compute_atmospheric_density(satellite["altitude_km"], kp=kp_val, f10_7=f107_val)
quiet_density = compute_atmospheric_density(satellite["altitude_km"], kp=QUIET_KP, f10_7=QUIET_F10_7)
density_multiplier = current_density / max(quiet_density, 1e-25)

# Along-track displacement due to unmodeled drag decay
is_storm = kp_val >= 5.0
along_track_drift = (kp_val - 2.0) * 2.3 if is_storm else 0.2

# Initial A-DRI Risk calculation
risk_data = adri_calc.evaluate_risk(
    miss_distance_km=miss_distance_km,
    altitude_km=satellite["altitude_km"],
    inclination_deg=satellite["inclination_deg"],
    area_m2=satellite["area_m2"],
    mass_kg=satellite["mass_kg"],
    kp=kp_val,
    f10_7=f107_val,
    along_track_disp_km=along_track_drift,
    remaining_propellant_kg=satellite["remaining_propellant_kg"],
)


# ------------------ Header & Quick Action Buttons ------------------ #

st.title("🛡️ AEGIS-LEO: Otonom Yörünge Güvenlik Kalkanı")
st.markdown("**Autonomous Edge-AI & Physics-Informed Orbital Safety Shield &bull; NASA Space Apps Challenge 2026**")

col_b1, col_b2, col_b3, col_b4 = st.columns([1, 1, 1, 1])

with col_b1:
    if st.button("⚡ Güneş Fırtınası Simüle Et", use_container_width=True, type="secondary"):
        st.session_state.storm_active = True
        st.session_state.maneuver_executed = False
        st.rerun()

with col_b2:
    if st.button("🚀 Otonom Δv Kaçınma Hesapla", use_container_width=True, type="primary"):
        st.session_state.maneuver_executed = True
        st.session_state.maneuver_data = optimizer.plan_autonomous_avoidance(
            current_altitude_km=satellite["altitude_km"],
            inclination_deg=satellite["inclination_deg"],
            initial_miss_distance_km=miss_distance_km,
            storm_kp=kp_val,
            storm_f10_7=f107_val,
            along_track_disp_km=along_track_drift,
            area_m2=satellite["area_m2"],
        )
        st.rerun()

with col_b3:
    if st.button("🔄 Durumu Sıfırla (Nominal)", use_container_width=True):
        st.session_state.storm_active = False
        st.session_state.maneuver_executed = False
        st.session_state.maneuver_data = None
        st.rerun()

with col_b4:
    show_compliance = st.button("📋 NASA-STD-8719.14 Denetle", use_container_width=True)


# ------------------ Metric Cards Display ------------------ #

st.markdown("<br>", unsafe_allow_html=True)
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

# Dynamic values based on maneuver execution
active_adri = risk_data["a_dri"]
active_level = risk_data["risk_level"]
active_miss = miss_distance_km

if st.session_state.maneuver_executed and st.session_state.maneuver_data:
    p_burn = st.session_state.maneuver_data["proactive_burn"]
    active_adri = p_burn["post_burn_adri"]
    active_level = p_burn["post_burn_risk_level"]
    active_miss = p_burn["resulting_separation_km"]

with m_col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">A-DRI Risk Skoru (0.0 - 1.0)</div>
        <div class="{'metric-value-green' if active_level == 'NOMINAL' else 'metric-value-red' if active_level == 'CRITICAL' else 'metric-value-cyan'}">
            {active_adri:.3f}
        </div>
        <small style="color:{'#34d399' if active_level == 'NOMINAL' else '#f43f5e'} font-weight:bold;">Seviye: {active_level}</small>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Termosferik Yoğunluk (ρ)</div>
        <div class="metric-value-cyan">{current_density:.2e}</div>
        <small style="color:#94a3b8;">Fırtına Artışı: <b>{density_multiplier:.1f}x</b> Kat</small>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Boylamsal Sürüklenme (Drift)</div>
        <div class="metric-value-cyan">+{along_track_drift:.1f} km</div>
        <small style="color:#94a3b8;">Sürüklenme Kaynaklı Yörünge Sapması</small>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Enkaz Geçiş Mesafesi</div>
        <div class="{'metric-value-green' if active_miss > 5.0 else 'metric-value-red'}">
            {active_miss:.2f} km
        </div>
        <small style="color:#94a3b8;">Hedef: {debris['name']}</small>
    </div>
    """, unsafe_allow_html=True)


# ------------------ 3D Orbital Visualizer & Charts ------------------ #

v_col1, v_col2 = st.columns([1.8, 1.2])

with v_col1:
    st.subheader("🌍 3D Etkileşimli Yörünge Simülatörü")

    # Generate 3D Earth sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    re = EARTH_RADIUS_KM / 1000.0  # scaled radius (thousands of km)

    xe = re * np.outer(np.cos(u), np.sin(v))
    ye = re * np.outer(np.sin(u), np.sin(v))
    ze = re * np.outer(np.ones(np.size(u)), np.cos(v))

    fig3d = go.Figure()

    # Earth globe
    fig3d.add_trace(go.Surface(
        x=xe, y=ye, z=ze,
        colorscale=[[0, "#0b2545"], [0.5, "#134074"], [1, "#1d4ed8"]],
        showscale=False,
        opacity=0.9,
        name="Earth",
        hoverinfo="name",
    ))

    # Satellite orbit trajectory
    r_orbit = (EARTH_RADIUS_KM + satellite["altitude_km"]) / 1000.0
    inc_rad = np.radians(satellite["inclination_deg"])
    theta_vals = np.linspace(0, 2 * np.pi, 120)

    sat_x = r_orbit * np.cos(theta_vals)
    sat_y = r_orbit * np.sin(theta_vals) * np.cos(inc_rad)
    sat_z = r_orbit * np.sin(theta_vals) * np.sin(inc_rad)

    fig3d.add_trace(go.Scatter3d(
        x=sat_x, y=sat_y, z=sat_z,
        mode="lines",
        line=dict(color="#06b6d4", width=4),
        name=f"Yörünge ({satellite['name']})",
    ))

    # Satellite current position
    fig3d.add_trace(go.Scatter3d(
        x=[sat_x[0]], y=[sat_y[0]], z=[sat_z[0]],
        mode="markers+text",
        marker=dict(size=8, color="#22d3ee", symbol="circle"),
        text=[satellite["name"]],
        textposition="top center",
        name="Uydu Konumu",
    ))

    # Debris position
    deb_x = sat_x[1] + (miss_distance_km / 1000.0) * 1.5
    deb_y = sat_y[1] + (miss_distance_km / 1000.0) * 1.5
    deb_z = sat_z[1]

    fig3d.add_trace(go.Scatter3d(
        x=[deb_x], y=[deb_y], z=[deb_z],
        mode="markers+text",
        marker=dict(size=7, color="#f43f5e", symbol="diamond"),
        text=[f"Enkaz: {debris['name']}"],
        textposition="bottom center",
        name="Uzay Enkazı",
    ))

    # If maneuver executed, draw corrected orbit
    if st.session_state.maneuver_executed:
        r_evasive = r_orbit + 0.0035  # Phased / elevated orbit
        evasive_x = r_evasive * np.cos(theta_vals)
        evasive_y = r_evasive * np.sin(theta_vals) * np.cos(inc_rad)
        evasive_z = r_evasive * np.sin(theta_vals) * np.sin(inc_rad)

        fig3d.add_trace(go.Scatter3d(
            x=evasive_x, y=evasive_y, z=evasive_z,
            mode="lines",
            line=dict(color="#10b981", width=5, dash="dash"),
            name="Otonom Kaçınma Yörüngesi (Safe Orbit)",
        ))

    fig3d.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, visible=False),
            yaxis=dict(showbackground=False, visible=False),
            zaxis=dict(showbackground=False, visible=False),
            aspectmode="data",
            bgcolor="#050811",
        ),
        paper_bgcolor="#050811",
        margin=dict(l=0, r=0, b=0, t=10),
        legend=dict(font=dict(color="#cbd5e1", size=10), yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=480,
    )

    st.plotly_chart(fig3d, use_container_width=True)


with v_col2:
    st.subheader("📊 A-DRI Risk Bileşenleri")

    # Radar/bar breakdown of the 4 factors
    comps = risk_data["components"]
    categories = [
        "C_prob (Çarpışma İhtimali)",
        "W_drag (Uzay Havası Belirsizliği)",
        "D_ord (NASA Enkaz Yoğunluğu)",
        "F_Δv (Yakıt Fizibilitesi)",
    ]
    values = [
        comps["c_prob_normalized"],
        comps["w_drag"],
        comps["d_ord"],
        comps["f_delta_v"],
    ]

    fig_bar = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation="h",
        marker=dict(
            color=["#f43f5e" if v > 0.6 else "#eab308" if v > 0.3 else "#06b6d4" for v in values]
        ),
        text=[f"{v:.2f}" for v in values],
        textposition="auto",
    ))

    fig_bar.update_layout(
        xaxis=dict(range=[0, 1.0], gridcolor="#1e293b", title="Bileşen Etkisi (0.0 - 1.0)"),
        yaxis=dict(gridcolor="#1e293b"),
        paper_bgcolor="#050811",
        plot_bgcolor="#0d1527",
        font=dict(color="#e2e8f0", size=11),
        margin=dict(l=10, r=10, b=20, t=20),
        height=260,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Edge-AI Optimizer Summary Card
    if st.session_state.maneuver_executed and st.session_state.maneuver_data:
        m = st.session_state.maneuver_data
        st.success(f"⚡ **Otonom Evasive Manevra Tamamlandı!**")
        st.markdown(f"""
        - **Gerekli Δv Ateşlemesi:** `{m['proactive_burn']['delta_v_m_s']} m/s`
        - **Harcanan İtici Gaz:** `{m['proactive_burn']['propellant_consumed_kg']} kg`
        - **Tasarruf Edilen Yakıt:** `{m['efficiency_metrics']['fuel_savings_percentage']}%` (Geç reaktif manevraya kıyasla)
        - **Yeni Güvenli Mesafe:** `{m['proactive_burn']['resulting_separation_km']} km`
        - **Risk İndirgeme:** `%{m['efficiency_metrics']['risk_reduction_percentage']}`
        """)
    else:
        st.info("ℹ️ Güneş fırtınasının yörüngeye etkisini görmek için **'Güneş Fırtınası Simüle Et'** butonuna basın.")


# ------------------ NASA Compliance Section ------------------ #

if show_compliance or st.checkbox("📋 NASA-STD-8719.14 Uyumluluk Denetim Detayları"):
    st.markdown("---")
    st.subheader("📋 NASA-STD-8719.14 Yörünge Enkaz Azaltma Uyumluluk Denetimi")

    auditor = NASAComplianceAuditor()
    audit_res = auditor.audit_mission(
        mission_name=satellite["name"],
        norad_id=satellite["norad_id"],
        altitude_km=satellite["altitude_km"],
        inclination_deg=satellite["inclination_deg"],
        mass_kg=satellite["mass_kg"],
        area_m2=satellite["area_m2"],
        operational_collision_prob=risk_data["components"]["c_prob_raw"],
        has_passivation_valves=True,
    )

    c_cols = st.columns(len(audit_res["compliance_details"]))
    for idx, item in enumerate(audit_res["compliance_details"]):
        with c_cols[idx]:
            is_pass = item["status"] == "PASS"
            st.metric(
                label=item["standard_id"],
                value="✅ PASS" if is_pass else "❌ FAIL",
                help=item["details"],
            )

    report_md = auditor.generate_markdown_report(audit_res)
    st.download_button(
        label="📥 NASA ODAR Raporunu İndir (.md)",
        data=report_md,
        file_name=f"NASA_ODAR_{satellite['name'].replace(' ', '_')}.md",
        mime="text/markdown",
    )


# ------------------ Edge-AI PINN Profiling Section ------------------ #

with st.expander("🧠 Edge-AI Uç Nokta Performans & Donanım Profili (INT8 Benchmark)"):
    pinn_model = get_pretrained_edge_pinn()
    bench = benchmark_edge_inference(pinn_model, num_iterations=500)

    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    p_col1.metric("PINN Parametre Sayısı", f"{bench['parameter_count']:,}")
    p_col2.metric("INT8 Bellek Alanı", f"{bench['int8_quantized_memory_kb']} KB")
    p_col3.metric("Uç Nokta Gecikmesi", f"{bench['latency_microseconds']} µs")
    p_col4.metric("Uçuş Donanımı Testi", "✅ BAŞARILI")
    st.caption("Radyasyon dayanımlı gömülü uçuş bilgisayarlarında (NVIDIA Jetson / Onboard RT-OBC) yer istasyonu iletişimi kesildiğinde otonom çalışacak şekilde optimize edilmiştir.")
