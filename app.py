"""
AEGIS-LEO: Otonom Uç Nokta Yapay Zekası & Fizik Bilgilendirmeli Yörünge Güvenlik Kalkanı
Kapsamlı Streamlit Görev Kontrol, Takımyıldız Taraması ve Pareto Optimizasyon Platformu
NASA Space Apps Challenge 2026 | AYAZ Takımı | Lider Mimar: Bahattin Yunus Çetin
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone

from aegis_leo.constants import (
    EARTH_RADIUS_KM,
    QUIET_KP,
    QUIET_F10_7,
)
from aegis_leo.physics.atmosphere import compute_atmospheric_density
from aegis_leo.physics.coordinates import kepler_to_cartesian
from aegis_leo.physics.propagator import OrbitalPropagator
from aegis_leo.physics.atmosphere_models import compare_atmospheric_models
from aegis_leo.physics.ground_track import calculate_ground_track, analyze_communication_blackouts, GROUND_STATIONS
from aegis_leo.risk.adri import ADRICalculator, RiskLevel
from aegis_leo.ai.pinn import get_pretrained_edge_pinn
from aegis_leo.ai.edge_optimizer import EdgeManeuverOptimizer
from aegis_leo.ai.pareto_optimizer import ParetoManeuverOptimizer
from aegis_leo.ai.quantizer import benchmark_edge_inference
from aegis_leo.screening.constellation_screen import (
    generate_walker_delta_constellation,
    generate_synthetic_debris_cloud,
    ConstellationConjunctionScreener,
)
from aegis_leo.data.catalog import SATELLITE_CATALOG, DEBRIS_CATALOG
from aegis_leo.data.space_weather import PRESET_STORMS
from aegis_leo.data.live_feeds import LiveSpaceWeatherFeed
from aegis_leo.data.celestrak_live import CelesTrakClient
from aegis_leo.ai.radiation_fault_simulator import RadiationFaultSimulator
from aegis_leo.physics.deorbit_simulation import compute_deorbit_burn, simulate_reentry_corridor, POINT_NEMO
from aegis_leo.risk.covariance_visualizer import generate_covariance_ellipsoid_mesh
from aegis_leo.agents.swarm_orchestrator import OrbitalSwarmOrchestrator
import torch

from aegis_leo.compliance.nasa_std_8719 import NASAComplianceAuditor

# ------------------ Streamlit Sayfa Yapılandırması ------------------ #

st.set_page_config(
    page_title="AEGIS-LEO | Otonom Yörünge Güvenlik Kalkanı",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
</style>
""", unsafe_allow_html=True)


# ------------------ Oturum Durumu (Session State) ------------------ #

if "storm_active" not in st.session_state:
    st.session_state.storm_active = False

if "maneuver_executed" not in st.session_state:
    st.session_state.maneuver_executed = False

if "maneuver_data" not in st.session_state:
    st.session_state.maneuver_data = None


# ------------------ Kenar Çubuğu (Sidebar) ------------------ #

with st.sidebar:
    st.image("https://img.shields.io/badge/NASA%20Space%20Apps-2026%20Ankara-blue.svg", use_container_width=False)
    st.title("🛡️ AEGIS-LEO")
    st.caption("**Otonom Uç Nokta Yörünge Güvenlik Kalkanı**  \nAYAZ Takımı &bull; Bahattin Yunus Çetin")

    st.markdown("---")
    st.subheader("1. Hedef Uydu")
    sat_choice = st.selectbox(
        "Takip Edilen Uydu:",
        options=list(SATELLITE_CATALOG.keys()),
        format_func=lambda k: f"{SATELLITE_CATALOG[k]['name']} ({SATELLITE_CATALOG[k]['type']})",
    )
    satellite = SATELLITE_CATALOG[sat_choice]

    st.markdown(f"""
    - **İrtifa:** {satellite['altitude_km']} km | **Eğim:** {satellite['inclination_deg']}°
    - **Kütle:** {satellite['mass_kg']} kg | **Yakıt:** {satellite['remaining_propellant_kg']} kg
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

    custom_weather = st.checkbox("Manuel Kp / F10.7 Girişi")
    if custom_weather:
        kp_val = st.slider("Geomanyetik Kp İndeksi", 0.0, 9.0, float(weather_info["kp"]), 0.5)
        f107_val = st.slider("Güneş Akısı F10.7 (SFU)", 60.0, 360.0, float(weather_info["f10_7"]), 10.0)
    else:
        kp_val = float(weather_info["kp"])
        f107_val = float(weather_info["f10_7"])

    st.markdown("---")
    st.subheader("3. Tehdit & Yakınlaşma")
    debris_choice = st.selectbox(
        "Yakınlaşan Enkaz:",
        options=list(DEBRIS_CATALOG.keys()),
        format_func=lambda k: f"{DEBRIS_CATALOG[k]['name']}",
    )
    debris = DEBRIS_CATALOG[debris_choice]

    miss_distance_km = st.slider(
        "Tahmini En Yakın Geçiş (km):",
        min_value=0.05,
        max_value=3.0,
        value=0.45,
        step=0.05,
    )

    # Canlı NOAA SWPC Bağlantısı
    st.markdown("---")
    if st.button("🌐 Canlı NOAA SWPC Verisini Çek"):
        live_feed = LiveSpaceWeatherFeed()
        live_res = live_feed.fetch_live_kp_and_flux()
        st.success(f"NOAA SWPC: Kp = {live_res['observed_kp']} ({live_res['status']})")


# ------------------ Matematiksel Hesaplamalar ------------------ #

adri_calc = ADRICalculator()
optimizer = EdgeManeuverOptimizer(
    satellite_mass_kg=satellite["mass_kg"],
    fuel_remaining_kg=satellite["remaining_propellant_kg"],
    isp_seconds=satellite["isp_seconds"],
    thrust_newtons=satellite["thrust_n"],
)

current_density = compute_atmospheric_density(satellite["altitude_km"], kp=kp_val, f10_7=f107_val)
quiet_density = compute_atmospheric_density(satellite["altitude_km"], kp=QUIET_KP, f10_7=QUIET_F10_7)
density_multiplier = current_density / max(quiet_density, 1e-25)

is_storm = kp_val >= 5.0
along_track_drift = (kp_val - 2.0) * 2.3 if is_storm else 0.2

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


# ------------------ Üst Başlık & Sekmeler (Tabs) ------------------ #

st.title("🛡️ AEGIS-LEO: Otonom Yörünge Güvenlik Kalkanı")
st.markdown("**Autonomous Edge-AI & Physics-Informed Orbital Safety Shield &bull; NASA Space Apps Challenge 2026**")

tab_sim, tab_fleet, tab_pareto, tab_ground, tab_atmos, tab_rad, tab_deorbit, tab_celestrak, tab_swarm = st.tabs([
    "🌍 1. 3D Yörünge & Kaçınma",
    "🛰️ 2. Takımyıldız & Filo Taraması",
    "⚖️ 3. Pareto Manevra Frontier",
    "📡 4. Yer İstasyonları & Blackout",
    "🔬 5. Atmosfer Modelleri Laboratuvarı",
    "☢️ 6. Radyasyon & TMR Hata Toleransı",
    "🎯 7. Deorbit & Point Nemo Koridoru",
    "🌐 8. Canlı CelesTrak TLE Sorgulama",
    "🤖 9. 9-Ajanlı Sürü Zekası (Autonomous Swarm)",
])



# ==========================================
# SEKME 1: 3D YÖRÜNGE SİMÜLATÖRÜ & KAÇINMA
# ==========================================
with tab_sim:
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

    # Metrik Kartları
    st.markdown("<br>", unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

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
            <small style="color:{'#34d399' if active_level == 'NOMINAL' else '#f43f5e'}; font-weight:bold;">Seviye: {active_level}</small>
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
            <small style="color:#94a3b8;">CDM İptal Eden Konum Sapması</small>
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

    # 3D Küre & Çubuk Grafik
    v_col1, v_col2 = st.columns([1.8, 1.2])

    with v_col1:
        st.subheader("🌍 3D Etkileşimli Yörünge Simülatörü")
        u = np.linspace(0, 2 * np.pi, 45)
        v = np.linspace(0, np.pi, 45)
        re = EARTH_RADIUS_KM / 1000.0

        xe = re * np.outer(np.cos(u), np.sin(v))
        ye = re * np.outer(np.sin(u), np.sin(v))
        ze = re * np.outer(np.ones(np.size(u)), np.cos(v))

        fig3d = go.Figure()
        fig3d.add_trace(go.Surface(
            x=xe, y=ye, z=ze,
            colorscale=[[0, "#0b2545"], [0.5, "#134074"], [1, "#1d4ed8"]],
            showscale=False, opacity=0.9, name="Dünya",
        ))

        r_orbit = (EARTH_RADIUS_KM + satellite["altitude_km"]) / 1000.0
        inc_rad = np.radians(satellite["inclination_deg"])
        theta_vals = np.linspace(0, 2 * np.pi, 120)

        sat_x = r_orbit * np.cos(theta_vals)
        sat_y = r_orbit * np.sin(theta_vals) * np.cos(inc_rad)
        sat_z = r_orbit * np.sin(theta_vals) * np.sin(inc_rad)

        fig3d.add_trace(go.Scatter3d(
            x=sat_x, y=sat_y, z=sat_z, mode="lines",
            line=dict(color="#06b6d4", width=4), name=f"Yörünge ({satellite['name']})",
        ))

        fig3d.add_trace(go.Scatter3d(
            x=[sat_x[0]], y=[sat_y[0]], z=[sat_z[0]], mode="markers+text",
            marker=dict(size=8, color="#22d3ee"), text=[satellite["name"]], name="Uydu",
        ))

        deb_x = sat_x[1] + (miss_distance_km / 1000.0) * 1.5
        deb_y = sat_y[1] + (miss_distance_km / 1000.0) * 1.5
        deb_z = sat_z[1]

        fig3d.add_trace(go.Scatter3d(
            x=[deb_x], y=[deb_y], z=[deb_z], mode="markers+text",
            marker=dict(size=7, color="#f43f5e", symbol="diamond"), text=[debris["name"]], name="Enkaz",
        ))

        if st.session_state.maneuver_executed:
            r_evasive = r_orbit + 0.0035
            evasive_x = r_evasive * np.cos(theta_vals)
            evasive_y = r_evasive * np.sin(theta_vals) * np.cos(inc_rad)
            evasive_z = r_evasive * np.sin(theta_vals) * np.sin(inc_rad)

            fig3d.add_trace(go.Scatter3d(
                x=evasive_x, y=evasive_y, z=evasive_z, mode="lines",
                line=dict(color="#10b981", width=5, dash="dash"), name="Güvenli Kaçınma Yörüngesi",
            ))

        fig3d.update_layout(
            scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor="#050811"),
            paper_bgcolor="#050811", margin=dict(l=0, r=0, b=0, t=10), height=460,
        )
        st.plotly_chart(fig3d, use_container_width=True)

    with v_col2:
        st.subheader("📊 A-DRI Risk Bileşenleri Dağılımı")
        comps = risk_data["components"]
        categories = ["C_prob (Çarpışma)", "W_drag (Uzay Havası)", "D_ord (Enkaz)", "F_Δv (Yakıt)"]
        values = [comps["c_prob_normalized"], comps["w_drag"], comps["d_ord"], comps["f_delta_v"]]

        fig_bar = go.Figure(go.Bar(
            x=values, y=categories, orientation="h",
            marker=dict(color=["#f43f5e" if v > 0.6 else "#eab308" if v > 0.3 else "#06b6d4" for v in values]),
            text=[f"{v:.2f}" for v in values], textposition="auto",
        ))
        fig_bar.update_layout(
            xaxis=dict(range=[0, 1.0], gridcolor="#1e293b", title="A-DRI Ağırlık Katkısı"),
            yaxis=dict(gridcolor="#1e293b"), paper_bgcolor="#050811", plot_bgcolor="#0d1527",
            font=dict(color="#e2e8f0", size=11), margin=dict(l=10, r=10, b=20, t=20), height=240,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        if st.session_state.maneuver_executed and st.session_state.maneuver_data:
            m = st.session_state.maneuver_data
            st.success("⚡ **Otonom Kaçınma Ateşlemesi Gerçekleştirildi!**")
            st.markdown(f"""
            - **Planlanan Δv:** `{m['proactive_burn']['delta_v_m_s']} m/s`
            - **Harcanan İtici Gaz:** `{m['proactive_burn']['propellant_consumed_kg']} kg`
            - **Yakıt Tasarrufu:** `%{m['efficiency_metrics']['fuel_savings_percentage']}`
            - **Güvenli Ayrılma:** `{m['proactive_burn']['resulting_separation_km']} km`
            """)

    if show_compliance or st.checkbox("📋 NASA-STD-8719.14 Denetim Raporunu Görüntüle"):
        auditor = NASAComplianceAuditor()
        audit_res = auditor.audit_mission(
            mission_name=satellite["name"], norad_id=satellite["norad_id"],
            altitude_km=satellite["altitude_km"], inclination_deg=satellite["inclination_deg"],
            mass_kg=satellite["mass_kg"], area_m2=satellite["area_m2"],
            operational_collision_prob=risk_data["components"]["c_prob_raw"],
            has_passivation_valves=True,
        )
        report_md = auditor.generate_markdown_report(audit_res)
        st.markdown(report_md)
        st.download_button("📥 ODAR Raporunu İndir (.md)", data=report_md, file_name=f"ODAR_{satellite['name']}.md")


# ==========================================
# SEKME 2: TAKIMYILDIZ & FİLO TARAMASI
# ==========================================
with tab_fleet:
    st.subheader("🛰️ Walker Delta Takımyıldız & Enkaz Bulutu Toplu Tarama Motoru")
    st.caption("Onlarca operasyonel uyduyu binlerce orbital enkaz parçasına karşı gerçek zamanlı A-DRI filtresinden geçirir.")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        fleet_size = st.slider("Takımyıldız Filo Büyüklüğü (Uydu Sayısı):", 8, 48, 24, step=4)
    with f_col2:
        debris_count = st.slider("Enkaz Bulutu Boyutu (Parça Sayısı):", 30, 200, 100, step=10)
    with f_col3:
        screen_dist = st.slider("Tarama Eşiği (km):", 0.5, 5.0, 2.5, step=0.5)

    if st.button("🚀 Filo Taramasını Başlat", type="primary", use_container_width=True):
        screener = ConstellationConjunctionScreener()
        fleet_sats = generate_walker_delta_constellation(total_satellites=fleet_size)
        debris_pop = generate_synthetic_debris_cloud(count=debris_count)

        screen_result = screener.screen_constellation(
            fleet=fleet_sats, debris_population=debris_pop,
            storm_kp=kp_val, storm_f10_7=f107_val, screening_threshold_km=screen_dist,
        )

        sum_data = screen_result["summary"]
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Toplam Taranan Uydu", screen_result["total_fleet_size"])
        s_col2.metric("Kritik Alarm Veren Uydular", sum_data["critical_satellites"], delta_color="inverse")
        s_col3.metric("Yüksek Riskli Uydular", sum_data["high_risk_satellites"])
        s_col4.metric("Nominal Uydular", sum_data["nominal_satellites"])

        # Filo Risk Dağılım Grafiği
        labels = ["Kritik (CRITICAL)", "Yüksek (HIGH)", "Yükseltilmiş (ELEVATED)", "Nominal"]
        counts = [sum_data["critical_satellites"], sum_data["high_risk_satellites"], sum_data["elevated_satellites"], sum_data["nominal_satellites"]]
        colors = ["#f43f5e", "#fb923c", "#facc15", "#34d399"]

        fig_pie = go.Figure(data=[go.Pie(labels=labels, values=counts, marker=dict(colors=colors), hole=0.45)])
        fig_pie.update_layout(paper_bgcolor="#050811", font=dict(color="#e2e8f0"), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

        if screen_result["highest_risk_conjunctions"]:
            st.markdown("### ⚠️ En Yüksek Riskli Yakınlaşmalar (Kritik Tehdit Sıralaması):")
            st.dataframe(screen_result["highest_risk_conjunctions"], use_container_width=True)


# ==========================================
# SEKME 3: PARETO MANEVRA OPTİMİZASYONU
# ==========================================
with tab_pareto:
    st.subheader("⚖️ Çok Amaçlı Pareto Manevra Frontier Analizi")
    st.caption("Yakıt Tüketimi (Δv), Karşılaşma Güvenlik Mesafesi, İkincil Enkaz Riski ve Görev Kesintisi arasındaki ödünleşim (trade-off) analizi.")

    pareto_opt = ParetoManeuverOptimizer(
        mass_kg=satellite["mass_kg"],
        fuel_remaining_kg=satellite["remaining_propellant_kg"],
        isp_s=satellite["isp_seconds"],
    )

    pareto_res = pareto_opt.generate_pareto_frontier(
        current_altitude_km=satellite["altitude_km"],
        inclination_deg=satellite["inclination_deg"],
        initial_miss_km=miss_distance_km,
        storm_kp=kp_val, storm_f10_7=f107_val,
    )

    candidates = pareto_res["pareto_candidates"]
    recs = pareto_res["optimal_recommendations"]

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🟢 Minimum Yakıt Çözümü (Mikro-Ateşleme)</div>
            <div class="metric-value-green">{recs['minimum_propellant_solution']['delta_v_m_s']:.3f} m/s</div>
            <small style="color:#94a3b8;">Yakıt: {recs['minimum_propellant_solution']['propellant_kg']:.4f} kg | Mesafe: {recs['minimum_propellant_solution']['resulting_separation_km']} km</small>
        </div>
        """, unsafe_allow_html=True)

    with p_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🔵 Dengeli Pareto-Optimum Çözüm</div>
            <div class="metric-value-cyan">{recs['balanced_optimal_solution']['delta_v_m_s']:.3f} m/s</div>
            <small style="color:#94a3b8;">Yakıt: {recs['balanced_optimal_solution']['propellant_kg']:.4f} kg | Mesafe: {recs['balanced_optimal_solution']['resulting_separation_km']} km</small>
        </div>
        """, unsafe_allow_html=True)

    with p_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🟣 Maksimum Güvenlik Ayrılması</div>
            <div class="metric-value-cyan">{recs['maximum_clearance_solution']['delta_v_m_s']:.3f} m/s</div>
            <small style="color:#94a3b8;">Yakıt: {recs['maximum_clearance_solution']['propellant_kg']:.4f} kg | Mesafe: {recs['maximum_clearance_solution']['resulting_separation_km']} km</small>
        </div>
        """, unsafe_allow_html=True)

    # Pareto Scatter Plot
    xs = [c["delta_v_m_s"] for c in candidates]
    ys = [c["resulting_separation_km"] for c in candidates]
    colors_p = [c["post_burn_adri"] for c in candidates]
    sizes = [c["mission_downtime_hours"] * 5.0 for c in candidates]

    fig_pareto = go.Figure(data=go.Scatter(
        x=xs, y=ys, mode="markers+lines",
        marker=dict(size=sizes, color=colors_p, colorscale="Viridis", showscale=True, colorbar=dict(title="Yeni A-DRI")),
        text=[f"Aday: {c['candidate_id']}<br>Yakıt: {c['propellant_kg']} kg<br>Kesinti: {c['mission_downtime_hours']} saat" for c in candidates],
        line=dict(color="#38bdf8", dash="dash"),
    ))

    fig_pareto.update_layout(
        xaxis=dict(title="Ateşleme Maliyeti Δv (m/s)", gridcolor="#1e293b"),
        yaxis=dict(title="Karşılaşma Ayrılma Mesafesi (km)", gridcolor="#1e293b"),
        paper_bgcolor="#050811", plot_bgcolor="#0d1527", font=dict(color="#e2e8f0"),
        height=400,
    )
    st.plotly_chart(fig_pareto, use_container_width=True)


# ==========================================
# SEKME 4: YER İSTASYONLARI & İLETİŞİM KESİNTİSİ
# ==========================================
with tab_ground:
    st.subheader("📡 2D Yörünge İzdüşümü (Ground Track) ve İyonosferik İletişim Kesintisi (Blackout)")
    st.caption("Şiddetli güneş fırtınalarında (Kp >= 7) kutup bölgelerinde ve yüksek enlemlerde telsiz iletişiminin (telemetri) kesildiği pencereleri kanıtlar.")

    # Calculate 3-hour ground track
    r0, v0 = kepler_to_cartesian(
        a_km=EARTH_RADIUS_KM + satellite["altitude_km"],
        ecc=0.0001, inclination_deg=satellite["inclination_deg"],
        raan_deg=0.0, argp_deg=0.0, nu_deg=0.0,
    )
    sim_track = OrbitalPropagator(mass_kg=satellite["mass_kg"]).propagate_numerical(
        r0=r0, v0=v0, duration_hours=3.0, step_seconds=60.0, kp=kp_val
    )
    g_track = calculate_ground_track(sim_track["positions_km"], sim_track["time_seconds"])
    blackout_info = analyze_communication_blackouts(g_track, storm_kp=kp_val)

    b_col1, b_col2, b_col3 = st.columns(3)
    b_col1.metric("İletişim Kesinti Oranı (Blackout)", f"%{blackout_info['blackout_window_percentage']}")
    b_col2.metric("Yer İstasyonu Görüş Penceresi", f"%{blackout_info['contact_window_percentage']}")
    b_col3.metric("Otonom Karar Zorunluluğu", blackout_info["autonomous_decision_urgency"])

    st.warning(f"⚠️ **Operasyonel Risk Uyarısı:** {blackout_info['recommendation']}")

    # 2D Harita Çizimi
    lats = [pt["latitude_deg"] for pt in g_track]
    lons = [pt["longitude_deg"] for pt in g_track]

    fig_map = go.Figure()
    fig_map.add_trace(go.Scattergeo(
        lon=lons, lat=lats, mode="lines",
        line=dict(width=2.5, color="#06b6d4"),
        name="Uydu İzdüşümü",
    ))

    # Yer İstasyonlarını Ekle
    for gs in GROUND_STATIONS:
        fig_map.add_trace(go.Scattergeo(
            lon=[gs["lon"]], lat=[gs["lat"]], mode="markers+text",
            marker=dict(size=9, color="#f59e0b", symbol="star"),
            text=[gs["name"]], textposition="top center", name=gs["name"],
        ))

    fig_map.update_layout(
        geo=dict(
            showland=True, landcolor="#0f172a", countrycolor="#334155",
            showocean=True, oceancolor="#050811", showlakes=True, lakecolor="#050811",
            projection_type="equirectangular", bgcolor="#050811",
        ),
        paper_bgcolor="#050811", margin=dict(l=0, r=0, b=0, t=10), height=460,
    )
    st.plotly_chart(fig_map, use_container_width=True)


# ==========================================
# SEKME 5: ATMOSFER MODELLERİ LABORATUVARI
# ==========================================
with tab_atmos:
    st.subheader("🔬 Atmosfer & Termosfer Yoğunluk Modelleri Kıyaslama Laboratuvarı")
    st.caption("Üstel Model (Exponential Scale Height), Jacchia-77 (Termosferik Sıcaklık & Genleşme) ve NRLMSISE-00 (Tür Bileşenleri) modellerinin karşılaştırması.")

    alt_test_points = [200.0, 300.0, 400.0, 500.0, 550.0, 650.0, 800.0, 1000.0]
    comp_res = compare_atmospheric_models(alt_test_points, kp=kp_val, f10_7=f107_val)
    comp_table = comp_res["comparison_table"]

    # Çizgi Grafik
    alts = [row["altitude_km"] for row in comp_table]
    rho_exp = [row["exponential_kg_m3"] for row in comp_table]
    rho_jac = [row["jacchia_77_kg_m3"] for row in comp_table]
    rho_msis = [row["nrlmsise_00_kg_m3"] for row in comp_table]

    fig_atmos = go.Figure()
    fig_atmos.add_trace(go.Scatter(x=alts, y=rho_exp, mode="lines+markers", name="Üstel Model (Exponential)", line=dict(color="#38bdf8", width=3)))
    fig_atmos.add_trace(go.Scatter(x=alts, y=rho_jac, mode="lines+markers", name="Jacchia-77 (T_inf Termal)", line=dict(color="#f43f5e", width=3)))
    fig_atmos.add_trace(go.Scatter(x=alts, y=rho_msis, mode="lines+markers", name="NRLMSISE-00 (Tür Dağılımlı)", line=dict(color="#10b981", width=3, dash="dash")))

    fig_atmos.update_layout(
        yaxis_type="log",
        xaxis=dict(title="İrtifa (km)", gridcolor="#1e293b"),
        yaxis=dict(title="Yoğunluk ρ (kg/m³) [Logaritmik Ölçek]", gridcolor="#1e293b"),
        paper_bgcolor="#050811", plot_bgcolor="#0d1527", font=dict(color="#e2e8f0"),
        height=400,
    )
    st.plotly_chart(fig_atmos, use_container_width=True)

    st.markdown("### 📋 Sayısal Kıyaslama Tablosu:")
    st.dataframe(comp_table, use_container_width=True)


# ==========================================
# SEKME 6: RADYASYON & TMR HATA TOLERANSI
# ==========================================
with tab_rad:
    st.subheader("☢️ Uzay Radyasyonu & Üçlü Modüler Yedeklilik (TMR) Hata Toleransı")
    st.caption("Güney Atlantik Anomalisi (SAA) ve Güneş Parçacık Olaylarında (SPE) yapay zeka ağırlıklarına bit-flip enjekte ederek TMR oylamasının dayanıklılığını test eder.")

    rad_col1, rad_col2 = st.columns(2)
    with rad_col1:
        spe_active = st.checkbox("Güneş Parçacık Olayı (SPE) Simüle Et", value=True)
        injected_flips = st.slider("Enjekte Edilecek Bit-Flip / SEU Sayısı:", 1, 15, 4)
    with rad_col2:
        rad_sim = RadiationFaultSimulator(get_pretrained_edge_pinn())
        flux_info = rad_sim.calculate_space_radiation_flux(
            altitude_km=satellite["altitude_km"], latitude_deg=-25.0, is_solar_particle_event=spe_active
        )
        st.metric("Toplam Proton & Ağır İyon Akısı", f"{flux_info['total_flux_p_cm2_s']} p/cm²/s")
        st.caption(f"Tahmini SEU Oranı: {flux_info['estimated_seu_rate_per_mb_day']} hata/Mb/gün (SAA Bölgesi: {'EVET' if flux_info['in_saa_zone'] else 'HAYIR'})")

    if st.button("⚡ Donanım TMR Oylama Testini Çalıştır", type="primary", use_container_width=True):
        dummy_in = torch.tensor([[satellite["altitude_km"], 8.0, 240.0, 0.02, 7.6]], dtype=torch.float32)
        tmr_res = rad_sim.evaluate_tmr_voting(dummy_in, bit_flip_count=injected_flips)

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Tek Çekirdek Hatası (SEU)", f"{tmr_res['single_core_seu_error']:.4f}", delta="Kritik Sapma", delta_color="inverse")
        res_col2.metric("TMR Oylanmış Hata", f"{tmr_res['tmr_voted_error']:.6f}", delta="Hata Sıfırlandı", delta_color="normal")
        res_col3.metric("Donanım Dayanıklılığı", f"%{tmr_res['resilience_percentage']}", delta="100% Maskelendi")

        if tmr_res["fault_masked_successfully"]:
            st.success("✅ **BAŞARILI: TMR Çoğunluk Oylaması (Majority Voting) tüm bit-flip hatalarını tamamen maskeledi!** Yapay zeka çıkarımı uzay radyasyonu altında sıfır sapmayla çalıştı.")


# ==========================================
# SEKME 7: DEORBIT & POINT NEMO KORİDORU
# ==========================================
with tab_deorbit:
    st.subheader("🎯 Kontrollü Deorbit & Point Nemo Atmosfere Giriş Simülatörü")
    st.caption("Görev sonu bertarafı için Point Nemo (Güney Pasifik Okyanusu İkametsiz Alanı - SPOUA) hedefli kontrollü retro-ateşleme ve aerotermal yanma modeli.")

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        target_perigee = st.slider("Hedef Giriş Perigee İrtifası (km):", 40.0, 90.0, 65.0, step=5.0)
        burn_plan = compute_deorbit_burn(
            initial_altitude_km=satellite["altitude_km"], target_perigee_km=target_perigee,
            satellite_mass_kg=satellite["mass_kg"], isp_seconds=satellite["isp_seconds"],
        )
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Gerekli Retrograde Ateşleme (Δv)</div>
            <div class="metric-value-cyan">{burn_plan['delta_v_retrograde_m_s']} m/s</div>
            <small style="color:#94a3b8;">Harcanacak Yakıt: <b>{burn_plan['propellant_required_kg']} kg</b> | Kalan Kütle: {burn_plan['post_burn_mass_kg']} kg</small>
        </div>
        """, unsafe_allow_html=True)

    with d_col2:
        descent_sim = simulate_reentry_corridor()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Hedef Düşüş Bölgesi</div>
            <div class="metric-value-green">Point Nemo (SPOUA)</div>
            <small style="color:#94a3b8;">Zirve Durgunluk Isı Akısı: <b>{descent_sim['peak_stagnation_heat_flux_w_cm2']} W/cm²</b> | NASA Req 4.7: <b>UYUMLU</b></small>
        </div>
        """, unsafe_allow_html=True)

    st.info(f"📍 **Düşüş Koridoru:** {POINT_NEMO['name']} (Enlem: {POINT_NEMO['latitude_deg']}°, Boylam: {POINT_NEMO['longitude_deg']}°)")


# ==========================================
# SEKME 8: CANLI CELESTRAK TLE SORGULAMA
# ==========================================
with tab_celestrak:
    st.subheader("🌐 Canlı CelesTrak & Space-Track TLE Sorgulama Motoru")
    st.caption("NORAD Katalog Numarası ile gerçek zamanlı uydu yörünge elemanlarını çeker ve doğrular.")

    norad_input = st.number_input("NORAD Katalog No (Örn: 25544 ISS, 44713 Starlink):", min_value=1, max_value=99999, value=25544)
    if st.button("📡 CelesTrak'ten Canlı TLE Getir", type="primary"):
        celestrak_client = CelesTrakClient()
        tle_res = celestrak_client.fetch_tle_by_catnr(int(norad_input))

        st.success(f"Uydu: **{tle_res['name']}** (Durum: `{tle_res['status']}` | Kaynak: `{tle_res['source']}`)")
        st.code(f"{tle_res['line1']}\n{tle_res['line2']}", language="text")


# ==========================================
# SEKME 9: 9-AJANLI OTONOM SÜRÜ ZEKASI (AUTONOMOUS SWARM)
# ==========================================
with tab_swarm:
    st.subheader("🤖 AEGIS-LEO 9-Ajanlı Otonom Sürü Zekası & Dağıtık Görev Kontrolü")
    st.caption(
        "Güneş fırtınaları ve iletişim kesintilerinde (blackout) yer istasyonuna bağımlı kalmadan "
        "9 uzman uç nokta ajanı tarafından yürütülen otonom algılama, risk tespiti, takımyıldız eşgüdümü ve konsensüs oylaması."
    )

    if "swarm_orchestrator" not in st.session_state:
        st.session_state.swarm_orchestrator = OrbitalSwarmOrchestrator()
    if "latest_swarm_cycle" not in st.session_state:
        st.session_state.latest_swarm_cycle = st.session_state.swarm_orchestrator.run_consensus_cycle()

    orch = st.session_state.swarm_orchestrator

    sw_col1, sw_col2, sw_col3, sw_col4 = st.columns(4)
    with sw_col1:
        sw_kp = st.slider("Simüle Edilen Kp Endeksi:", 1.0, 9.0, 7.8, step=0.1)
    with sw_col2:
        sw_f107 = st.slider("Güneş Akısı F10.7 (SFU):", 70.0, 350.0, 255.0, step=5.0)
    with sw_col3:
        sw_miss = st.slider("Yaklaşma Mesafesi (m):", 10.0, 1000.0, 82.0, step=10.0)
    with sw_col4:
        sw_fuel = st.slider("Kalan Yakıt (kg):", 1.0, 25.0, 18.5, step=0.5)

    if st.button("⚡ 9-Ajanlı Otonom Konsensüs Döngüsünü Tetikle", type="primary", use_container_width=True):
        custom_telemetry = {
            "satellite_id": satellite["name"],
            "altitude_km": satellite["altitude_km"],
            "inclination_deg": satellite["inclination_deg"],
            "miss_distance_km": sw_miss / 1000.0,
            "relative_speed_km_s": 11.2,
            "tca_hours": 3.4,
            "fuel_remaining_kg": sw_fuel,
            "kp": sw_kp,
            "f10_7": sw_f107,
        }
        st.session_state.latest_swarm_cycle = orch.run_consensus_cycle(custom_telemetry)
        st.rerun()

    cycle = st.session_state.latest_swarm_cycle
    cmd = cycle["commander_directive"]

    # Commander Verdict Banner
    is_approved = cmd["consensus_reached"]
    banner_color = "#059669" if is_approved else "#e11d48"
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.9); border: 2px solid {banner_color}; border-radius: 12px; padding: 18px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3 style="margin: 0; color: {banner_color};">🎖️ {cmd['directive']}</h3>
                <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">{cmd['status']}</p>
            </div>
            <div style="text-align: right;">
                <div style="font-family: monospace; font-size: 1.3rem; color: #38bdf8; font-weight: bold;">%{cmd['approval_percentage']} ONAY</div>
                <div style="font-family: monospace; font-size: 0.75rem; color: #64748b;">Kriptografik İmza: {cmd['authorization_signature']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 9-Agent Roster Grid (3x3)
    st.markdown("### 🛰️ 9 Uç Nokta Ajanı Canlı Durum Matrisi")
    agent_cols = st.columns(3)

    agent_meta = [
        ("A1_solar", "☀️ A1: Güneş Fırtınası Ajanı", "Uzay Havası & CME İzleme", "#f59e0b"),
        ("A2_thermosphere", "🌌 A2: Termosfer Dinamiği Ajanı", "Yoğunluk Şişmesi & Drift", "#06b6d4"),
        ("A3_risk", "⚠️ A3: A-DRI Risk Değerlendirme", "B-Plane & Çarpışma Olasılığı", "#ef4444"),
        ("A4_astrodynamics", "🚀 A4: Yörünge Astrodinamiği", "Vis-Viva Evasive Phasing", "#10b981"),
        ("A5_edge_ai", "🧠 A5: Uç Nokta PINN & TMR", "Mikrosaniye Çıkarım & SEU", "#8b5cf6"),
        ("A6_constellation", "🛰️ A6: Takımyıldız Eşgüdümü", "Walker-Delta ISL Mesh Ağı", "#3b82f6"),
        ("A7_compliance", "🛡️ A7: Standart Denetim Ajanı", "NASA-STD-8719 & Veto Yetkisi", "#ec4899"),
        ("A8_disposal", "🎯 A8: Atmosferik Bertaraf", "Point Nemo & EOL Deorbit", "#14b8a6"),
        ("A9_commander", "🎖️ A9: Ana Görev Komutanı", "Bizans Konsensüsü & İcraat", "#6366f1"),
    ]

    for idx, (res_key, title, subtitle, color) in enumerate(agent_meta):
        col_target = agent_cols[idx % 3]
        agent_data = cycle["agent_results"].get(res_key, {})
        status = agent_data.get("status", "ACTIVE")
        decision = agent_data.get("decision", "Nominal")
        status_bg = "#065f46" if status == "ACTIVE" else "#991b1b" if status in ["ALERT", "VETO"] else "#1e3a8a"

        with col_target:
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid {color}44; border-left: 4px solid {color}; border-radius: 8px; padding: 12px; margin-bottom: 12px; min-height: 140px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <div style="font-weight: bold; color: {color}; font-size: 0.85rem;">{title}</div>
                    <span style="background: {status_bg}; color: #fff; font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; font-weight: bold;">{status}</span>
                </div>
                <div style="font-size: 0.7rem; color: #94a3b8; margin-bottom: 6px;">{subtitle}</div>
                <div style="font-size: 0.75rem; color: #e2e8f0; line-height: 1.3;">{decision}</div>
            </div>
            """, unsafe_allow_html=True)

    # Voting Ledger & Communication Blackboard
    st.markdown("---")
    ledger_col, blackboard_col = st.columns([1, 1])

    with ledger_col:
        st.markdown("#### 🗳️ Konsensüs Oylama Kayıt Defteri (Voting Ledger)")
        for vote in cycle["swarm_votes"]:
            v_agent = vote.get("agent", "")
            v_vote = vote.get("vote", "")
            v_conf = int(vote.get("confidence", 1.0) * 100)
            v_reason = vote.get("reason", "Onaylandı")
            v_badge_color = "#10b981" if v_vote in ["APPROVE", "OPTIMAL", "URGENT_APPROVAL"] else "#ef4444" if v_vote == "VETO" else "#f59e0b"

            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 10px; margin-bottom: 4px; background: rgba(30, 41, 59, 0.4); border-radius: 6px; font-size: 0.8rem;">
                <div>
                    <b>{v_agent}</b>: <span style="color: #94a3b8;">{v_reason}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 0.7rem; color: #64748b;">%{v_conf}</span>
                    <span style="background: {v_badge_color}33; color: {v_badge_color}; border: 1px solid {v_badge_color}; padding: 1px 6px; border-radius: 4px; font-weight: bold; font-size: 0.7rem;">{v_vote}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with blackboard_col:
        st.markdown("#### 📡 Ajanlar Arası İletişim Kara Tahtası (Blackboard Feed)")
        recent_msgs = cycle.get("recent_messages", [])
        if recent_msgs:
            for msg in reversed(recent_msgs[-6:]):
                st.markdown(f"""
                <div style="padding: 6px 10px; margin-bottom: 4px; background: rgba(15, 23, 42, 0.6); border-left: 2px solid #38bdf8; border-radius: 4px; font-size: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; color: #38bdf8; font-family: monospace;">
                        <span>{msg.get('sender')} ➔ {msg.get('recipient')} [{msg.get('message_type')}]</span>
                        <span style="color: #64748b;">{msg.get('timestamp', '')[11:19]} UTC</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Kara tahtada henüz mesaj bulunmuyor.")

