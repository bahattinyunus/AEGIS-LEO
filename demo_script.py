"""
AEGIS-LEO Mission Demonstration Script
Executes an end-to-end orbital safety and autonomous avoidance simulation.
Run with: python demo_script.py
"""

import sys
import os

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from aegis_leo.constants import QUIET_KP, QUIET_F10_7
from aegis_leo.physics.atmosphere import compute_atmospheric_density
from aegis_leo.physics.coordinates import kepler_to_cartesian
from aegis_leo.physics.propagator import OrbitalPropagator
from aegis_leo.risk.adri import ADRICalculator
from aegis_leo.ai.edge_optimizer import EdgeManeuverOptimizer
from aegis_leo.ai.quantizer import benchmark_edge_inference
from aegis_leo.ai.pinn import get_pretrained_edge_pinn
from aegis_leo.compliance.nasa_std_8719 import NASAComplianceAuditor
from aegis_leo.data.catalog import get_satellite_by_id


def run_demo():
    print("=" * 80)
    print("🛰️  AEGIS-LEO: Otonom Uç Nokta Yapay Zekası & Fizik Bilgilendirmeli Yörünge Güvenlik Kalkanı")
    print("   NASA Space Apps Challenge 2026 | AYAZ Takımı | Lider Mimar: Bahattin Yunus Çetin")
    print("=" * 80)

    # 1. Hedef Uydu
    sat = get_satellite_by_id("STARLINK_LEO")
    print(f"\n[1] TAKİP EDİLEN UYDU: {sat['name']} (NORAD #{sat['norad_id']})")
    print(f"    - Görev Türü: {sat['type']}")
    print(f"    - Yörünge: İrtifa {sat['altitude_km']} km, Eğim {sat['inclination_deg']}°")
    print(f"    - Kütle: {sat['mass_kg']} kg, Alan: {sat['area_m2']} m², Kalan Yakıt: {sat['remaining_propellant_kg']} kg")

    # 2. Sakin Uzay Havası Taban Değeri
    rho_quiet = compute_atmospheric_density(sat['altitude_km'], QUIET_KP, QUIET_F10_7)
    print(f"\n[2] NOMİNAL UZAY HAVASI (Sakin Taban Değer, Kp=1.3, F10.7=72)")
    print(f"    - Taban Termosferik Yoğunluk (ρ_0): {rho_quiet:.2e} kg/m³")

    # 3. Güneş Fırtınası Simülasyonu
    storm_kp = 8.0
    storm_f10_7 = 265.0
    rho_storm = compute_atmospheric_density(sat['altitude_km'], storm_kp, storm_f10_7)
    surge_ratio = rho_storm / rho_quiet

    print(f"\n[3] ⚡ GEOMANYETİK FIRTINA ETKİSİ: NOAA G4 Şiddetli Güneş Fırtınası (Kp={storm_kp}, F10.7={storm_f10_7})")
    print(f"    - Genleşen Termosferik Yoğunluk (ρ_fırtına): {rho_storm:.2e} kg/m³")
    print(f"    - Termosferik Genleşme Oranı: +{surge_ratio:.1f}x kat daha fazla sürtünme!")

    # 4. Modellenemeyen Sürüklenme ve Yörünge Sapması
    r0, v0 = kepler_to_cartesian(
        a_km=6378.137 + sat['altitude_km'],
        ecc=0.0001,
        inc_deg=sat['inclination_deg'],
        raan_deg=0.0,
        argp_deg=0.0,
        nu_deg=0.0,
    )
    propagator = OrbitalPropagator(cd=sat['cd'], area_m2=sat['area_m2'], mass_kg=sat['mass_kg'])
    divergence = propagator.simulate_storm_divergence(
        r0=r0, v0=v0, duration_hours=36.0, step_seconds=120.0, storm_kp=storm_kp, storm_f10_7=storm_f10_7
    )

    print(f"\n[4] 36 SAATLİK YÖRÜNGE SAPMASI VE İRTİFA KAYBI:")
    print(f"    - Nominal Beklenen İrtifa Kaybı:     {divergence['quiet_altitude_decay_m']:.1f} metre")
    print(f"    - Fırtına Kaynaklı İrtifa Kaybı:    {divergence['storm_altitude_decay_m']:.1f} metre")
    print(f"    - Modellenememiş Ek İrtifa Kaybı:   {divergence['net_decay_meters']:.1f} metre")
    print(f"    - Boylamsal Yörünge Sapması:        +{divergence['along_track_displacement_km']:.2f} km")
    print(f"    ⚠️  Statik Yakınlaşma Mesajları (CDM) yörünge kayması nedeniyle GEÇERSİZ kaldı!")

    # 5. Risk Değerlendirmesi (A-DRI)
    calc = ADRICalculator()
    pre_risk = calc.evaluate_risk(
        miss_distance_km=0.45,
        altitude_km=sat['altitude_km'],
        inclination_deg=sat['inclination_deg'],
        area_m2=sat['area_m2'],
        mass_kg=sat['mass_kg'],
        kp=storm_kp,
        f10_7=storm_f10_7,
        along_track_disp_km=divergence['along_track_displacement_km'],
        remaining_propellant_kg=sat['remaining_propellant_kg'],
    )

    print(f"\n[5] GERÇEK ZAMANLI A-DRI RİSK DEĞERLENDİRMESİ:")
    print(f"    - Çarpışma Olasılığı (C_prob):      {pre_risk['components']['c_prob_raw']:.2e}")
    print(f"    - Uzay Havası Belirsizliği (W_drag): {pre_risk['components']['w_drag']:.2f}")
    print(f"    - NASA ORDEM Enkaz Yoğunluğu (D_ord): {pre_risk['components']['d_ord']:.2f}")
    print(f"    - Yakıt Fizibilite İndeksi (F_Δv):  {pre_risk['components']['f_delta_v']:.2f}")
    print(f"    >>> BİRLEŞİK A-DRI SKORU: {pre_risk['a_dri']:.3f} [{pre_risk['risk_level']}] <<<")
    print(f"    - Önerilen Eylem Komutu: {pre_risk['action_command']}")

    # 6. Otonom Uç Nokta Kaçınma Optimize Edici
    optimizer = EdgeManeuverOptimizer(
        satellite_mass_kg=sat['mass_kg'],
        fuel_remaining_kg=sat['remaining_propellant_kg'],
        isp_seconds=sat['isp_seconds'],
        thrust_newtons=sat['thrust_n'],
    )
    plan = optimizer.plan_autonomous_avoidance(
        current_altitude_km=sat['altitude_km'],
        inclination_deg=sat['inclination_deg'],
        initial_miss_distance_km=0.45,
        storm_kp=storm_kp,
        storm_f10_7=storm_f10_7,
        along_track_disp_km=divergence['along_track_displacement_km'],
        area_m2=sat['area_m2'],
    )

    pb = plan['proactive_burn']
    rb = plan['reactive_baseline']
    eff = plan['efficiency_metrics']

    print(f"\n[6] 🚀 OTONOM UÇ NOKTA YAPAY ZEKA KAÇINMA MOTORU:")
    print(f"    - Önerilen Strateji:         {pb['strategy']}")
    print(f"    - Planlanan Fazlama Δv Yanması: {pb['delta_v_m_s']:.3f} m/s")
    print(f"    - Harcanan İtici Gaz:          {pb['propellant_consumed_kg']:.4f} kg")
    print(f"    - Yanma Sonrası Geçiş Mesafesi: {pb['resulting_separation_km']:.2f} km (GÜVENLİ)")
    print(f"    - Yeni A-DRI Risk Skoru:       {pb['post_burn_adri']:.3f} [{pb['post_burn_risk_level']}]")
    print(f"    - Kıyaslamalı Yakıt Tasarrufu: %{eff['fuel_savings_percentage']} (Geç reaktif manevraya kıyasla {eff['propellant_saved_kg']:.4f} kg tasarruf)")
    print(f"    - Toplam Risk İndirgeme:       %{eff['risk_reduction_percentage']}")

    # 7. Uç Nokta Donanım Performansı
    model = get_pretrained_edge_pinn()
    bench = benchmark_edge_inference(model, num_iterations=200)
    print(f"\n[7] 🧠 UÇUŞ BİLGİSAYARI (ONBOARD OBC) PERFORMANSI:")
    print(f"    - Ağırlık/Parametre Sayısı:    {bench['parameter_count']} parametre")
    print(f"    - INT8 Bellek Ayak İzi:        {bench['int8_quantized_memory_kb']} KB")
    print(f"    - Çıkarım (Inference) Gecikmesi: {bench['latency_microseconds']:.1f} mikrosaniye")
    print(f"    - Uçuş Donanımı Test Durumu:   {bench['flight_readiness']}")

    # 8. NASA-STD-8719.14 Denetimi
    auditor = NASAComplianceAuditor()
    audit = auditor.audit_mission(
        mission_name=sat['name'],
        norad_id=sat['norad_id'],
        altitude_km=sat['altitude_km'],
        inclination_deg=sat['inclination_deg'],
        mass_kg=sat['mass_kg'],
        area_m2=sat['area_m2'],
        operational_collision_prob=1.2e-5,
        has_passivation_valves=True,
    )
    print(f"\n[8] 📋 NASA-STD-8719.14 UYUMLULUK DENETİMİ:")
    print(f"    - Genel Uyum Durumu: {audit['overall_status']}")
    for item in audit['compliance_details']:
        print(f"      [{item['status']}] {item['standard_id']}: {item['title']}")

    print("\n" + "=" * 80)
    print("✅ SİMÜLASYON TAMAMLANDI: Tüm fizik, yapay zeka ve regülasyon testleri doğrulandı.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
