# 🛡️ AEGIS-LEO: Otonom Uç Nokta Yapay Zekası ve Fizik Bilgilendirmeli Yörünge Güvenlik Kalkanı

[![NASA Space Apps 2026](https://img.shields.io/badge/NASA%20Space%20Apps-2026%20Ankara-0B3D91.svg?style=for-the-badge&logo=nasa&logoColor=white)](https://www.spaceappschallenge.org/)
[![Takım](https://img.shields.io/badge/Takım-AYAZ-orange.svg?style=for-the-badge)]()
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)]()
[![Testler](https://img.shields.io/badge/Testler-352%2F352%20Geçti-brightgreen.svg?style=for-the-badge)]()
[![Lisans: MIT](https://img.shields.io/badge/Lisans-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="assets/aegis_leo_hero_banner.jpg" alt="AEGIS-LEO Görev Bannerı" width="100%" style="border-radius: 14px; box-shadow: 0 12px 32px rgba(6,182,212,0.35);" />
</p>

> **"Uzay Havası Fiziği, Yörünge Mekaniği ve Otonom Uç Nokta Yapay Zekasını Birleştirerek Alçak Dünya Yörüngesini (LEO) Koruyoruz."**

**AEGIS-LEO** (*Autonomous Edge-AI & Physics-Informed Orbital Safety Shield*), **NASA Space Apps Challenge 2026** için **AYAZ Takımı** tarafından geliştirilen uçtan uca yörünge risk istihbaratı, takımyıldız taraması ve otonom çarpışma önleme platformudur.

Platform; **Güneş fırtınalarının (CME) tetiklediği ani termosferik genleşmeyi**, uydularda oluşan öngörülemeyen boylamsal yörünge kaymalarını (along-track drift), statik Yakınlaşma Veri Mesajlarının (CDM) geçersiz kalmasını ve Kessler Sendromu riskini çözmek için uzay ajansı seviyesinde (NASA/ESA/CCSDS) mühendislik altyapısı sunar.

---

## 📑 İçindekiler
- [🎯 Çözülen Kritik Problem & Tarihsel Arka Plan](#-çözülen-kritik-problem--tarihsel-arka-plan)
- [🤖 9-Ajanlı Otonom Sürü Zekası Mimarisi (Swarm Intelligence)](#-9-ajanlı-otonom-sürü-zekası-mimarisi-swarm-intelligence)
- [⚡ Temel İnovasyonlar & Bilimsel Derinlik](#-temel-inovasyonlar--bilimsel-derinlik)
  - [1. Çok Modelli Termosferik Sürüklenme Fiziği](#1-çok-modelli-termosferik-sürüklenme-fiziği)
  - [2. Uyarlanabilir Risk İndeksi (A-DRI) Formülasyonu](#2-uyarlanabilir-risk-indeksi-a-dri-formülasyonu)
  - [3. Uç Nokta Yapay Zekası & Pareto Manevra Optimizasyonu](#3-uç-nokta-yapay-zekası--pareto-manevra-optimizasyonu)
  - [4. Walker Delta Takımyıldız & Enkaz Taraması](#4-walker-delta-takımyıldız--enkaz-taraması)
  - [5. Yer İstasyonları & İletişim Kesintisi (Blackout)](#5-yer-istasyonları--iletişim-kesintisi-blackout)
- [🏗️ Sistem Mimarisi](#️-sistem-mimarisi)
- [🖥️ 9 Sekmeli Görev Kontrol Simülatörü](#️-9-sekmeli-görev-kontrol-simülatörü)
- [💻 Profesyonel Aegis CLI Rehberi](#-profesyonel-aegis-cli-rehberi)
- [🌐 REST API Dokümantasyonu (FastAPI)](#-rest-api-dokümantasyonu-fastapi)
- [📋 NASA-STD-8719.14 Uyumluluk Matrisi](#-nasa-std-871914-uyumluluk-matrisi)
- [📊 Vaka Analizi: NOAA G4/G5 Fırtınası Stres Testi](#-vaka-analizi-noaa-g4g5-fırtınası-stres-testi)
- [🧪 Doğrulama & Test Paketi](#-doğrulama--test-paketi)
- [🐳 Docker Konteynerizasyonu](#-docker-konteynerizasyonu)
- [👥 AYAZ Takımı & Lisans](#-ayaz-takımı--lisans)


---

## 🎯 Çözülen Kritik Problem & Tarihsel Arka Plan

Alçak Dünya Yörüngesinde (LEO: 160 – 2.000 km) görev yapan binlerce uydu, standart operasyonlarda yer tabanlı radar ağlarından (Space-Track / 18th SDS) gelen **statik Yakınlaşma Veri Mesajlarına (CDM)** güvenir. Ancak bu sistemler uzay havasındaki ani fırtınaları hesaba katmaz:

* **Tarihsel Uyarı (Şubat 2022 Starlink Olayı):** Fırlatılan 49 Starlink uydusundan **38 tanesi**, orta şiddette (G1/G2) bir jeomanyetik fırtınanın atmosfer yoğunluğunu beklenmedik şekilde %50 artırması sonucu frenlenmiş, yörüngeden çıkarak atmosferde yanmıştır.
* **Mayıs 2024 G5 Süper Fırtınası:** Son 20 yılın en güçlü Güneş fırtınasında termosfer yoğunluğu 6 ila 8 kat genleşmiş, LEO'daki binlerce uydu boylamsal eksende günde onlarca kilometre kayarak mevcut tüm çarpışma tahminlerini geçersiz kılmıştır.
* **Kör Nokta:** Fırtına esnasında kutup auroral iyonosferik emilimi (PCA) sebebiyle yer istasyonlarıyla uydular arasındaki telsiz telemetri bağlantısı kesilmektedir. Yerden komut verilemediği bu kritik pencerelerde, **uydunun kendi kendine karar verip kaçınma yapabilmesi şarttır.**

**AEGIS-LEO**, uydunun uçuş bilgisayarına (Onboard Edge-AI) yerleştirilen yapay zeka ve fizik motoru sayesinde bu problemi kökten çözer.

---

## 🤖 9-Ajanlı Otonom Sürü Zekası Mimarisi (Swarm Intelligence)

Güneş fırtınaları sırasında iyonosferik plazma tabakasının aşırı ısınması ve kutup auroral emilimi (PCA), yer istasyonları ile LEO uyduları arasındaki telsiz telemetri bağlantısını tamamen keser (**Ground Communication Blackout**). Bu kritik pencerede uyduların yer kontrol merkezini beklemesi felaketle sonuçlanır. 

AEGIS-LEO, uzay aracının uçuş yazılımı (FSW) içerisinde eşzamansız çalışan **9 uzmanlaşmış otonom ajandan oluşan dağıtık bir sürü zekası** mimarisi içerir:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│               AEGIS-LEO 9-AJANLI DAĞITIK UÇUŞ YAZILIMI SÜRÜ ZEKASI HİYERARŞİSİ        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                        │
│  [FAZ 1: ALGILAMA]        ☀️ A1: SolarStormAgent        🌌 A2: ThermosphereAgent       │
│                                      │                             │                   │
│                                      ▼                             ▼                   │
│  [FAZ 2: RİSK TESPİTİ]                       ⚠️ A3: RiskAssessmentAgent                │
│                                                            │                           │
│                                      ┌─────────────────────┴─────────────────────┐     │
│                                      ▼                                           ▼     │
│  [FAZ 3: MANEVRA & YAPAY ZEKA] 🚀 A4: AstrodynamicsAgent     🧠 A5: EdgeInferenceAgent │
│                                      │                                           │     │
│                                      └─────────────────────┬─────────────────────┘     │
│                                                            ▼                           │
│  [FAZ 4: EŞGÜDÜM & DENETİM]    🛰️ A6: ConstellationAgent    🛡️ A7: ComplianceWatchdog  │
│                                                            🎯 A8: ReentryDisposal      │
│                                                            │                           │
│                                                            ▼                           │
│  [FAZ 5: BİZANS KONSENSÜSÜ]                 🎖️ A9: MissionCommanderAgent              │
│                                          (Kriptografik İmza & Ateşleme Emri)           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### 🛰️ 9 Uç Nokta Ajanının Görev ve Yetki Matrisi:

| Ajan No | Ajan Adı & Rolü | Sorumluluk Alanı | Ağırlık | Veto Yetkisi |
| :---: | :--- | :--- | :---: | :---: |
| **A1** | `SolarStormAgent-01` | NOAA SWPC güneş rüzgarı, CME hızı, proton akısı ve Kp endeksini izler; fırtına uyarısı yayınlar. | 1.0x | Yok |
| **A2** | `ThermosphereAgent-02` | Jacchia-77/NRLMSISE-00 ile termosferik şişmeyi ve uydunun boylamsal sürüklenme sapmasını (along-track drift) hesaplar. | 1.0x | Yok |
| **A3** | `RiskAssessmentAgent-03` | Gerçek zamanlı A-DRI skorunu, B-Düzlemi kovaryans elipsoidlerini ve $P_c$ çarpışma olasılığını belirler. | 1.5x | Yok |
| **A4** | `AstrodynamicsAgent-04` | Vis-Viva mikro-fazlama manevralarını, teğetsel itki vektörlerini ve minimum yakıtlı kaçınma yörüngesini sentezler. | 1.5x | Yok |
| **A5** | `EdgeInferenceAgent-05` | Radyasyon dayanımlı uçuş çipinde (ARM Cortex-R5F / Xilinx RT-Kintex) <150 µs PINN çıkarımı yapar ve TMR çoğunluk oylaması ile SEU bit-flip hatalarını maskeler. | 1.2x | Yok |
| **A6** | `ConstellationConsensusAgent-06` | Uydular arası lazer/radyo veri bağlarını (ISL) yönetir; planlanan manevranın komşu takımyıldız yuvalarında (Walker-Delta) ikincil çarpışma zinciri yaratmayacağını teyit eder. | 1.0x | Yok |
| **A7** | `ComplianceWatchdogAgent-07` | **Bağımsız Denetçi:** Manevranın NASA-STD-8719.14B, IADC ve ISO 24113 kurallarına uygunluğunu, en az 2.0 kg pasivasyon yakıt rezervinin korunduğunu denetler. | **2.0x** | **🚨 MUTLAK VETO** |
| **A8** | `ReentryDisposalAgent-08` | Görev sonu (EOL) deorbit süresini izler; acil durumda Güney Pasifik Okyanusu İkametsiz Alanı'na (Point Nemo - SPOUA) kontrollü düşüş için $\Delta v$ bütçesini rezerve eder. | 1.0x | Yok |
| **A9** | `MissionCommanderAgent-09` | 8 uzmandan gelen oyları Bizans Hata Toleranslı (BFT) konsensüs protokolü ile toplar; en az %70 ağırlıklı onay sağlandığında SHA-256 dijital imzalı otonom manevra emrini (`EXECUTE_AUTONOMOUS_MANEUVER`) yayınlar. | 1.0x | Nihai Karar |

---


## ⚡ Temel İnovasyonlar & Bilimsel Derinlik

### 1. Çok Modelli Termosferik Sürüklenme Fiziği
AEGIS-LEO; **NASA OMNIWeb** ve **NOAA SWPC** güneş rüzgarı akılarını işleyerek üst atmosfer yoğunluğunu ($\rho$) ve aerodinamik sürtünme ivmesini dinamik olarak hesaplar:

$$a_{drag} = -\frac{1}{2} \rho(h, K_p, F_{10.7}) \left(\frac{C_d A}{m}\right) v_{rel}^2 \hat{v}_{rel}$$

<p align="center">
  <img src="assets/space_weather_thermosphere_drag.jpg" alt="Güneş Fırtınası Termosferik Genleşme Simülasyonu" width="100%" style="border-radius: 12px; box-shadow: 0 8px 24px rgba(244,63,94,0.3);" />
</p>

Platform 3 farklı termosfer yoğunluk modelini aynı anda çalıştırıp karşılaştırabilir:
1. **Üstel Ölçek Modeli (Exponential Scale-Height):** Hızlı analitik referans.
2. **Jacchia-77 Termal Modeli:** $T_\infty = 379 + 3.24\bar{F}_{10.7} + 1.3(F_{10.7} - \bar{F}_{10.7}) + 28 K_p + 0.03 e^{K_p}$ denklemiyle Güneş fırtınası sıcaklık artışını ve atmosfer şişmesini modeller.
3. **NRLMSISE-00 Modeli:** $O, N_2, O_2, He, H$ kimyasal bileşenlerinin kısmi basınç ve difüzyon denge yüksekliklerini irtifaya göre çözer.

---

### 2. Uyarlanabilir Risk İndeksi (A-DRI) Formülasyonu
Takip edilen her nesne için gerçek zamanlı $0.0 \to 1.0$ ölçeğinde normalize edilmiş risk puanı üretilir:

$$\text{A-DRI} = w_1 C_{prob} + w_2 W_{drag} + w_3 D_{ord} + w_4 F_{\Delta v}$$

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ A-DRI BİLEŞENLERİ VE AĞIRLIK DAĞILIMI                                     │
├────────────────────────────────────────────────────────────────────────────┤
│ • C_prob (0.40): 3D-2D B-Düzlemi Gauss Kovaryans Çarpışma Olasılığı        │
│ • W_drag (0.25): Alan/Kütle Oranı (A/m) & Güneş Fırtınası Sürüklenme Sapması│
│ • D_ord  (0.20): NASA ORDEM 3.1 İrtifa/Eğim Mekânsal Enkaz Yoğunluk Modeli│
│ • F_Δv   (0.15): Kalan Yakıt Marjı ve Geç Manevra Maliyet Cezası           │
└────────────────────────────────────────────────────────────────────────────┘
```

| A-DRI Skoru | Risk Seviyesi | Otonom Eylem Komutu |
| :---: | :---: | :--- |
| **0.00 – 0.29** | 🟢 **NOMİNAL** | `HOLD_NOMINAL` (Rutin izleme) |
| **0.30 – 0.59** | 🟡 **YÜKSELTİLMİŞ** | `OBSERVE_CLOSE_TRACK` (Takip sıklığını artır) |
| **0.60 – 0.79** | 🟠 **YÜKSEK** | `SCHEDULE_PROACTIVE_MICRO_BURN` (Erken fazlama ateşlemesi planla) |
| **0.80 – 1.00** | 🔴 **KRİTİK** | `EMERGENCY_AUTONOMOUS_EVASION` (Derhal acil durum kaçınmasını ateşle) |

---

### 3. Uç Nokta Yapay Zekası & Pareto Manevra Optimizasyonu
Radyasyon dayanımlı uçuş bilgisayarlarında (NVIDIA Jetson / Onboard RT-OBC) bağımsız çalışmak üzere eğitilmiş **PyTorch Fizik Bilgilendirmeli Sinir Ağı (PINN)**:

$$\mathcal{L} = \mathcal{L}_{veri} + \lambda_{fizik} \left\| a_{drag}^{pred} - \frac{1}{2} \rho_{pred} \left(\frac{A}{m}\right) v^2 \right\|^2$$

* **Çıkarım Hızı:** **< 150 mikrosaniye** gecikme süresi.
* **Bellek Alanı:** **< 10 KB** INT8 kuantize model boyutu.

<p align="center">
  <img src="assets/autonomous_avoidance_hud.jpg" alt="Otonom Kaçınma ve Uç Nokta Yapay Zeka HUD Arayüzü" width="100%" style="border-radius: 12px; box-shadow: 0 8px 24px rgba(56,189,248,0.3);" />
</p>

**Çok Amaçlı Pareto Frontier:** Tek tip kör ateşleme yerine üç yönlü optimum çözümler üretir:
* **Minimum Yakıt:** $\Delta v = 0.325\text{ m/s}$ mikro-fazlama ile %88 yakıt tasarrufu.
* **Dengeli Optimum:** Yakıt maliyeti, karşılaşma mesafesi ve ikincil enkaz riski arasında altın oran.
* **Maksimum Güvenlik:** Derin orbital irtifa yükseltme manevrası.

---

### 4. Walker Delta Takımyıldız & Enkaz Taraması
* 24 ila 72 uyduluk takımyıldız filolarını (Starlink, OneWeb mimarisi) yüzlerce orbital enkaz parçasına karşı $\pm 15\text{ km}$ mekânsal irtifa kapısı ile saniyeler içinde tarar.
* Filo çapında anlık risk dağılımı (Kritik, Yüksek, Nominal) üretir.

---

### 5. Yer İstasyonları & İletişim Kesintisi (Blackout)
* **Ankara TÜBİTAK UZAY**, Svalbard (Norveç), Goldstone (ABD), Madrid (İspanya) ve Canberra (Avustralya) yer istasyonlarının görüş pencerelerini hesaplar.
* $K_p \ge 7$ fırtınalarında oluşan kutup radyo kesintilerini (Polar Cap Absorption) modelleyerek yer istasyonundan bağımsız **Uç Nokta Otonomisinin** zorunluluğunu kanıtlar.

---

## 🏗️ Sistem Mimarisi

```text
┌─────────────────────────────────┐        ┌─────────────────────────────────┐
│     NASA OMNIWeb / NOAA SWPC    │        │      Space-Track TLE / CDM      │
│  (Güneş Rüzgarı, Kp, F10.7, Ap) │        │   (Yörünge Konumu & Kovaryans)  │
└────────────────┬────────────────┘        └────────────────┬────────────────┘
                 │                                          │
                 ▼                                          ▼
┌────────────────────────────────────────────────────────────────────────────┐
│               Fizik Bilgilendirmeli Yörünge & Sürüklenme Motoru            │
│  • Cowell RK4 Sayısal İntegratörü  • J2 Yerçekimi Basıklık Modeli          │
│  • Çok Modelli Termosfer: Üstel, Jacchia-77, NRLMSISE-00 Çözücüleri        │
│  • Sonlu Süreli Motor İtki ve Kütle Tüketim Entegratörü (Maneuver Splicer) │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    A-DRI Risk Motoru & Uç Nokta Optimize Edici             │
│  • 3D-2D B-Düzlemi Kovaryans İzdüşümü ve Monte Carlo Doğrulayıcısı         │
│  • Çok Amaçlı Pareto Frontier Manevra Optimize Edicisi                     │
│  • Uçuş Bilgisayarı PINN Yapay Zeka Çekirdeği (< 150 µs, INT8 Hazır)       │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│   5 Sekmeli İnteraktif Görev    │       │   NASA-STD-8719.14 Uyumluluk    │
│  Kontrol (Streamlit + FastAPI)  │       │   Otomatik Denetim & ODAR Raporu│
│  • 3D Yörünge Küresi            │       │   CCSDS OEM / OPM Ephemeris     │
│  • Walker Delta Filo Taraması   │       │   NASA Standart Breakup Modeli  │
│  • TÜBİTAK UZAY Ground Track    │       │   20 Yıllık Kessler Kaskad Sim. │
└─────────────────────────────────┘       └─────────────────────────────────┘
```

---

## 🖥️ 8 Sekmeli Görev Kontrol Simülatörü

Arayüzü başlatmak için:
```bash
streamlit run app.py
```

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ [Sekme 1: 🌍 3D Yörünge Simülatörü & Otonom Kaçınma]                         │
│  Plotly 3D Dünya küresi, uydu yörüngesi, enkaz konumu, tek tıkla Güneş        │
│  fırtınası simülasyonu ve otonom kaçınma ateşlemesi.                         │
├──────────────────────────────────────────────────────────────────────────────┤
│ [Sekme 2: 🛰️ Takımyıldız & Filo Taraması]                                    │
│  Walker Delta takımyıldız filolarını (24-72 uydu) yüzlerce enkaz parçasına    │
│  karşı anlık filtreleyip en yüksek riskli uyduları sıralar.                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ [Sekme 3: ⚖️ Pareto Çok Amaçlı Manevra Optimizasyonu]                        │
│  Yakıt Tüketimi (Δv), Güvenli Ayrılma Mesafesi ve İkincil Enkaz Riski         │
│  arasındaki interaktif Pareto Frontier saçılım grafiği.                      │
├──────────────────────────────────────────────────────────────────────────────┤
│ [Sekme 4: 📡 Yer İstasyonları & İletişim Kesintisi (Blackout)]                │
│  2D Dünya haritasında Ankara TÜBİTAK UZAY ve dünya yer istasyonları izleri;   │
│  Güneş fırtınasında telsiz bağlantısının koptuğu pencereler.                 │
├──────────────────────────────────────────────────────────────────────────────┤
│ [Sekme 5: 🔬 Atmosfer Modelleri Laboratuvarı]                                 │
│  Üstel Model, Jacchia-77 ve NRLMSISE-00 modellerini LEO irtifaları boyunca     │
│  logaritmik eğrilerle yan yana karşılaştırır.                                │
├──────────────────────────────────────────────────────────────────────────────┤
│ [Sekme 6: ☢️ Radyasyon & TMR Hata Toleransı Simülatörü]                      │
│  Güney Atlantik Anomalisi (SAA) ve Güneş Parçacık Olayı (SPE) altında         │
│  ağırlıklara SEU bit-flip enjekte ederek TMR çoğunluk oylamasını test eder. │
├──────────────────────────────────────────────────────────────────────────────┤
│ [Sekme 7: 🎯 Deorbit & Point Nemo Güvenli Giriş Simülatörü]                  │
│  Point Nemo hedefli retro-ateşleme, hipersonik atmosferik giriş ve aerotermal │
│  yanma modeli ile NASA Req 4.7 can güvenliği değerlendirmesi.                │
├──────────────────────────────────────────────────────────────────────────────┤
│ [Sekme 8: 🌐 Canlı CelesTrak TLE Sorgulama Motoru]                           │
│  NORAD ID ile CelesTrak ve Space-Track üzerinden canlı TLE verisi çeker.      │
├──────────────────────────────────────────────────────────────────────────────┤
│ [Sekme 9: 🤖 9-Ajanlı Otonom Sürü Zekası (Autonomous Swarm)]                 │
│  9 uçuş ajanının eşzamanlı kararları, BFT oylama defteri, canlı kara tahta    │
│  ve SHA-256 kriptografik imzalı otonom manevra yetkilendirmesi.               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 💻 Profesyonel Aegis CLI Rehberi

Komut satırı aracı, terminal ortamında tam teşekküllü görev operasyonları sunar:

| Komut | Açıklama | Örnek Kullanım |
| :--- | :--- | :--- |
| `status` | Sistem durumu ve uzay havası telemetrisi | `python -m aegis_leo.cli status` |
| `eval` | Birleşik A-DRI risk puanını hesaplar | `python -m aegis_leo.cli eval --sat STARLINK_LEO --miss 0.45 --kp 8.0` |
| `burn` | Otonom kaçınma ateşlemesini planlar | `python -m aegis_leo.cli burn --sat STARLINK_LEO --miss 0.45` |
| `swarm` | **9-Ajanlı Otonom Sürü Zekası Konsensüs Döngüsü** | `python -m aegis_leo.cli swarm --sat STARLINK_LEO --kp 7.8 --miss 0.082` |
| `consensus` | **Bizans Hata Toleranslı (BFT) Sürü Manevra Konsensüsü** | `python -m aegis_leo.cli consensus --sat-a SAT-A1 --sat-b SAT-A2` |
| `lifetime` | **FCC 5-Yıl & NASA 25-Yıl Yörünge Ömrü Tahmini** | `python -m aegis_leo.cli lifetime --alt 450 --mass 260` |
| `tether` | **Elektrodinamik Tether (EDT) İtkisiz Deorbit Analizi** | `python -m aegis_leo.cli tether --mass 250 --alt 750` |
| `laser` | **Darbeli Lazer Ablasyon (LODR) Enkaz İtme Analizi** | `python -m aegis_leo.cli laser --mass 1.2 --dia 12 --power 10` |
| `monte-carlo` | Monte Carlo kovaryans dağılımı simülasyonu | `python -m aegis_leo.cli monte-carlo --miss 0.45 --trials 1000` |
| `breakup` | NASA standart parçalanma kaskad modeli | `python -m aegis_leo.cli breakup --target-mass 260.0 --proj-mass 5.0` |
| `deorbit` | Point Nemo hedefli kontrollü deorbit analizi | `python -m aegis_leo.cli deorbit --alt 550.0 --mass 260.0` |
| `rad-test` | SAA/SPE radyasyon & TMR hata toleransı testi | `python -m aegis_leo.cli rad-test --alt 550.0 --flips 4` |
| `celestrak` | CelesTrak'ten canlı TLE yörünge elemanı çeker | `python -m aegis_leo.cli celestrak --norad 25544` |


---

## 🌐 REST API Dokümantasyonu (FastAPI)

FastAPI sunucusunu başlatmak için:
```bash
uvicorn backend.main:app --reload --port 8000
```
* **Etkileşimli Swagger Dokümantasyonu:** `http://localhost:8000/docs`
* **Three.js Web HUD Kontrol Paneli:** `http://localhost:8000/dashboard`

### Temel API Uç Noktaları:
- `GET  /api/health`: Sistem alt bileşen sağlık kontrolü.
- `GET  /api/space-weather/live`: Canlı NOAA SWPC güneş akısı ve Kp indeksi.
- `POST /api/space-weather/simulate-storm`: G1-G5 fırtına senaryosu tetikleme.
- `POST /api/risk/evaluate-conjunction`: Çarpışma riski ve sentetik CDM üretimi.
- `POST /api/edge-ai/optimize-maneuver`: Uç nokta otonom kaçınma planlaması.
- `POST /api/fleet/screen`: Takımyıldız toplu enkaz taraması.
- `POST /api/maneuvers/pareto-frontier`: Çok amaçlı Pareto aday ateşleme listesi.
- `GET  /api/atmosphere/compare-models`: Çok modelli termosfer yoğunluk analizi.
- `GET  /api/ground-track/calculate`: 2D yer izdüşümü ve blackout analizi.
- `POST /api/compliance/audit`: NASA-STD-8719.14 denetimi ve ODAR raporu.
- `GET  /api/swarm/status`: 9 otonom uçuş ajanının anlık sağlık ve telemetri durumu.
- `POST /api/swarm/consensus-cycle`: Uçtan uca 9-ajanlı otonom konsensüs ve ateşleme döngüsü.
- `POST /api/lifetime/decay-prediction`: 11 yıllık Güneş döngüsü altında yörünge ömrü & FCC 5-Yıl denetimi.
- `GET  /api/adr/electrodynamic-tether`: Elektrodinamik Tether (EDT) Lorentziyen itkisiz deorbit simülasyonu.
- `POST /api/adr/laser-ablation`: Uzay/yer konuşlu darbeli lazer ablasyonu (LODR) ile enkaz perije düşürme.
- `POST /api/swarm/bft-consensus`: Bizans hata toleranslı (BFT) ve Nash pazarlıklı otonom kaçınma tahsisi.

---


## 📋 NASA-STD-8719.14 Uyumluluk Matrisi

AEGIS-LEO, **NASA-STD-8719.14** (*Process for Limiting Orbital Debris*) gereksinimlerine göre operasyonel denetim yapar:

| Standart Maddesi | Başlık | Eşik Kriteri | AEGIS-LEO Değerlendirmesi |
| :--- | :--- | :---: | :---: |
| **Req 4.3** | Operasyonel Enkaz Salımı | Sıfır planlı enkaz | ✅ **UYUMLU** |
| **Req 4.4** | Kaza Sonucu Patlama Riski | Batarya & yakıt pasivasyonu | ✅ **UYUMLU** (Telemetri onaylı) |
| **Req 4.5.1** | Görev Sonrası Yörünge Ömrü | $\le 25$ Yıl (NASA kuralı) | ✅ **UYUMLU** (548 km'de ~5.2 yıl) |
| **Req 4.5.2** | FCC Hızlı Temizleme Kuralı | $\le 5$ Yıl (Modern kural) | ⚠️ **Kontrollü deorbit gerekir** |
| **Req 4.6.1** | Çarpışma Olasılığı Limiti | $P_c < 10^{-4}$ ($1:10.000$) | ✅ **UYUMLU** (Otonom manevra ile) |
| **Req 4.7** | Atmosfere Giriş Can Güvenliği | Risk $< 1:10.000$ | ✅ **UYUMLU** |

---

## 📊 Vaka Analizi: NOAA G4/G5 Fırtınası Stres Testi

548 km irtifada Güneş Eşzamanlı (SSO) bir LEO uydusu ($A/m = 0.0085 - 0.02\text{ m}^2/\text{kg}$) için G4 şiddetli jeomanyetik fırtına simülasyonu:

| Metrik | AEGIS-LEO Olmadan (Statik CDM) | AEGIS-LEO İle (Otonom Kalkan) | Kazanım / Fark |
| :--- | :---: | :---: | :---: |
| **Termosferik Yoğunluk ($\rho$)** | Taban Değer ($2.45 \times 10^{-13}\text{ kg/m}^3$) | Dinamik Fırtına ($1.44 \times 10^{-12}\text{ kg/m}^3$) | Gerçek Zamanlı Fizik Farkındalığı |
| **36 Saatlik İrtifa Kaybı** | $10.52\text{ km}$ (tahmin edilen) | $10.67\text{ km}$ (gerçekleşen) | $+151\text{ m}$ ek sürtünme kaybı |
| **Boylamsal Yörünge Kayması** | $0.0\text{ km}$ (varsayılan) | **$+16.72\text{ km}$** (gerçekleşen) | Sapma Tespit Edildi & Düzeltildi |
| **A-DRI Risk Seviyesi** | Kör / Statik ($P_c$ geçersiz) | **0.449 $\to$ 0.842 (KRİTİK)** | Anında Çarpışma Alarmı |
| **Kaçınma Manevrası Türü** | Geç Reaktif Acil Durum Kaçınması | **Otonom Erken Fazlama Mikro Ateşlemesi** | Proaktif Yörünge Düzeltme |
| **Ateşleme Maliyeti ($\Delta v$)** | $2.80\text{ m/s}$ | **$0.325 - 0.850\text{ m/s}$** | **%88'e Varan Yakıt Tasarrufu** |
| **Manevra Sonrası Risk (A-DRI)** | Yüksek Yakıt Tüketim Cezası | **0.146 (NOMİNAL)** | Görev Ömrü Korundu & Uyum Sağlandı |

---

## 🧪 Doğrulama & Test Paketi

Depo, 18 bağımsız otonom geliştirici ajan ve ana mimari tarafından yazılan **352 adet** birim ve entegrasyon testinden oluşan uzay ajansı seviyesinde kapsamlı bir test paketi içerir:
```bash
pytest -v
```

```text
tests/test_advanced_features.py ............... [PASSED] (5 test: CelesTrak, TMR & CLI)
tests/test_agents_swarm.py .................... [PASSED] (11 test: 9 Ajan & Swarm Telemetri)
tests/test_ai_edge.py ......................... [PASSED] (4 test: PINN & INT8 Kuantizasyon)
tests/test_api.py ............................. [PASSED] (13 test: FastAPI & BFT Swarm API)
tests/test_ccsds_cdm_xml.py ................... [PASSED] (20 test: CCSDS 508.0-B-1 CDM)
tests/test_compliance.py ...................... [PASSED] (2 test: NASA-STD-8719)
tests/test_cw_proximity.py .................... [PASSED] (23 test: HCW & Yamanaka-Ankersen)
tests/test_debris_capture.py .................. [PASSED] (25 test: 6-DOF Kol & Detumbling)
tests/test_drag_cross_section.py .............. [PASSED] (11 test: Açısal Kesit & Tutum)
tests/test_drag_makeup.py ..................... [PASSED] (19 test: VLEO L1 Adaptif Kontrol)
tests/test_electrodynamic_tether.py ........... [PASSED] (5 test: EDT Lorentziyen İtkisiz Deorbit)
tests/test_federated_swarm.py ................. [PASSED] (25 test: ISL FedAvg & FedProx)
tests/test_foster_bplane.py ................... [PASSED] (11 test: Foster/Chan B-Düzlemi)
tests/test_global_standards.py ................ [PASSED] (6 test: CCSDS OEM/OPM)
tests/test_ground_track_and_models.py ......... [PASSED] (3 test: Yer İzdüşümü & Blackout)
tests/test_hardware_telemetry.py .............. [PASSED] (7 test: OBC Donanım & SEU Watchdog)
tests/test_iod_gauss.py ....................... [PASSED] (20 test: Gauss/Laplace/Gibbs IOD)
tests/test_laser_ablation.py .................. [PASSED] (5 test: LODR Darbeli Lazer Ablasyon)
tests/test_laser_radar.py ..................... [PASSED] (23 test: LIDAR & Radar Swerling)
tests/test_low_thrust.py ...................... [PASSED] (9 test: İyon İtki & Edelbaum)
tests/test_mpc_avoidance.py ................... [PASSED] (9 test: MPC & SCP Optimizasyonu)
tests/test_odar_generator.py .................. [PASSED] (10 test: NASA ODAR Raporlama)
tests/test_orbit_lifetime.py .................. [PASSED] (6 test: King-Hele & FCC 5-Yıl Kuralı)
tests/test_pareto.py .......................... [PASSED] (1 test: Çok Amaçlı Frontier)
tests/test_perturbations.py ................... [PASSED] (12 test: Güneş Işınımı & 3. Cisim)
tests/test_physics.py ......................... [PASSED] (6 test: RK4 & Termosfer)
tests/test_reentry_footprint.py ............... [PASSED] (10 test: Point Nemo Düşüş Elipsi)
tests/test_risk.py ............................ [PASSED] (3 test: A-DRI & ORDEM)
tests/test_screening.py ....................... [PASSED] (3 test: Walker Delta Tarama)
tests/test_sensor_fusion.py ................... [PASSED] (16 test: EKF & UKF Navigasyon)
tests/test_stationkeeping.py .................. [PASSED] (12 test: Macar Algoritması & Yuva)
tests/test_swarm_consensus.py ................. [PASSED] (3 test: BFT Quorum & Nash Pazarlığı)
tests/test_swarm_telemetry_bus.py ............. [PASSED] (14 test: Hızlı Yayın Hattı)
============================ 352 passed in 27.48s ============================
```




---

## 🐳 Docker Konteynerizasyonu

Tek komutla hem API backend'i hem Streamlit arayüzünü ayağa kaldırın:
```bash
docker compose up --build
```
* **Streamlit Görselleştirici:** `http://localhost:8501`
* **FastAPI Backend:** `http://localhost:8000`

---

## 👥 AYAZ Takımı & Lisans

Bu proje **NASA Space Apps Challenge 2026** Ankara etabı için tutkuyla geliştirilmiştir.

* **Bahattin Yunus Çetin** ([@bahattinyunus](https://github.com/bahattinyunus)) - *Lider Sistem, Yörünge Mekaniği ve Uç Nokta Yapay Zeka Mimarı*

### 📜 Lisans
Bu proje **MIT Lisansı** altında lisanslanmıştır. Ayrıntılar için [LICENSE](LICENSE) dosyasına bakabilirsiniz.
