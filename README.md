# 🛡️ AEGIS-LEO: Otonom Uç Nokta Yapay Zekası ve Fizik Bilgilendirmeli Yörünge Güvenlik Kalkanı

[![NASA Space Apps 2026](https://img.shields.io/badge/NASA%20Space%20Apps-2026%20Ankara-0B3D91.svg)](https://www.spaceappschallenge.org/)
[![Takım](https://img.shields.io/badge/Takım-AYAZ-orange.svg)]()
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?logo=pytorch&logoColor=white)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg?logo=streamlit&logoColor=white)]()
[![Lisans: MIT](https://img.shields.io/badge/Lisans-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"Uzay Havası Fiziği, Yörünge Mekaniği ve Otonom Uç Nokta Yapay Zekasını Birleştirerek Alçak Dünya Yörüngesini (LEO) Koruyoruz."**

**AEGIS-LEO**, **NASA Space Apps Challenge 2026** için **AYAZ Takımı** tarafından geliştirilen uçtan uca yörünge risk istihbaratı ve otonom çarpışma önleme simülasyon platformudur. Mevcut uzay durumsal farkındalık (SSA) sistemlerindeki en kritik kör noktayı çözer: **Geomanyetik fırtınaların tetiklediği ani termosferik genleşme**, öngörülemeyen atmosferik sürüklenmeye yol açar, statik Yakınlaşma Veri Mesajlarını (CDM) geçersiz kılar ve Kessler Sendromu riskini katlar.

---

## ⚡ Temel İnovasyonlar

### 1. Fizik Bilgilendirmeli Termosferik Sürüklenme Modeli
**NASA OMNIWeb** ve **NOAA SWPC** güneş rüzgarı ile jeomanyetik parametrelerini kullanan AEGIS-LEO, güneş fırtınaları ($K_p \ge 5$) sırasında üst atmosfer yoğunluğundaki ($\rho$) ani dalgalanmaları ve sürüklenme ivmesi sapmalarını anlık olarak modeller:

$$a_{drag} = -\frac{1}{2} \rho(h, K_p, F_{10.7}) \left(\frac{C_d A}{m}\right) v_{rel}^2 \hat{v}_{rel}$$

* Gezegensel jeomanyetik fırtına indeksi ($K_p$) ve 10.7 cm güneş radyo akısı ($F_{10.7}$) ile dinamik ölçekleme.
* G4/G5 fırtınalarında termosfer yoğunluğu **3.5x ila 7.6x kat** genleşir; bu durum 36 saat içinde **14 ila 26 km** arasında modellenememiş boylamsal yörünge kaymalarına (along-track drift) sebep olur.

### 2. Uyarlanabilir Enkaz ve Sürüklenme Risk İndeksi (A-DRI)
Takip edilen her LEO uydusu ve uzay aracı için gerçek zamanlı hesaplanan birleşik risk metriği ($0.0 \to 1.0$):

$$\text{A-DRI} = w_1 C_{prob} + w_2 W_{drag} + w_3 D_{ord} + w_4 F_{\Delta v}$$

* **$C_{prob}$ ($w_1 = 0.40$):** 3D'den 2D karşılaşma B-düzlemine kovaryans izdüşümü ile hesaplanan taban çarpışma olasılığı.
* **$W_{drag}$ ($w_2 = 0.25$):** Alan/kütle oranı ($A/m$) ve güneş fırtınası genleşmesinden kaynaklanan yörünge konumu belirsizliği.
* **$D_{ord}$ ($w_3 = 0.20$):** **NASA ORDEM 3.1** ve ESA MASTER modelleriyle kalibre edilmiş irtifa ve eğim uzay enkazı yoğunluk faktörü.
* **$F_{\Delta v}$ ($w_4 = 0.15$):** Kalan yakıt bütçesi ve geç kaçınma manevrası maliyetini cezalandıran yakıt fizibilite katsayısı.

### 3. Uç Nokta Yapay Zekası Otonom Kaçınma Motoru
Radyasyon dayanımlı uçuş bilgisayarlarında (NVIDIA Jetson / Onboard RT-OBC) bağımsız çalışmak üzere tasarlanmıştır:
* **Fizik Bilgilendirmeli Sinir Ağı (PINN):** Mekanik enerji korunumu ve sürüklenme kaybı kısıtlarıyla, termosferik sapmayı **< 150 mikrosaniyede** ve **< 10 KB INT8 bellek** alanında hesaplar.
* **Otonom Fazlama ve İrtifa Motoru:** **Vis-Viva denklemi** ve **Tsiolkovsky roket denklemini** kullanarak erken proaktif mikro ateşlemeler ($\Delta v \approx 0.3 - 0.85\text{ m/s}$) planlar; yer istasyonundan komut beklenen geç reaktif manevralara kıyasla **%88'e varan yakıt tasarrufu** sağlar.

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
│          (Cowell RK4 + J2 Basıklığı + NRLMSISE Termosfer Modeli)           │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                    A-DRI Risk Motoru & Uç Nokta Optimize Edici             │
│            (B-Düzlemi Kovaryans İzdüşümü + Vis-Viva Fazlaması)             │
│             [Radyasyon Dayanımlı Uçuş PINN Çekirdeği - INT8 Hazır]         │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│   3D İnteraktif Görev Kontrol   │       │   NASA-STD-8719.14 Uyumluluk    │
│  (Streamlit 3D Küre + FastAPI)  │       │   Otomatik Denetim & ODAR Raporu│
└─────────────────────────────────┘       └─────────────────────────────────┘
```

---

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler & Kurulum

Repoyu klonlayın ve bağımlılıkları yükleyin:
```bash
git clone https://github.com/bahattinyunus/AEGIS-LEO.git
cd AEGIS-LEO

# Gerekli kütüphaneleri yükleyin
pip install -r requirements.txt
```

### 2. 3D İnteraktif Simülatörü Başlatın (Streamlit)
Görsel 3D Dünya küresi, uydu seçimi ve fırtına simülasyonunu başlatmak için:
```bash
streamlit run app.py
```
* **Etkileşimli 3D Dünya:** Uydu yörüngesini, enkaz bulutunu ve manevra sonrası güvenli rotayı çizer.
* **Tek Tıkla Güneş Fırtınası:** Termosferik yoğunluğun fırlamasını ve A-DRI skorunun kırmızıya (CRITICAL) dönüşünü izleyin.
* **Tek Tıkla Otonom Kaçınma:** Optimum $\Delta v$ mikro ateşlemesini hesaplar, tasarruf edilen yakıtı gösterir ve riski yeşile (NOMINAL) çeker.

### 3. FastAPI Görev Kontrol & 3D Web HUD
REST API ve Three.js arayüzünü çalıştırmak için:
```bash
uvicorn backend.main:app --reload --port 8000
```
* **Web Arayüzü (HUD):** [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
* **Otomatik Swagger API Dokümantasyonu:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Terminal Simülasyonu & Doğrulama
Grafik arayüz olmadan komut satırından uçtan uca simülasyonu çalıştırmak için:
```bash
python demo_script.py
```

### 5. Test Paketini Çalıştırın
Tüm fizik, yörünge mekaniği, yapay zeka ve API uçlarını test etmek için:
```bash
pytest -v
```

---

## 📊 Vaka Analizi: NOAA G4/G5 Güneş Fırtınası Stres Testi

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

## 📋 NASA-STD-8719.14 Uyumluluk Motoru

AEGIS-LEO, **NASA-STD-8719.14** (*Yörünge Enkazını Sınırlama Süreci*) teknik standardı için otomatik denetim ve raporlama sağlar:
* **Madde 4.3:** Normal operasyonlar sırasında serbest bırakılan enkaz sınırları.
* **Madde 4.4:** Kaza sonucu patlama riski ve pasivasyon (yakıt/batarya deşarjı) kabiliyeti.
* **Madde 4.5:** Görev sonrası yörünge ömrü ($\le 25$ yıl NASA kuralı, $\le 5$ yıl güncel FCC kuralı).
* **Madde 4.6:** Operasyonel çarpışma olasılığı limiti ($P_c < 10^{-4}$ eşiği).
* **Otomatik ODAR Rapor Üreticisi:** Standartlara tam uyumlu Yörünge Enkazı Değerlendirme Raporu (.md / JSON).

---

## 🇹🇷 30 Saniyelik Hızlı Demo & Sunum Adımları

1. **Simülatörü Başlatın:** Terminalde `streamlit run app.py` komutunu çalıştırın.
2. **Hedef Uyduyu Seçin:** Kenar çubuğundan Starlink-1007 veya Sentinel-1A'yı seçin.
3. **Fırtınayı Başlatın:** **"⚡ Güneş Fırtınası Simüle Et"** butonuna basın; termosferik yoğunluğun fırladığını, $+16\text{ km}$ yörünge sapmasını ve A-DRI skorunun kırmızıya (CRITICAL) döndüğünü gösterin.
4. **Otonom Kaçınmayı Çalıştırın:** **"🚀 Otonom Δv Kaçınma Hesapla"** butonuna tıklayarak uç nokta yapay zekasının mikrosaniyeler içinde mikro ateşlemeyi ($\Delta v = 0.325\text{ m/s}$) hesaplayıp yörüngeyi düzelttiğini ve %88 yakıt tasarrufu sağladığını sergileyin.
5. **Denetim Raporunu İndirin:** **"📋 NASA-STD-8719.14 Denetle"** butonundan resmi ODAR raporunu indirin.

---

## 👥 AYAZ Takımı

**NASA Space Apps Challenge 2026** Ankara etabı için tutkuyla geliştirilmiştir.

* **Bahattin Yunus Çetin** ([@bahattinyunus](https://github.com/bahattinyunus)) - *Lider Sistem, Yörünge Mekaniği ve Uç Nokta Yapay Zeka Mimarı*

---

## 📜 Lisans

Bu proje **MIT Lisansı** ile lisanslanmıştır - ayrıntılar için [LICENSE](LICENSE) dosyasına bakabilirsiniz.
