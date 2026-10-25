# TrackLab: Görev Listesi

Repoda `docs/TASKS.md` olarak durur. Bir görevi bitirince `[ ]` işaretini `[x]` yap
(VS Code'da önizleme: Cmd+Shift+V). Kısa notlar için `docs/PROGRESS.md` kullanılır.
Ayrıntılar `docs/PROJECT_PLAN.pdf` (devir belgesi) ve `docs/PROJECT_PLAN_ADDENDUM.md` (uygulama güncellemeleri) içindedir.

Kurallar:
- Her gün sonunda commit at.
- Hold-out sekanslarına (02, 10, 13) yalnızca **Gün 17**'de bakılır.
- Her aşamanın sonunda "Bitti ölçütü" sağlanmadan sonrakine geçme.

---

## Hafta 1: Uçtan uca iskelet (v0.1)

### Gün 1: Kurulum
- [x] `~/Desktop/tracklab` klasörü ve `git init`
- [x] Python 3.12 kurulu (`python3.12 --version`)
- [x] `.venv` 3.12 ile yeniden oluşturuldu
- [x] `pip install ultralytics ipykernel`; MPS kontrolü `True`
- [x] Proje kökünde `.gitignore` var (içinde `.venv/` dahil)
- [x] Klasör iskeleti: `src/tracklab`, `scripts`, `data`, `outputs`, `tests`, `notebooks`
- [x] `notebooks/00_sanity_check.ipynb` 3.12 kernel ile çalışıyor
- [x] `data/sample.mp4` (Pexels'ten 10-20 sn yaya videosu)
- [x] `yolo track ...` komutu çalıştı, ID'ler tutarlı mı kontrol edildi
- [x] İlk commit: `Initial project setup`

**Bitti ölçütü:** CLI ile ID'li takip videosu üretildi, ilk commit atıldı.

### Gün 2: Python pipeline'ı (Ultralytics API ile ID'li MP4 + JSON)
- [x] `LICENSE` kararı (öneri: AGPL-3.0)
- [x] `src/tracklab/schemas.py`
- [x] `src/tracklab/config.py` ve `configs/pipeline.yaml`, `detector.yaml`, `bytetrack.yaml`
- [x] `src/tracklab/io/video.py` (kare okuma/yazma)
- [x] `src/tracklab/io/results.py` (JSON)
- [x] `src/tracklab/viz/draw.py`
- [x] `src/tracklab/tracking/yolo_bytetrack.py`
- [x] `src/tracklab/pipeline.py`
- [x] `scripts/run_video.py`
- [x] `tests/unit/test_results_io.py`
- [x] Commit

**Bitti ölçütü:** `run_video.py` örnek videoda ID'li MP4 + JSON üretiyor.

### Gün 3: FastAPI servisi
- [x] `api/settings.py`, `api/models.py`
- [x] `api/service.py` (JobManager)
- [x] `api/routes.py`, `api/main.py`
- [x] `tests/integration/test_api.py` (sahte tracker ile)
- [x] Commit

**Bitti ölçütü:** `POST /jobs` sonrası iş `done` oluyor; test yeşil.

### Gün 4: Yükleme sayfası ve video uyumu
- [x] `api/static/index.html`
- [x] H.264 yeniden kodlama (`io/video.py`, ffmpeg)
- [x] Boyut/süre sınırları ve hata durumları
- [x] Commit

**Bitti ölçütü:** Tarayıcıda video yüklenip sonuç oynatılabiliyor.

### Gün 5: Docker ve ilk sürüm
- [x] `docker/Dockerfile`, `docker/docker-compose.yml`
- [x] `README.md` ilk sürüm
- [x] Taslak `ci.yml`
- [x] `v0.1.0` release etiketi

**Bitti ölçütü:** `docker compose up` ile aynı akış çalışıyor.

---

## Hafta 2: Ölçüm ve ön işleme

### Gün 6: MOT17 verisi
- [x] MOT17 indirildi (`data/raw/MOT17`), lisans notu `DATA_LICENSES.md`'ye eklendi
- [x] `src/tracklab/datasets/mot17.py`, `src/tracklab/datasets/splits.py`
- [x] `scripts/prepare_mot17.py`
- [x] `tests/unit/test_splits.py`, `tests/unit/test_mot17_reader.py`

**Bitti ölçütü:** Split manifesti hazır (tuning: 04, 05, 09, 11 / hold-out: 02, 10, 13).

### Gün 7: TrackEval
- [x] `third_party/TrackEval` kuruldu
- [x] `evaluation/trackeval_bridge.py`
- [x] **E0:** GT'yi tahmin olarak ver, HOTA ≈ 100

**Bitti ölçütü:** Ölçüm hattı doğrulandı.

### Gün 8: Baseline
- [x] `scripts/evaluate_mot.py`, `configs/eval.yaml`
- [x] `evaluation/mot17_detector_comparison.py` ve birim testleri
- [x] **E1:** COCO YOLO11n + ByteTrack ve MOT17 hazır FRCNN tespitleri + ByteTrack; tuning sekansları 04, 05, 09 ve 11
- [x] TrackEval raporu: HOTA, DetA, AssA, LocA, MOTA, IDSW, IDF1, IDP ve IDR
- [x] `conf` ve `imgsz` için 12 konfigürasyonluk tuning taraması
- [x] Dengeli başlangıç yapılandırması seçildi: `conf=0.10`, `imgsz=960`

**Bitti ölçütü:** Sekans bazlı baseline tablosu hazır; `conf`/`imgsz` seçimi ölçümle gerekçelendirildi. En yüksek HOTA veren `0.15/1280` ayarı MOT17-05'te belirgin gerileme ve daha fazla ID switch oluşturduğu için genel baseline olarak seçilmedi.

### Gün 9: Kalite ve ön işleme modülleri
- [x] `quality/metrics.py`, `quality/report.py`; yapılandırılabilir kare örneklemesi ve özet rapor
- [x] `preprocessing/` base, gamma, CLAHE, denoise ve factory modülleri
- [x] Birim ve pipeline testleri; geometri/veri tipi değişmezleri ve varsayılan `mode: none` doğrulandı
- [x] Ön işleme yalnızca tracker girişine uygulanıyor; kalite ölçümü ve görselleştirme orijinal kareyi kullanıyor

**Bitti ölçütü:** Ön işleyiciler geometriyi değiştirmiyor, testler yeşil. Tam proje doğrulaması: 109 test geçti; Ruff ve biçim kontrolleri başarılı.

### Gün 10: Sentetik bozulmalar ve ablation
- [x] `evaluation/degradations.py`, `ablation.py`, `scripts/run_ablation.py`
- [x] **E2:** none / gamma / CLAHE / (denoise) gerçek videolarda
- [x] **E2b:** aynı deney sentetik bozulmalar altında

<!-- DAY10_RESULTS_AND_COLORSPACE -->
- [x] Gerçek ve sentetik sonuçlar ayrı tablolarda `reports/day10_ablation_results.md` içine kaydedildi.
- [x] Ek araştırma: birleşik işlem sıraları, LAB `L` / HSV `V` / YCrCb `Y` renk-uzayı varyantları ve FPS ölçümü.
- [x] Gerçek zamanlı üç koşul optimize edilmiş ön işleme koduyla ayrı çıktı dizininde tekrar ölçüldü.
**Bitti ölçütü:** Gerçek ve sentetik sonuçlar ayrı tablolarda `reports/day10_ablation_results.md` içinde; testler ve lint kontrolleri başarılı.
---

## Hafta 3: Fine-tune ve analitik

### Gün 11: YOLO veri seti
- [x] `src/tracklab/datasets/to_yolo.py`, `tests/unit/test_to_yolo.py`
- [x] `data/processed/MOT17_yolo_day11/dataset.yaml` üretildi; hold-out sızıntısı yok

**Bitti ölçütü:** YOLO veri kümesi sequence-level train/validation ayrımıyla üretildi; tüm görüntü ve etiket dosyaları denetlendi, hold-out sekansları dışarıda bırakıldı.

### Gün 12-13: Fine-tune
- [x] `training/finetune.py`, `configs/finetune.yaml`, `configs/finetune_mps_bench.yaml`, `scripts/run_finetune_pilot.sh`
- [x] Nano model fine-tune pilotları (`imgsz=640`, batch 4 ve batch 8 benchmark)
- [x] `best.pt` ve eğitim metrikleri kaydedildi

### Gün 14: Fine-tuned vs baseline
- [x] **E3:** tuning iç doğrulamasında karşılaştırma

**Bitti ölçütü:** Önce/sonra tablosu.

### Gün 15: Analitik
- [x] `analytics/geometry.py`, `line_counter.py`, `trajectories.py`
- [x] `viz/draw.py` içinde sayaç ve iz çizimi
- [x] Birim testleri

**Bitti ölçütü:** Demo videosunda sayaç ve rotalar görünüyor.

---

## Hafta 4: Teslim

### Gün 16: Tracker parametreleri
- [x] **E4:** `track_buffer`, `match_thresh`, eşikler (küçük tarama, tuning'de)

  Sonuç: `reports/day16_tracker_ablation.md`. Dengeli sekans kriteri nedeniyle
  mevcut baseline tracker ayarları korundu; `match_thresh=0.9` yalnızca
  birleşik metriklere göre daha iyi görünse de MOT17-05'te geriledi.

### Gün 17: Final değerlendirme
- [x] Hold-out üzerinde final değerlendirme (bir kez)
- [x] **E5:** elle sayım vs sistem sayımı
- [x] **E6:** hız ve aşama bazlı gecikme
- [x] Hata analizi (`reports/error_analysis.md`)

### Gün 18: Test ve CI
- [x] Eksik testler tamamlandı
- [x] CI yeşil

  Lint, format, pytest and Docker image/health smoke checks were recorded as
  successful in the release validation.

### Gün 19: Dokümantasyon
- [x] README (mimari diyagram, demo GIF, sonuç tablosu, kurulum)
- [x] `MODEL_CARD.md`, `DATA_LICENSES.md`

### Gün 20: Teslim
- [x] 60-90 sn demo videosu
- [x] `v1.0` etiketi
- [x] Canlı demo kararı

---

## Proje bitti sayılma ölçütü
- [x] Docker image ve `/health` smoke testi CI'da başarılı; Compose yapılandırması doğrulandı
- [x] Hold-out sonuçları: baseline, fine-tuned ve ön işleme varyantları sekans bazında tabloda
- [x] Testler ve CI yeşil
- [x] README, MODEL_CARD, DATA_LICENSES, LICENSE mevcut
- [x] Demo videosu hazır


## v0.1.1: Sağlamlaştırma ve dağıtım

- [x] API yeniden başladığında yarım kalan queued/running işler failed olarak işaretleniyor.
- [x] Video pipeline hatalarında geçici ham video dosyası temizleniyor.
- [x] Yüklenen kaynak video, iş tamamlandığında veya başarısız olduğunda siliniyor; reddedilen yüklemeler de temizleniyor.
- [x] Varsayılan tracker/model, video çalıştırması başına yalnızca bir kez oluşturuluyor.
- [x] API çıktı dosyalarının izin verilen çıktı klasörünün dışına erişmesi engelleniyor.
- [x] Video doğrulaması asenkron API event loop'unu bloke etmeyecek şekilde çalıştırılıyor.
- [x] API hata mesajlarında dahili dosya yolları maskeleniyor.
- [x] SQLite bağlantı ayarları ve isteğe bağlı PostgreSQL bağımlılığı düzenlendi.
- [x] JSON serileştirme ve görüntü sınırındaki kutu çizimi için testler eklendi.
- [x] Kaynak video saklama davranışı README'de belgelendi.
- [x] Initial validation covered the test suite, Ruff lint/format and diff checks.
- [x] `tracklab:0.1.1` Docker imajı oluşturuldu ve ayrı portta denendi.
- [x] Gerçek video testi: 505 kare, 4.890 tespit, 145 benzersiz takip ID'si; JSON ve MP4 çıktıları doğrulandı.
- [x] Ana Docker Compose servisi `tracklab:0.1.1` sürümüne geçirildi ve `/health` kontrolü doğrulandı.
- [x] Mevcut veritabanındaki 7 iş kaydının korunduğu doğrulandı.
- [x] `v0.1.1` etiketi oluşturuldu; aktif Git commit/etiket geçmişinden kurumsal e-posta adresi kaldırıldı.

**Hafta 1 kapanış durumu:** Yerel geliştirme, test, Docker dağıtımı ve sürümleme tamamlandı. GitHub Actions, GitHub üzerindeki master commit'inde çalıştırıldı; lint, format ve test adımlarının tamamı başarılı.

### Gün 21: Çözünürlükten bağımsız analytics ve Docker denetimi

- [x] Çizgi koordinatları için geriye dönük uyumlu `pixels` / `normalized` alanı
- [x] Normalize çizgilerin kaynak video boyutuna göre tek sefer çözülmesi
- [x] Aynı çözülmüş geometrinin counter ve renderer'a geçirilmesi
- [x] Sınır, geçersiz koordinat, yönlü geçiş ve farklı çözünürlük regresyon testleri
- [x] Fine-tuning/MOT17 değerlendirme yolundan analytics bağımlılığı bulunmadı
- [x] Gerçek API Docker akışı yeniden doğrulandı: H.264 output, JSON download,
  health endpoint and resolution-independent line placement all passed.
- [x] README ve denetim raporu doğrulanmış sınırlamalarla güncellendi

### Gün 22: Zaman tabanlı trajectory trail ve demo varlıkları

- [x] `trail_duration_seconds` config alanı eklendi; varsayılan 3,0 saniye.
- [x] Trail görünürlüğü kaynak video frame index/FPS zaman damgasına göre
  sınırlandırıldı; kaybolan track'lerin izleri de süre sonunda siliniyor.
- [x] JSON trajectory summary tam geçmişi koruyor; smoothing davranışı testlerle
  korunuyor.
- [x] Licensed Pexels daytime and rainy-road traffic sources were processed with
  the default COCO YOLO11n + ByteTrack configuration.
- [x] Daytime traffic produced one verified directional crossing on the default
  normalized center line; rainy-road traffic produced one verified crossing on
  the reproducible `configs/pipeline_demo_rainy.yaml` analytics line, without
  changing detector or tracker settings.
- [x] Published compact H.264/GIF derivatives under `docs/assets/demos/` for
  pedestrian, daytime traffic and rainy-road traffic.
- [x] Output JSON/video pairs, H.264 encoding, source dimensions/FPS, visible
  IDs, yellow lines, counter changes and 3-second trail expiry were checked.
- [x] Final local gate: 220 tests passed; Ruff, Compose configuration,
  `git diff --check` and published-media validation passed.
