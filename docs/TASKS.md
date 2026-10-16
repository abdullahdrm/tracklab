# TrackLab: Görev Listesi

Repoda `docs/TASKS.md` olarak durur. Bir görevi bitirince `[ ]` işaretini `[x]` yap
(VS Code'da önizleme: Cmd+Shift+V). Kısa notlar için `docs/PROGRESS.md` kullanılır.
Ayrıntılar `docs/PROJECT_PLAN.md` (devir belgesi) içindedir.

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
- [ ] `quality/metrics.py`, `quality/report.py`
- [ ] `preprocessing/` (base, gamma, clahe, gerekirse denoise, factory)
- [ ] Birim testleri (geometri değişmezi dahil)

**Bitti ölçütü:** Ön işleyiciler geometriyi değiştirmiyor, testler yeşil.

### Gün 10: Sentetik bozulmalar ve ablation
- [ ] `evaluation/degradations.py`, `ablation.py`, `scripts/run_ablation.py`
- [ ] **E2:** none / gamma / CLAHE / (denoise) gerçek videolarda
- [ ] **E2b:** aynı deney sentetik bozulmalar altında

**Bitti ölçütü:** Gerçek ve sentetik sonuçlar ayrı tablolarda.

---

## Hafta 3: Fine-tune ve analitik

### Gün 11: YOLO veri seti
- [ ] `datasets/to_yolo.py`, `tests/unit/test_to_yolo.py`
- [ ] `dataset.yaml` üretildi, hold-out sızıntısı yok

### Gün 12-13: Fine-tune
- [ ] `training/finetune.py`, `configs/finetune.yaml`
- [ ] Nano model fine-tune (imgsz pilotu)
- [ ] `best.pt` ve eğitim metrikleri kaydedildi

### Gün 14: Fine-tuned vs baseline
- [ ] **E3:** tuning iç doğrulamasında karşılaştırma

**Bitti ölçütü:** Önce/sonra tablosu.

### Gün 15: Analitik
- [ ] `analytics/geometry.py`, `line_counter.py`, `trajectories.py`
- [ ] `viz/draw.py` içinde sayaç ve iz çizimi
- [ ] Birim testleri

**Bitti ölçütü:** Demo videosunda sayaç ve rotalar görünüyor.

---

## Hafta 4: Teslim

### Gün 16: Tracker parametreleri
- [ ] **E4:** `track_buffer`, `match_thresh`, eşikler (küçük tarama, tuning'de)

### Gün 17: Final değerlendirme
- [ ] Hold-out üzerinde final değerlendirme (bir kez)
- [ ] **E5:** elle sayım vs sistem sayımı
- [ ] **E6:** hız ve aşama bazlı gecikme
- [ ] Hata analizi (`reports/error_analysis.md`)

### Gün 18: Test ve CI
- [ ] Eksik testler tamamlandı
- [ ] CI yeşil

### Gün 19: Dokümantasyon
- [ ] README (mimari diyagram, demo GIF, sonuç tablosu, kurulum)
- [ ] `MODEL_CARD.md`, `DATA_LICENSES.md`

### Gün 20: Teslim
- [ ] 60-90 sn demo videosu
- [ ] `v1.0` etiketi
- [ ] Canlı demo kararı

---

## Proje bitti sayılma ölçütü
- [ ] `docker compose up` ile servis çalışıyor; yüklenen videodan ID'li MP4 + JSON + sayım çıkıyor
- [ ] Hold-out sonuçları: baseline, fine-tuned ve ön işleme varyantları sekans bazında tabloda
- [ ] Testler ve CI yeşil
- [ ] README, MODEL_CARD, DATA_LICENSES, LICENSE mevcut
- [ ] Demo videosu hazır


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
- [x] Yerel doğrulama: 52 test geçti; Ruff lint, format ve diff kontrolleri başarılı.
- [x] `tracklab:0.1.1` Docker imajı oluşturuldu ve ayrı portta denendi.
- [x] Gerçek video testi: 505 kare, 4.890 tespit, 145 benzersiz takip ID'si; JSON ve MP4 çıktıları doğrulandı.
- [x] Ana Docker Compose servisi `tracklab:0.1.1` sürümüne geçirildi ve `/health` kontrolü doğrulandı.
- [x] Mevcut veritabanındaki 7 iş kaydının korunduğu doğrulandı.
- [x] `v0.1.1` etiketi oluşturuldu; aktif Git commit/etiket geçmişinden kurumsal e-posta adresi kaldırıldı.

**Hafta 1 kapanış durumu:** Yerel geliştirme, test, Docker dağıtımı ve sürümleme tamamlandı. GitHub Actions, GitHub üzerindeki master commit'inde çalıştırıldı; lint, format ve test adımlarının tamamı başarılı.
