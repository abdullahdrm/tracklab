# İlerleme Günlüğü

## 2026-10-12: Gün 1
- Kurulum tamam: Python 3.12, `.venv`, PyTorch 2.14.1 (MPS True), ultralytics, Jupyter.
- İlk takip denemesi: COCO yolo11n + ByteTrack, birkaç farklı videoda.
- Baseline gözlemleri:
  - Yakındaki insanlar iyi tespit ediliyor; uzaktakilerin çoğu kaçıyor.
  - Kutu kopunca ID değişiyor (ID switch).
  - Yağmur/sis + araçlı videoda tespit neredeyse yok; araçlar yaklaşınca `person` olarak etiketlendi.
- Hipotezler (Hafta 2-3'te ölçülecek): `imgsz` artırma, fine-tune, `track_buffer` ayarı, ön işleme (gamma/CLAHE).

## 2026-10-14: Gün 2
- YOLO + ByteTrack Python pipeline'ı ve CLI doğrulandı.
- Örnek video için takip edilmiş MP4 ve JSON sonuçları üretildi.
- AGPL-3.0 lisans dosyası ve proje metadata'sı eklendi.
- Doğrulama: 35 test geçti; Ruff kontrolleri ve biçim kontrolü başarılı.
- Not: Testlerde httpx/Starlette deprecation uyarısı var; test başarısını etkilemiyor.

## 2026-10-14: Gün 3
- FastAPI iş akışı ve SQLAlchemy tabanlı kalıcı iş kayıtları doğrulandı.
- Başarılı ve başarısız iş durumları test edildi.
- Boş dosya, desteklenmeyen uzantı ve aşırı büyük yükleme reddi doğrulandı.
- SQLite kaydının uygulama yeniden oluşturulduktan sonra okunması doğrulandı.
- Doğrulama: 38 test geçti; Ruff lint, format ve diff kontrolleri temiz.
- Not: Starlette/httpx deprecation uyarısı sürüyor; test başarısını etkilemiyor.

## 2026-10-14: Gün 4
- Tarayıcı arayüzü ve MP4/JSON indirme akışı hazır.
- FFmpeg ile H.264 MP4 çıktısı ve faststart uyumluluğu mevcut.
- API tarafında varsayılan 10 dakika video süresi ve 500 MB yükleme sınırı uygulanıyor.
- Boş, bozuk, desteklenmeyen uzantılı ve süre sınırını aşan yüklemeler test edildi.
- Arayüzde dosya türü, boyut ve desteklenen durumlarda video süresi için erken kontrol eklendi.
- Doğrulama: 41 test geçti; Ruff, format ve diff kontrolleri temiz.
- Not: Starlette/httpx deprecation uyarısı sürüyor; test başarısını etkilemiyor.

## 2026-10-14: Gün 5
- CPU odaklı Docker imajı, Docker Compose, README ve GitHub Actions CI taslağı eklendi.
- Docker Compose yapılandırması doğrulandı ve imaj başarıyla oluşturuldu.
- Container içindeki eksik `lap` bağımlılığı proje metadata'sına eklendi; imaj yeniden oluşturuldu.
- Container sağlık kontrolü `/health` üzerinden HTTP 200 döndürdü.
- Docker üzerinden `sample2.mp4` başarıyla işlendi: 505 kare, 4.890 tespit, 145 benzersiz takip ID'si.
- MP4 ve JSON çıktıları host üzerindeki `outputs/jobs/` dizinine yazıldı; tarayıcı önizlemesi test edildi.
- `analytics.yaml` ve `preprocessing.yaml`, temiz checkout için gerekli olduklarından sürüm kapsamına dahil edilecek.
- Doğrulama: 41 test geçti; Ruff, format ve diff kontrolleri temiz.


## 2026-10-14: Gün 5 sonrası — v0.1.1 sağlamlaştırma ve dağıtım

- İlk sürüm sonrası kod denetiminde belirlenen API ve video pipeline sorunları giderildi.
- Sunucu yeniden başladığında queued/running işler otomatik devam ettirilemiyor; bu işler açık hata durumuyla kaydediliyor.
- Başarı ve hata senaryolarında yükleme/geçici dosya temizliği güçlendirildi.
- Modelin video çalıştırması sırasında gereksiz yere tekrar oluşturulması önlendi.
- API çıktı yolu doğrulaması, video doğrulamasının thread pool'da çalışması ve hata mesajı maskelemesi eklendi.
- SQLite ayarları, opsiyonel PostgreSQL bağımlılığı, JSON serileştirme ve görüntü sınırındaki çizim için iyileştirmeler yapıldı.
- Kaynak videoların iş tamamlandıktan sonra silindiği davranışı README'ye eklendi.
- Son doğrulama: 52 test geçti; Ruff lint ve format kontrolleri temiz, `git diff --check` başarılı.
- Tek uyarı: Starlette/httpx deprecation uyarısı mevcut; test başarısını etkilemiyor.
- `tracklab:0.1.1` ayrı portta denendi: 505 kare, 4.890 tespit, 145 benzersiz takip ID'si; JSON geçerli ve MP4 indirilebilir.
- Ana Compose servisi `tracklab:0.1.1` imajına geçirildi; `/health`, arayüz ve `/docs` HTTP 200 döndü.
- Canlı veritabanındaki 7 iş kaydı korundu: 4 done, 3 failed. Veritabanı dağıtım öncesi ayrıca yedeklendi.
- Test konteyneri ve yalnızca test için oluşturulan klasörler temizlendi.
- Git geçmişi ve etiket metadata'sından kurumsal e-posta adresi kaldırıldı; yeni depo yerel Git kimliği `noreply@localhost` olarak ayarlandı.
- `v0.1.1` sürüm etiketi doğrulandı ve çalışma ağacı temiz.

**CI notu:** GitHub Actions, GitHub üzerindeki `master` commit'inde çalıştı; lint, format ve test adımlarının tamamı başarılı.

## 2026-10-16: Gün 6 — MOT17 veri katmanı

- MOT17 eğitim arşivi indirildi ve çıkarıldı; 21 sekans klasörü bulundu.
- Deneylerde kullanılacak FRCNN varyantındaki yedi sekansın görüntüleri,
  metadata dosyaları ve ground-truth dosyaları doğrulandı.
- MOT17 lisans ve atıf notları `DATA_LICENSES.md` dosyasına eklendi.
- `src/tracklab/datasets/mot17.py`: sequence metadata doğrulama,
  sekans keşfi ve ham dokuz sütunlu etiket okuyucusu eklendi.
- `src/tracklab/datasets/splits.py` ve `configs/mot17_splits.yaml`:
  tuning (04, 05, 09, 11) ve hold-out (02, 10, 13) ayrımı tanımlandı.
- `scripts/prepare_mot17.py` ile doğrulanmış JSON manifesti üretildi:
  tuning 4 sekans / 3.312 kare; hold-out 3 sekans / 2.004 kare.
- Ham etiket satırı sayıları kaydedildi; henüz geçerli yaya sayısı veya
  takip performansı olarak yorumlanmıyor.
- Dokuz yeni birim testi eklendi; toplam 61 test başarılı.
- Ruff lint ve format kontrolleri temiz.
- Sonraki aşama: TrackEval entegrasyonu ve GT'nin kendisiyle
  değerlendirme yaparak ölçüm hattını doğrulama (E0).


## 2026-10-16: Gün 7 — TrackEval ve E0 doğrulaması

- TrackEval, MIT lisanslı ayrı bir Git submodule olarak eklendi ve commit
  `12c8791b303e0a0b50f753af204249e622d0281a` sürümüne sabitlendi.
- NumPy 2 uyumluluğu, yalnızca TrackEval yüklenirken uygulanan süreç içi
  alias'larla sağlandı; submodule kaynak kodu değiştirilmedi.
- `evaluation/trackeval_bridge.py`, MOT17'nin dokuz sütunlu ham GT biçiminden
  TrackEval'in on sütunlu tahmin biçimine dönüşüm için eklendi.
- Sentetik iki karelik GT self-evaluation testi geçti.
- Gerçek MOT17 FRCNN tuning sekansları 04, 05, 09 ve 11 üzerinde E0 başarılı:
  HOTA=1.0, DetA=1.0, AssA=1.0, LocA=1.0.
- E0 raporu `outputs/mot17/e0/report.json` konumuna yazıldı.
- Test keşfi `testpaths = ["tests"]` olacak şekilde sınırlandı; böylece
  TrackEval'in bağımsız veri kümelerini bekleyen kendi testleri, TrackLab
  test paketiyle karıştırılmıyor.
- Son doğrulama: 63 test başarılı. Mevcut FastAPI/Starlette/httpx deprecation
  uyarısı sürüyor; test başarısını etkilemiyor.
- Not: E0 bir ölçüm hattı sanity check'idir; YOLO baseline performansı değildir.

## 2026-10-16: Gün 8 — İlk MOT17 baseline ölçümü

- `configs/eval.yaml` ve `scripts/evaluate_mot.py` ile MOT17 değerlendirme akışı eklendi.
- `evaluation/mot17_detector_comparison.py`, ByteTrack yaklaşımını iki tespit kaynağıyla değerlendirir: MOT17'nin hazır FRCNN `det.txt` tespitleri ve COCO ön eğitimli YOLO11n.
- İlk YOLO ölçüm ayarları: `imgsz=960`, `confidence=0.10`, `IoU=0.7`, `device=auto`, model `models/yolo11n.pt`; değerlendirme yaya sınıfına odaklanır.
- Yalnızca MOT17-04, MOT17-05, MOT17-09 ve MOT17-11 tuning sekansları kullanıldı. Hold-out sekansları 02, 10 ve 13 değerlendirilmedi.
- TrackEval köprüsüne HOTA ailesinin yanında CLEAR ve Identity metrikleri eklendi: MOTA, IDSW, IDF1, IDP ve IDR.
- Makine tarafından üretilen özet `outputs/mot17/comparison/selected_sequences_summary.json` konumundadır. `outputs/` sürüm kontrolü dışında tutulduğu için sonuçlar bu belgeye de kaydedildi.

### İlk tuning sonuçları

| Sekans | Tespit kaynağı + takip | HOTA | DetA | AssA | LocA | MOTA | IDSW | IDF1 | IDP | IDR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 04 | MOT17 FRCNN tespitleri + ByteTrack | 0.5306 | 0.4838 | 0.5832 | 0.9036 | 0.5424 | 43 | 0.6223 | 0.8821 | 0.4807 |
| 04 | YOLO11n + ByteTrack | 0.4277 | 0.4143 | 0.4472 | 0.8313 | 0.4576 | 142 | 0.4998 | 0.6881 | 0.3924 |
| 05 | MOT17 FRCNN tespitleri + ByteTrack | 0.4293 | 0.4043 | 0.4567 | 0.8429 | 0.4840 | 57 | 0.5455 | 0.8052 | 0.4125 |
| 05 | YOLO11n + ByteTrack | 0.4034 | 0.4182 | 0.3929 | 0.7703 | 0.4292 | 130 | 0.5439 | 0.6281 | 0.4795 |
| 09 | MOT17 FRCNN tespitleri + ByteTrack | 0.4811 | 0.4870 | 0.4755 | 0.8993 | 0.5454 | 17 | 0.5850 | 0.8171 | 0.4556 |
| 09 | YOLO11n + ByteTrack | 0.4325 | 0.5138 | 0.3670 | 0.8181 | 0.5775 | 60 | 0.5078 | 0.5538 | 0.4689 |
| 11 | MOT17 FRCNN tespitleri + ByteTrack | 0.5422 | 0.5256 | 0.5601 | 0.9058 | 0.5719 | 44 | 0.6105 | 0.8077 | 0.4907 |
| 11 | YOLO11n + ByteTrack | 0.4874 | 0.5067 | 0.4730 | 0.8370 | 0.5197 | 39 | 0.5395 | 0.6213 | 0.4768 |
| Tuning birleşik (04/05/09/11) | MOT17 FRCNN tespitleri + ByteTrack | 0.5195 | 0.4813 | 0.5621 | 0.8982 | 0.5408 | 161 | 0.6103 | 0.8587 | 0.4733 |
| Tuning birleşik (04/05/09/11) | YOLO11n + ByteTrack | 0.4351 | 0.4353 | 0.4408 | 0.8246 | 0.4725 | 371 | 0.5111 | 0.6563 | 0.4185 |

Birleşik satırlar, dört sekansın aritmetik ortalaması değil, TrackEval'in sekansları birlikte değerlendirmesinin sonucudur. Metrikler sekanslara göre farklılaşabilir; örneğin MOT17-09 üzerinde YOLO'nun MOTA değeri FRCNN tespitleriyle elde edilenden yüksektir.

**Yorumlama sınırı:** FRCNN satırları MOT17'nin hazır tespit dosyalarını kullanır; bu deneyde bağımsız bir Faster R-CNN modeli çalıştırılmadı. Bu sonuçlar, tespit kaynağının ByteTrack hattındaki etkisine yönelik bir başlangıç referansıdır; kontrollü bir model-model karşılaştırması olarak yorumlanmamalıdır.

Bunlar ilk baseline ölçümleridir. `conf` ve `imgsz` için tarama henüz yapılmadığından Gün 8 tamamlanmış sayılmıyor. Sonraki adım, yalnızca tuning sekanslarında bu parametreleri ölçerek başlangıç ayarlarını sabitlemektir.

- Doğrulama: 68 test geçti; Ruff lint ve format kontrolleri başarılı, `git diff --check` temiz.
- Tek test uyarısı FastAPI/Starlette test istemcisindeki `httpx` deprecation uyarısıdır; test başarısını etkilemiyor.
## 2026-10-16: Gün 8 — conf/imgsz tuning ve baseline seçimi

- Tuning sekanslarında (`MOT17-04/05/09/11-FRCNN`) 3 confidence değeri
  (`0.10`, `0.15`, `0.20`) ve 4 görüntü boyutu (`640`, `736`, `960`, `1280`)
  değerlendirildi: toplam 12 geçerli yapılandırma.
- `imgsz=720` ilk denemede Ultralytics tarafından stride 32 nedeniyle 736'ya
  yuvarlandı. Karşılaştırılabilir sweep için 736 doğrudan kullanıldı; eksik
  kalan eski 720 raporu sonuç tablosuna alınmadı.
- Hold-out sekansları `MOT17-02/10/13-FRCNN` değerlendirilmedi.

### Birleşik tuning sweep sonuçları

| conf | imgsz | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | IDP | IDR | Süre (sn) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 640 | 0.3951 | 0.3489 | 0.4511 | 0.3767 | 0.4598 | 300 | 0.6967 | 0.3431 | 51.7 |
| 0.15 | 640 | 0.3922 | 0.3425 | 0.4528 | 0.3705 | 0.4534 | 301 | 0.6999 | 0.3353 | 58.3 |
| 0.20 | 640 | 0.3878 | 0.3365 | 0.4506 | 0.3642 | 0.4457 | 319 | 0.7013 | 0.3266 | 52.6 |
| 0.10 | 736 | 0.4267 | 0.3847 | 0.4778 | 0.4214 | 0.5103 | 307 | 0.7273 | 0.3930 | 53.2 |
| 0.15 | 736 | 0.4144 | 0.3766 | 0.4606 | 0.4127 | 0.4880 | 309 | 0.7114 | 0.3714 | 62.4 |
| 0.20 | 736 | 0.4116 | 0.3707 | 0.4617 | 0.4059 | 0.4898 | 321 | 0.7254 | 0.3697 | 65.8 |
| 0.10 | 960 | 0.4351 | 0.4353 | 0.4408 | 0.4725 | 0.5111 | 371 | 0.6563 | 0.4185 | 72.7 |
| 0.15 | 960 | 0.4352 | 0.4301 | 0.4458 | 0.4687 | 0.5117 | 362 | 0.6703 | 0.4138 | 76.9 |
| 0.20 | 960 | 0.4336 | 0.4251 | 0.4472 | 0.4649 | 0.5083 | 364 | 0.6766 | 0.4071 | 74.3 |
| 0.10 | 1280 | 0.4554 | 0.4717 | 0.4470 | 0.4627 | 0.5253 | 445 | 0.5995 | 0.4675 | 79.9 |
| 0.15 | 1280 | 0.4566 | 0.4686 | 0.4517 | 0.4662 | 0.5257 | 463 | 0.6094 | 0.4623 | 79.7 |
| 0.20 | 1280 | 0.4544 | 0.4662 | 0.4496 | 0.4697 | 0.5249 | 470 | 0.6172 | 0.4566 | 80.7 |

Birleşik metrikler TrackEval'in dört tuning sekansını birlikte değerlendirmesidir;
sekans bazlı sonuçların aritmetik ortalaması değildir. Süre, dört sekansın
raporlanan inference/tracking süreleri toplamıdır ve donanım/çalışma koşullarına
bağlı olarak değişebilir.

### Başlangıç yapılandırması kararı

`configs/eval.yaml` içindeki `conf=0.10`, `imgsz=960` başlangıç ayarı korundu.
Bu ayarda birleşik HOTA `0.4351`, MOTA `0.4725`, IDF1 `0.5111`, IDSW `371`
ve raporlanan toplam sekans süresi `72.7 s` oldu.

`conf=0.15`, `imgsz=1280` en yüksek HOTA (`0.4566`) ve IDF1 (`0.5257`) verdi;
ancak IDSW `463`'e yükseldi ve MOT17-05 sekansında HOTA `0.3621`, MOTA `0.2514`
ve IDF1 `0.4452`'ye geriledi. Aynı sekansın başlangıç baseline değerleri sırasıyla
HOTA `0.4034`, MOTA `0.4292` ve IDF1 `0.5439` idi. Bu nedenle sadece birleşik
HOTA'yı maksimize etmek yerine sekanslar arası denge, MOTA, IDSW ve çalışma süresi
birlikte değerlendirildi; `0.10/960` genel başlangıç yapılandırması olarak tutuldu.

`conf=0.15`, `imgsz=960` birleşik HOTA'yı yalnızca `0.00008` artırdı ve IDSW'yi
371'den 362'ye düşürdü; buna karşılık MOTA `0.4725`'ten `0.4687`'ye geriledi ve
süre yaklaşık 4.2 saniye arttı. Bu küçük fark mevcut yapılandırmayı değiştirmek
için yeterli görülmedi.

Sweep raporları `outputs/mot17/param_sweep/` altındadır. `outputs/` Git dışında
tutulduğu için tabloda önemli sonuçlar ayrıca kaydedildi. Gün 8 tamamlandı;
sonraki adım Gün 9 kalite metrikleri ve klasik ön işleme modülleridir.

## 2026-10-16: Gün 9 — Kalite ölçümü ve klasik ön işleme

- `src/tracklab/quality/metrics.py` eklendi: BGR kare giriş doğrulaması ve brightness, contrast, Laplacian varyansı tabanlı sharpness ile robust `noise_estimate` hesaplaması.
- `noise_estimate` bir proxy ölçümdür; doku ve kenarlar da değeri etkileyebileceğinden doğrudan sensör gürültüsü olarak yorumlanmamalıdır.
- `src/tracklab/quality/report.py` kalite örneklerinin özetini ve `every_n_frames` aralığında kare örneklemesini sağlar. Örnekleme sıfır tabanlı kare indeksi 0'dan başlar; örnek indeksleri ve özet metrikler JSON'a uygun biçimde hazırlanır.
- Kalite raporu `VideoPipeline` içinde `quality_every_n_frames` ayarıyla üretilir ve `summary["quality"]` altında bellekteki sonuca ve JSON çıktısına eklenir.
- Klasik ön işleme modülleri eklendi: base doğrulamaları, gamma düzeltmesi, LAB parlaklık kanalında CLAHE, bilateral/Fast Non-Local Means denoise ve factory seçimi.
- `DenoiseConfig` için Fast Non-Local Means'e özgü `h`/`h_color` parametreleri eklendi.
- Pipeline entegrasyonunda ön işleme yalnızca tracker girişine uygulanıyor. Kalite metrikleri ve bbox/ID görselleştirmesi orijinal kareyi kullanıyor; kare geometrisi ve `uint8` formatı korunuyor.
- `configs/preprocessing.yaml` varsayılanı `mode: none` olarak kaldı. Yeni filtrelerin nicel takip etkisi henüz ölçülmedi; gamma/CLAHE/denoise karşılaştırmaları ve bozulma deneyleri Gün 10 kapsamındadır.
- Doğrulama: 109 test geçti; Ruff lint, format ve `git diff --check` başarılı. Tek uyarı mevcut FastAPI/Starlette `httpx` deprecation uyarısıdır; test başarısını etkilemedi.

## 2026-10-17: Gün 10 — Sentetik bozulmalar, ablation ve renk uzayı keşfi

- E2 gerçek görüntü ablation'ı tamamlandı: `none`, `gamma`, `clahe`, `denoise`; dört MOT17 tuning sekansı kullanıldı.
- E2b sentetik ablation tamamlandı: düşük ışık, düşük kontrast, Gaussian gürültü ve Gaussian blur bozulmalarının her birinde dört ön işleme koşulu (16 koşul) değerlendirildi.
- Gerçek ve sentetik koşulların ayrı, birleşik TrackEval metrikleri ile tabloları `reports/day10_ablation_results.md` içinde saklanıyor; makine tarafından üretilen JSON'lar `outputs/mot17/ablations/` altındadır ve Git dışında tutulur.
- Ek deneyler: Gamma/CLAHE işlem sıraları ve LAB `L`, HSV `V`, YCrCb `Y` renk-uzayı/kanal varyantları; dört tuning sekansında gerçekleştirildi. Hold-out 02/10/13 kullanılmadı.
- Dengeli renk-uzayı adayı `ycrcb_clahe_gamma`: HOTA=0.4518, IDF1=0.5395; birleşik HOTA bakımından `gamma_clahe` daha yüksek çıktı ancak bazı sekanslarda 30 FPS altına indi.
- FPS ölçümü inference/tracking döngüsünü, ön işleme ve tahmin dosyasına yazımı kapsar; son FastAPI/MP4 kodlama yolunun gecikmesi bu ölçümde dahil değildir. Sentetik bozulma yaratma maliyeti E2b FPS'sine dahildir.
- Kapsam sınırı: E2b gerçek sis/yağmur videosu değildir; MOT17 değerlendirmesi yalnızca yaya sınıfını ölçer. Araç sınıflandırma ve gerçek sis dayanıklılığı henüz doğrulanmadı.
- Yeniden kullanılabilir gamma LUT/CLAHE nesnesi ve derlenmiş preprocessor eklendi; piksel sonuçlarının önceki doğrudan çağrılarla eşleşmesi test edildi.
- Doğrulama: 157 test geçti; Ruff lint, format ve `git diff --check` başarılı. Tek uyarı FastAPI/Starlette/httpx test istemcisi deprecation uyarısıdır; test başarısını etkilemez.
- Optuna ve YOLO fine-tuning henüz çalıştırılmadı; bunlar Week 3 kapsamına bırakıldı.


## 2026-10-18: Gün 11 — MOT17 YOLO veri kümesi dönüştürücüsü

- `src/tracklab/datasets/to_yolo.py` MOT17 yaya anotasyonlarını YOLO sınıfı `0` ve normalize edilmiş `xywh` etiketlerine dönüştürür; `tests/unit/test_to_yolo.py` dönüşüm, sınır kırpma ve split güvenliğini test eder.
- Veri kümesi `data/processed/MOT17_yolo_day11/` altında oluşturuldu. Görüntüler kaynak MOT17 karelerine sembolik bağlantıdır; dosyalar kopyalanmadı.
- Train: MOT17-04/05/09-FRCNN, 2.412 görüntü ve 59.799 geçerli yaya kutusu. Validation: MOT17-11-FRCNN, 900 görüntü ve 9.436 geçerli yaya kutusu.
- Toplam 3.312 görüntü için bire bir etiket dosyası üretildi; anotasyonu olmayan görüntülerin boş etiketleri korundu.
- Denetim `dataset.yaml`, symlink hedefleri, etiket sayısı/şeması, sonlu normalize kutu koordinatları ve sequence-level train/validation ayrımını doğruladı.
- Hold-out MOT17-02/10/13 görüntü yolları/etiketleri çıktıda bulunmadı; veri kümesi manifestteki tuning=3.312 ve hold-out=2.004 kare sayılarını korur.
- Yerel doğrulama: 167 test geçti; Ruff lint/format başarılı. Tek uyarı mevcut Starlette/httpx test istemcisi deprecation uyarısıdır; test başarısını etkilemedi.
- `data/` `.gitignore` kapsamında; üretilmiş görüntü bağlantıları ve etiketler commit'e eklenmeyecek. Kaynak kod, test ve dokümantasyon sürüm kontrolüne alınacak.
