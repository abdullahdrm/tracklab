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
