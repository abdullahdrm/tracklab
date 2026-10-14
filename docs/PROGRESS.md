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

**CI notu:** GitHub Actions workflow'u yapılandırıldı; ancak depoda uzak Git remote olmadığı için GitHub üzerinde gerçek bir workflow çalıştırması henüz gözlemlenmedi.
