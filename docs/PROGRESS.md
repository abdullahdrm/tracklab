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
