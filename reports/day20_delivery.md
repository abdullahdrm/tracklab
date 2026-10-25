# TrackLab — Gün 20 teslim kaydı

**Tarih:** 2026-10-25

## Demo

Gerçek MOT17 görüntü dizilerinden üretilen ve TrackLab pipeline'ından geçirilen
çıktılar birleştirilerek aşağıdaki yerel demo hazırlandı:

- `outputs/final_demo/tracklab_final_demo.mp4`
- 960×540, 25 FPS, 74.2 saniye
- MOT17-13: 750 kare, 8.297 detection, 230 benzersiz track ID, 21 line-count
  olayı
- MOT17-02: 600 kare, 7.541 detection, 171 benzersiz track ID, 11 line-count
  olayı
- README önizlemesi: `docs/assets/demo_preview.gif` (gerçek pipeline
  çıktısının ilk 10 saniyesi)

Tam MP4, yerel veri ve lisans koşulları nedeniyle Git'e eklenmedi. Tekrar
üretim komutları README ve ilgili config/script dosyalarında yer alıyor.

## Release gate

- Yerel kalite kapısı: `207 passed, 1 warning`; Ruff lint, format ve
  `git diff --check` başarılı.
- GitHub Actions run `40194671710`: test job'ı başarılı.
- Aynı run'ın Docker job'ı başarılı: `docker/Dockerfile` ile image build edildi
  ve konteyner `/health` endpoint'i smoke test edildi.
- Hold-out protokolü ve sonuçları `reports/day17_final_evaluation.md` altında;
  fine-tuned modelin baseline'a göre gerilemesi raporlandı ve varsayılan model
  COCO YOLO11n olarak bırakıldı.
- Release etiketi: `v1.0`.

## Canlı demo kararı

Bu sürüm için public canlı demo açılmadı. Teslim, local Docker/API demo olarak
tamamlandı. Public deployment için henüz authentication, upload quota,
durable worker queue, production observability ve model/veri lisanslarının
hosted kullanımına dair ayrı doğrulama bulunmuyor. Bu karar başarısızlık değil,
mevcut kapsam ve doğrulama kanıtlarının sınırlandırdığı bilinçli bir v1.0
kararıdır.
