# TrackLab — Gün 16 tracker ablation sonucu

**Tarih:** 2026-10-25
**Deney:** E4 — ByteTrack parametre taraması
**Split:** Yalnızca tuning: MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-11-FRCNN

## Protokol

- Dedektör: COCO ön eğitimli `models/yolo11n.pt`.
- YOLO ayarları: `imgsz=960`, `conf=0.10`, `IoU=0.70`, `classes=[0]`.
- Ön işleme: `none`.
- Tarama: `track_buffer ∈ {15, 30, 60}`, `match_thresh ∈ {0.7, 0.8, 0.9}` ve `track_high_thresh/new_track_thresh ∈ {0.15, 0.25}` için küçük, önceden belirlenmiş aday seti.
- Değerlendirme: TrackEval MOTChallenge2DBox; HOTA, MOTA, IDF1 ve IDSW.
- Hold-out sekansları `MOT17-02/10/13-FRCNN` bu deneyde kullanılmadı.

Çalıştırma:

```bash
.venv/bin/python scripts/run_tracker_ablation.py \
  --output-root outputs/mot17/tracker_ablation_day16
```

Makine tarafından üretilen ayrıntılı sonuç: `outputs/mot17/tracker_ablation_day16/tracker_ablation_summary.yaml`.

## Birleşik sonuçlar

| Aday | buffer | match | high/new | HOTA | MOTA | IDF1 | IDSW |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 30 | 0.8 | 0.25 | 0.4351 | 0.4725 | 0.5111 | 371 |
| `buffer15` | 15 | 0.8 | 0.25 | 0.4324 | 0.4712 | 0.5065 | 372 |
| `buffer60` | 60 | 0.8 | 0.25 | 0.4366 | 0.4720 | 0.5137 | 376 |
| `match07` | 30 | 0.7 | 0.25 | 0.4113 | 0.4341 | 0.4726 | 741 |
| `match09` | 30 | 0.9 | 0.25 | 0.4483 | 0.4713 | 0.5324 | 327 |
| `thresholds015` | 30 | 0.8 | 0.15 | 0.4329 | 0.4619 | 0.5134 | 404 |

Birleşik satırlar TrackEval'in dört sekansı birlikte değerlendirmesidir; sekans
sonuçlarının aritmetik ortalaması değildir.

## Karar

`match09` adayı birleşik HOTA'yı `0.4351` → `0.4483`, IDF1'i
`0.5111` → `0.5324` yükseltti ve IDSW'yi `371` → `327` düşürdü. Ancak kazanım
sekanslar arasında eşit değil: MOT17-05'te HOTA `0.4034` → `0.3955`, MOTA
`0.4292` → `0.4073` ve IDF1 `0.5439` → `0.5232` geriledi. Bu nedenle yalnızca
birleşik HOTA'yı maksimize etmek yerine, hiçbir tuning sekansında belirgin
gerileme olmaması koşulu uygulanınca `match09` üretim tracker'ı olarak seçilmedi.

`buffer60` ise baseline'a göre HOTA ve IDF1'i çok küçük miktarda artırırken
IDSW'yi 371'den 376'ya yükseltti; bu fark da dengeli bir üretim kararı için
yeterli değil.

**Gün 17 final protokolü için tracker kararı: mevcut baseline korunuyor**
(`track_buffer=30`, `match_thresh=0.8`, `track_high_thresh=0.25`,
`new_track_thresh=0.25`, `fuse_score=true`). Bu karar hold-out sonuçları
görüldükten sonra değiştirilmeyecek; hold-out yalnızca sabitlenmiş final
protokolü değerlendirmek için kullanılacak.

## Sınırlamalar

- Bu tarama yalnızca dört tuning sekansında yapıldı; hold-out performansı hakkında
  genelleme yapmaz.
- Değerlendirme `pycocotools` gerektiren TrackEval BURST modülünü kullanmadı;
  MOTChallenge2DBox metrikleri başarıyla çalıştı.
- Çıktılar `outputs/` altında tutulur ve `.gitignore` nedeniyle commit edilmez;
  önemli sonuçlar bu raporda saklanmıştır.
