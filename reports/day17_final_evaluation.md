# TrackLab — Gün 17 final değerlendirme

**Tarih:** 2026-10-25
**Protocol:** `configs/day17_final_protocol.yaml` (`day17-final-v1`)
**Split:** MOT17 hold-out — 02, 10, 13; bu split ilk kez bu deneyde kullanıldı.

## Frozen protocol

Hold-out sonuçları okunmadan önce şu koşullar commit edilerek sabitlendi:

- `imgsz=960`, `conf=0.10`, `IoU=0.70`, `device=auto`, yalnızca pedestrian sınıfı.
- ByteTrack baseline: `track_high_thresh=0.25`, `track_low_thresh=0.10`,
  `new_track_thresh=0.25`, `track_buffer=30`, `match_thresh=0.80`,
  `fuse_score=true`.
- `coco_none`: COCO YOLO11n, ön işleme yok.
- `coco_clahe`: COCO YOLO11n + Day 10 LAB-L CLAHE.
- `coco_ycrcb_clahe_gamma`: COCO YOLO11n + Day 10 YCrCb
  CLAHE→Gamma adayı.
- `finetuned_none`: BDD100K altı sınıflı fine-tuned checkpoint, ön işleme yok.

Her koşulda aynı tracker ve aynı üç hold-out sekansı kullanıldı. Ayrıntılı
makine çıktısı `outputs/mot17/final_holdout_day17/final_holdout_summary.json`
altındadır; `outputs/` Git dışında tutulur.

## Hold-out sonuçları

### Birleşik TrackEval sonucu

| Koşul | HOTA | MOTA | IDF1 | IDSW |
|---|---:|---:|---:|---:|
| COCO + none | 0.3237 | 0.3334 | 0.3931 | 354 |
| COCO + CLAHE | 0.3238 | 0.3391 | 0.3981 | 371 |
| COCO + YCrCb CLAHE→Gamma | 0.3265 | 0.3376 | 0.3981 | 364 |
| fine-tuned + none | 0.3011 | 0.3389 | 0.3728 | 667 |

Birleşik değerler TrackEval'in sekansları birlikte değerlendirmesidir; sekans
sonuçlarının aritmetik ortalaması değildir.

### Sekans bazında HOTA / MOTA / IDF1 / IDSW

| Koşul | MOT17-02 | MOT17-10 | MOT17-13 |
|---|---|---|---|
| COCO + none | 0.3185 / 0.2867 / 0.3771 / 82 | 0.3265 / 0.4145 / 0.4009 / 157 | 0.3268 / 0.3185 / 0.4088 / 115 |
| COCO + CLAHE | 0.3075 / 0.2965 / 0.3711 / 79 | 0.3381 / 0.4171 / 0.4156 / 171 | 0.3308 / 0.3212 / 0.4188 / 121 |
| COCO + YCrCb CLAHE→Gamma | 0.3184 / 0.2957 / 0.3823 / 81 | 0.3337 / 0.4136 / 0.4020 / 169 | 0.3290 / 0.3207 / 0.4178 / 114 |
| fine-tuned + none | 0.2632 / 0.2654 / 0.3333 / 176 | 0.3235 / 0.4211 / 0.3825 / 321 | 0.3301 / 0.3656 / 0.4221 / 170 |

Her hücre sırasıyla `HOTA / MOTA / IDF1 / IDSW` içerir.

## Sonuç

- Fine-tuning hold-out'ta da COCO baseline'ı geçmedi: HOTA `0.3237` →
  `0.3011`, IDF1 `0.3931` → `0.3728`, IDSW `354` → `667`.
- YCrCb CLAHE→Gamma, COCO baseline'a göre küçük bir HOTA kazanımı verdi
  (`+0.0027`) fakat IDSW arttı (`+10`); bu nedenle güçlü bir genel iyileşme
  iddiası yapılmıyor.
- CLAHE, MOT17-10'da iyileşirken MOT17-02'de geriledi ve birleşik IDSW'yi
  artırdı. Ön işleme seçimi veri koşuluna duyarlı görünüyor.
- Üretim varsayılanları COCO + preprocessing `none` + Day 16 baseline
  ByteTrack olarak bırakıldı. Hold-out sonucu sonradan parametre seçmek için
  kullanılmadı.
