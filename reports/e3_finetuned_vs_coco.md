# E3 — COCO YOLO11n vs BDD100K fine-tuned YOLO11n

## Protokol

- Split: tuning; sekanslar: MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-11-FRCNN.
- Her iki modelde aynı imgsz, confidence, IoU ve ByteTrack ayarları kullanıldı.
- Her iki modelde de hedef sınıf `person` (modelin kendi etiket alanında class ID 0).
- Hold-out sekansları kullanılmadı.
- COCO ağırlıkları: `models/yolo11n.pt`
- Fine-tuned ağırlıkları: `outputs/training/bdd100k_yolo11n_six_class/weights/best.pt`

## Birleşik metrikler

| Metrik | COCO baseline | Fine-tuned | Değişim (FT − COCO) |
|---|---:|---:|---:|
| HOTA | 0.4351 | 0.3316 | -0.1035 |
| DetA | 0.4353 | 0.3233 | -0.1120 |
| AssA | 0.4408 | 0.3528 | -0.0879 |
| LocA | 0.8246 | 0.7926 | -0.0319 |
| MOTA | 0.4725 | 0.2802 | -0.1923 |
| IDSW | 371 | 476 | +105 |
| IDF1 | 0.5111 | 0.3790 | -0.1320 |
| IDP | 0.6563 | 0.5221 | -0.1342 |
| IDR | 0.4185 | 0.2975 | -0.1210 |

## Sekans bazında sonuçlar

| Sekans | HOTA COCO | HOTA FT | Δ HOTA | MOTA COCO | MOTA FT | IDF1 COCO | IDF1 FT | IDSW COCO | IDSW FT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MOT17-04-FRCNN | 0.4277 | 0.3120 | -0.1158 | 0.4576 | 0.2553 | 0.4998 | 0.3527 | 142 | 221 |
| MOT17-05-FRCNN | 0.4034 | 0.2347 | -0.1687 | 0.4292 | 0.1807 | 0.5439 | 0.3032 | 130 | 66 |
| MOT17-09-FRCNN | 0.4325 | 0.3722 | -0.0603 | 0.5775 | 0.4246 | 0.5078 | 0.4307 | 60 | 111 |
| MOT17-11-FRCNN | 0.4874 | 0.4328 | -0.0546 | 0.5197 | 0.3968 | 0.5395 | 0.5024 | 39 | 78 |

## Yorum

- Fine-tuned modelin birleşik HOTA, DetA, AssA, MOTA ve IDF1 değerleri baseline'dan düşük; IDSW daha yüksek.
- HOTA dört tuning sekansının tamamında düştü. En büyük düşüş MOT17-05'te görüldü.
- BDD100K validation sonuçları MOT17 takip performansını tek başına öngörmedi.
- Bu sonuçlara dayanarak oversampling veya class weighting hemen uygulanmayacak; öncelik hata analizi ve modelin MOT17'de neden genelleşmediğini incelemek.
- Sonuçlar tuning sekanslarıyla sınırlıdır; final hold-out değerlendirmesi Gün 17'ye ayrılmıştır.

Kaynak JSON raporları: `outputs/mot17/e3_coco/tuning_summary.json` ve `outputs/mot17/e3_finetuned/tuning_summary.json`.
