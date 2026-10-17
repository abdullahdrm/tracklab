# TrackLab — Gün 10 ön işleme ablation sonuçları

**Tarih:** 2026-10-17

**Durum:** E2 gerçek görüntü ablation'ı ve E2b sentetik bozulma ablation'ı tamamlandı.
**Çıktıların kaynağı:** `outputs/` altındaki tamamlanmış JSON raporları; bu dizin Git dışında tutulur. Bu Markdown raporu önemli sonuçları sürüm kontrolünde saklar.

## Protokol

- Tespit ve takip: COCO ön eğitimli YOLO11n + ByteTrack; yalnızca yaya sınıfı (`classes=[0]`).
- YOLO ayarları: `imgsz=960`, `conf=0.10`, `IoU=0.70`, MPS aygıtı.
- Tuning sekansları: MOT17-04-FRCNN, MOT17-05-FRCNN, MOT17-09-FRCNN, MOT17-11-FRCNN.
- Hold-out sekansları MOT17-02/10/13 bu deneylerin hiçbirinde kullanılmadı.
- HOTA, DetA, AssA, MOTA, IDF1 ve IDSW dört sekans birlikte TrackEval ile ölçüldü; birleşik metrikler sekans metriklerinin aritmetik ortalaması değildir.
- **FPS ölçüm sınırı:** inference/tracking döngüsü; görüntü okuma, seçilen ön işleme, YOLO, ByteTrack ve tahmin dosyasına yazma süresini kapsar. TrackEval metrik hesaplama süresi ve nihai FastAPI/MP4 kodlama hattı bu FPS'ye dahil değildir.

## E2 — Gerçek görüntüler

Aynı gerçek MOT17 kareleri için ön işleme uygulanmadan ve tek tek klasik yöntemler uygulanarak karşılaştırma yapıldı.

| Koşul | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Süre (sn) | FPS* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0.4351 | 0.4353 | 0.4408 | 0.4725 | 0.5111 | 371 | 67.4 | 49.14 |
| `gamma` | 0.4352 | 0.4367 | 0.4397 | 0.4776 | 0.5083 | 369 | 60.6 | 54.67 |
| `clahe` | 0.4512 | 0.4375 | 0.4703 | 0.4747 | 0.5373 | 372 | 95.6 | 34.66 |
| `denoise` | 0.4373 | 0.4350 | 0.4459 | 0.4708 | 0.5128 | 368 | 92.0 | 36.00 |

## E2b — Sentetik bozulmalar

Dört kontrollü bozulma uygulandı: düşük ışık, düşük kontrast, Gaussian gürültü ve Gaussian bulanıklık. Her bozulmada aynı `none`/`gamma`/`clahe`/`denoise` koşulları çalıştırıldı. Bu deney bozulma altında göreli dayanıklılığı ölçer; gerçek sis/yağmur videosu yerine geçmez.

| Koşul | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Süre (sn) | FPS* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `low_light__none` | 0.4394 | 0.4379 | 0.4462 | 0.4755 | 0.5151 | 355 | 69.3 | 47.82 |
| `low_light__gamma` | 0.4403 | 0.4365 | 0.4495 | 0.4709 | 0.5181 | 345 | 74.0 | 44.77 |
| `low_light__clahe` | 0.4424 | 0.4369 | 0.4530 | 0.4777 | 0.5205 | 339 | 109.1 | 30.36 |
| `low_light__denoise` | 0.4394 | 0.4365 | 0.4479 | 0.4720 | 0.5161 | 349 | 102.1 | 32.43 |
| `low_contrast__none` | 0.4495 | 0.4320 | 0.4731 | 0.4717 | 0.5352 | 341 | 82.5 | 40.16 |
| `low_contrast__gamma` | 0.4504 | 0.4304 | 0.4765 | 0.4727 | 0.5362 | 318 | 87.0 | 38.07 |
| `low_contrast__clahe` | 0.4384 | 0.4388 | 0.4433 | 0.4777 | 0.5172 | 361 | 129.7 | 25.54 |
| `low_contrast__denoise` | 0.4371 | 0.4297 | 0.4502 | 0.4708 | 0.5143 | 323 | 121.5 | 27.26 |
| `gaussian_noise__none` | 0.4333 | 0.4203 | 0.4510 | 0.4588 | 0.5132 | 357 | 161.0 | 20.58 |
| `gaussian_noise__gamma` | 0.4363 | 0.4182 | 0.4598 | 0.4575 | 0.5169 | 326 | 164.8 | 20.09 |
| `gaussian_noise__clahe` | 0.4290 | 0.4181 | 0.4451 | 0.4557 | 0.5021 | 351 | 197.2 | 16.80 |
| `gaussian_noise__denoise` | 0.4323 | 0.4243 | 0.4459 | 0.4641 | 0.5122 | 359 | 192.5 | 17.20 |
| `gaussian_blur__none` | 0.4426 | 0.4312 | 0.4606 | 0.4692 | 0.5213 | 349 | 69.7 | 47.52 |
| `gaussian_blur__gamma` | 0.4306 | 0.4307 | 0.4369 | 0.4731 | 0.5052 | 347 | 74.7 | 44.31 |
| `gaussian_blur__clahe` | 0.4364 | 0.4332 | 0.4453 | 0.4712 | 0.5177 | 312 | 115.3 | 28.73 |
| `gaussian_blur__denoise` | 0.4286 | 0.4296 | 0.4337 | 0.4671 | 0.5022 | 344 | 107.1 | 30.91 |

## Ek E2 araştırması — Sabit birleşik zincirler

Gün 10 kapsamını genişletmek için işlem sırası ve birden fazla yöntemin bir arada kullanılması da incelendi.

| Koşul | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Süre (sn) | FPS* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `gamma_clahe` | 0.4542 | 0.4390 | 0.4749 | 0.4770 | 0.5354 | 366 | 99.6 | 33.25 |
| `clahe_gamma` | 0.4428 | 0.4395 | 0.4516 | 0.4787 | 0.5229 | 349 | 99.0 | 33.44 |
| `denoise_clahe` | 0.4420 | 0.4387 | 0.4505 | 0.4767 | 0.5249 | 340 | 139.8 | 23.69 |
| `clahe_denoise` | 0.4428 | 0.4405 | 0.4505 | 0.4771 | 0.5258 | 349 | 138.8 | 23.86 |
| `gamma_clahe_denoise` | 0.4428 | 0.4418 | 0.4490 | 0.4771 | 0.5204 | 371 | 144.8 | 22.87 |
| `denoise_gamma_clahe` | 0.4377 | 0.4395 | 0.4415 | 0.4783 | 0.5138 | 363 | 145.6 | 22.74 |

## Ek araştırma — Renk uzayı ve kanal seçimi

LAB `L`, HSV `V` ve YCrCb `Y` kanalı üzerinde CLAHE ve gamma/CLAHE sıra varyantları karşılaştırıldı. Görseller YOLO'ya verilmeden önce tekrar BGR'ye çevrildi.

| Koşul | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Süre (sn) | FPS* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0.4351 | 0.4353 | 0.4408 | 0.4725 | 0.5111 | 371 | 71.9 | 46.06 |
| `gamma` | 0.4352 | 0.4367 | 0.4397 | 0.4776 | 0.5083 | 369 | 65.7 | 50.40 |
| `clahe` | 0.4512 | 0.4375 | 0.4703 | 0.4747 | 0.5373 | 372 | 99.2 | 33.39 |
| `gamma_clahe` | 0.4542 | 0.4390 | 0.4749 | 0.4770 | 0.5354 | 366 | 101.4 | 32.66 |
| `hsv_clahe` | 0.4377 | 0.4367 | 0.4443 | 0.4741 | 0.5171 | 377 | 83.3 | 39.74 |
| `ycrcb_clahe` | 0.4461 | 0.4377 | 0.4598 | 0.4732 | 0.5291 | 366 | 79.1 | 41.88 |
| `lab_gamma_clahe` | 0.4446 | 0.4375 | 0.4570 | 0.4767 | 0.5270 | 352 | 103.4 | 32.02 |
| `hsv_gamma_clahe` | 0.4336 | 0.4397 | 0.4335 | 0.4822 | 0.5126 | 382 | 91.2 | 36.31 |
| `ycrcb_gamma_clahe` | 0.4420 | 0.4397 | 0.4498 | 0.4758 | 0.5216 | 375 | 83.5 | 39.67 |
| `lab_clahe_gamma` | 0.4489 | 0.4406 | 0.4627 | 0.4808 | 0.5330 | 342 | 97.6 | 33.92 |
| `hsv_clahe_gamma` | 0.4438 | 0.4388 | 0.4542 | 0.4806 | 0.5251 | 362 | 81.9 | 40.46 |
| `ycrcb_clahe_gamma` | 0.4518 | 0.4409 | 0.4684 | 0.4798 | 0.5395 | 357 | 79.8 | 41.51 |

## Optimizasyon sonrası tekrarlı gerçek-zamanlı benchmark

Gamma LUT ve CLAHE nesnesini pipeline kurulurken tekrar kullanılabilir hale getiren refaktör sonrası üç kontrol koşulu yeni bir çıktı dizininde yeniden çalıştırıldı. Bu, önceki sürelerle birebir kıyaslanabilecek sabit bir donanım benchmark'ı değildir; ortam yükü FPS'yi değiştirebilir.

| Koşul | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | Süre (sn) | FPS* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0.4351 | 0.4353 | 0.4408 | 0.4725 | 0.5111 | 371 | 71.0 | 46.67 |
| `clahe` | 0.4512 | 0.4375 | 0.4703 | 0.4747 | 0.5373 | 372 | 94.9 | 34.91 |
| `gamma_clahe` | 0.4542 | 0.4390 | 0.4749 | 0.4770 | 0.5354 | 366 | 100.3 | 33.02 |

## Bulgular ve sınırlar

- E2'de CLAHE, ön işlemesiz baseline'a kıyasla HOTA/IDF1 iyileşmesi gösterdi. Gamma tek başına benzer bir HOTA kazanımı sağlamadı.
- E2b'nin sonuçları bozulmaya bağlı: örneğin `low_light__clahe` fayda gösterirken `low_contrast__clahe` ve gürültü koşullarında CLAHE geriledi. Bu, tek bir filtreyi her karede uygulamak yerine görüntü koşuluna göre seçim araştırmasını destekler.
- Sentetik gürültü koşullarının FPS'si, sentetik bozulma üretimi de zamanlanan transform içinde olduğu için normal üretim pipeline'ı FPS'si olarak yorumlanmamalıdır.
- Renk-uzayı taramasında `ycrcb_clahe_gamma` bir denge adayıdır; `gamma_clahe` HOTA bakımından yüksek sonuç verdi ancak bazı sekanslarda 30 FPS'nin altına indi. Bunlar tuning sonuçlarıdır, final hold-out sonucu değildir.
- Sentetik deney setinde gerçek sis/yağmur ve araç sınıflandırma problemi kapsanmıyor. Mevcut MOT17 değerlendirmesi yalnızca yayaları ölçüyor.
- Optuna araması, YOLO fine-tuning ve hold-out final değerlendirmesi henüz yapılmadı; Week 3/4 görevlerinde kalmalı.

## Yeniden üretilebilir dosyalar

- E2 kodu: `evaluation/ablation.py`, `evaluation/degradations.py`, `scripts/run_ablation.py`.
- E2b, aynı runner'ın `--stage synthetic` aşamasıdır.
- Renk uzayı koşulları: `src/tracklab/preprocessing/colorspace.py`, `scripts/run_colorspace_ablation.py`.
- Gerçek zamanlı üç koşul: `scripts/run_realtime_ablation.py`.
- Birim testleri: `tests/unit/test_ablation.py`, `tests/unit/test_degradations.py`, `tests/unit/test_colorspace.py`.

* FPS alanı toplam kare sayısı / toplam raporlanan inference-tracking süresi olarak yorumlanmalıdır. Sentetik aşamada sentetik bozulma oluşturma süresi de bu değere dahildir.
