# TrackLab — E5 elle sayım karşılaştırması

**Kaynak:** MOT17-13-FRCNN hold-out görüntüleri
**Koşul:** Gün 17 protokolündeki `coco_none` tahminleri
**Denetim aralığı:** 95–140. kareler, 25 FPS; yaklaşık 1.84 saniye
**Çizgi:** yatay `p1=(100,900)`, `p2=(1820,900)`; ölçüm noktası `foot`

## Karşılaştırma

| Kaynak | Sayım |
|---|---:|
| Elle, ardışık kare denetimi | 1 |
| Sistem, aynı aralıkta | 1 |
| Mutlak fark | 0 |

Sistem olayı track ID 4 için frame 123'te `negative_to_positive` olarak
raporlandı. Kare denetiminde aynı yaya çizgiyi yukarı taraftan aşağı tarafa
geçiyor; kısa süreli kutu titreşimleri ayrı bir geçiş olarak sayılmadı.

Denetlenebilir görsel: [MOT17-13 kare denetimi](../docs/assets/day17_manual_count_audit.jpg)

Bu küçük, tek sekanslı E5 örneği genel insan sayım doğruluğu değildir; yalnızca
analytics hattının gerçek görüntü üzerinde bir manuel referansla kontrolüdür.
