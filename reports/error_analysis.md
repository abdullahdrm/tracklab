# TrackLab — Hata analizi

## Hold-out bulguları

Gün 17 protokolünde fine-tuned model, üç hold-out sekansının tamamında birleşik
COCO baseline'ın altında kaldı. En büyük fark MOT17-02'de görüldü:
fine-tuned HOTA `0.2632`, COCO baseline HOTA `0.3185`; fine-tuned IDSW `176`,
baseline IDSW `82`. MOT17-10'da fine-tuned IDSW `321` ile baseline'ın `157`
değerinin iki katından fazla oldu.

Bu desen, Day 14 tuning sonucuyla tutarlıdır ve fine-tuning'in yalnızca tuning
splitine özgü bir rastlantı olmadığını gösterir. Buna rağmen tek başına neden
kanıtlamaz.

## Olası hata sınıfları

- **Domain shift:** BDD100K görüntüleri ile MOT17 kamera, perspektif, yoğunluk
  ve görüntü kompozisyonu farklıdır.
- **Sınıf/etiket hedefi farkı:** Fine-tuned model altı BDD100K sınıfı için
  optimize edildi; MOT17 değerlendirmesi yalnızca yaya takip eder.
- **Catastrophic forgetting / calibration:** BDD100K eğitimi COCO'nun genel
  yaya skor kalibrasyonunu ve küçük/uzak yaya davranışını bozmuş olabilir.
- **İz sürekliliği:** Fine-tuned koşulda yüksek IDSW, sorunun yalnızca kutu
  doğruluğu değil, ByteTrack'e giren kutuların sürekliliği olduğunu gösterir.

## Kapsam sınırı

Bu rapor görüntü başına yanlış pozitif/yanlış negatif örneklerini sınıflandıran
ayrıntılı bir insan anotasyon çalışması değildir. Sayısal kanıt TrackEval
metrikleri, sekans tabloları ve E5 kare denetimiyle sınırlıdır. Fine-tuned
modeli üretim varsayılanı yapmama kararı bu kanıta dayanır; yeni bir veri seti,
class-balanced eğitim veya MOT17'ye özgü domain adaptation yapılmadı.
