# TrackLab — Gün 18 test, CI ve Docker doğrulaması

**Tarih:** 2026-10-25

## Yerel sözleşme

- `207 passed, 1 warning` ile tüm test paketi başarılı.
- Ruff lint başarılı:
  `ruff check src/tracklab evaluation scripts tests`.
- Ruff biçim kontrolü başarılı:
  `ruff format --check src/tracklab evaluation scripts tests`.
- `git diff --check` temiz.
- Tek uyarı, mevcut FastAPI/Starlette `httpx` test istemcisi deprecation
  uyarısıdır; test başarısını etkilemiyor.

Day 17'de eklenen frozen hold-out protocol testi ve pipeline latency summary
testi bu pakete dahildir.

## GitHub Actions

Master üzerinde `ace67c5` için GitHub Actions CI run `40193226323` başarıyla
tamamlandı. `Install FFmpeg`, bağımlılık kurulumu, Ruff lint, format ve pytest
adımlarının tamamı yeşil. Docker smoke job'ı eklendikten sonra master üzerindeki
`b958569` için `40194671710` run'ı da test ve Docker job'larıyla başarıyla
tamamlandı.

GitHub runner iki bilgilendirme annotation'ı üretti: bazı action'lar Node.js 20
hedefliyor ve `ubuntu-latest` Ubuntu 26'ya taşınacak. Bunlar proje test
başarısızlığı değil, geleceğe dönük runner uyarılarıdır.

## Docker

- `docker compose -f docker/docker-compose.yml config` başarılı; servis, port,
  volume, environment ve healthcheck tanımları çözümlendi.
- Yerel Docker CLI/Compose kurulu olsa da bu çalışma ortamında Docker daemon
  socket'i (`~/.docker/run/docker.sock`) mevcut değildi; bu nedenle yerel
  image build'i çalıştırılamadı.
- GitHub Actions `40194671710` Docker job'ı `docker/Dockerfile` ile image'ı
  oluşturdu ve konteynerde `/health` endpoint'ini doğruladı. Build context'i
  `.dockerignore` ile yerel veri/model/çıktılardan ayrıştırıldı; startup için
  smoke curl komutuna retry eklendi.

Bu nedenle Gün 18'in test/CI ve Docker image smoke kabul ölçütü tamamlandı;
yalnızca bu makinede yerel daemon bulunmadığı ayrıca belirtilmiştir.
