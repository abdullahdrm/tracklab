# TrackLab

TrackLab is a portfolio-oriented computer vision application for video object
detection, multi-object tracking, and basic video analytics.

It uses Ultralytics YOLO and ByteTrack, a Python processing pipeline, a FastAPI
service, and a browser-based upload interface.

## Features

- YOLO-based detection and ByteTrack-based object tracking.
- Annotated MP4 output and JSON frame/track results.
- Configurable detector, tracker, preprocessing, and analytics settings.
- FastAPI upload and job-status endpoints.
- Browser interface for uploading videos and downloading results.
- Video upload size and duration validation.
- H.264 MP4 output for browser playback.
- Unit and API integration tests.
- Docker Compose deployment for CPU inference.

## Requirements

- Python 3.12 for local development.
- FFmpeg for H.264 output conversion.
- Docker Desktop or another working Docker Engine for container deployment.
- The `yolo11n.pt` model weights.

Model weights are not committed to this repository. Keep the weights in
`models/yolo11n.pt`. For a fresh checkout, download the official YOLO11n asset:

```bash
mkdir -p models
curl -fL \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt \
  -o models/yolo11n.pt
```

Review the applicable Ultralytics licensing terms before redistributing model
weights or deploying the application.

## Local development

Create and activate a Python 3.12 virtual environment, then install the project:

```bash
python -m pip install -e ".[dev]"
```

Install FFmpeg separately if it is not already available.

On macOS with Homebrew:

```bash
brew install ffmpeg
```

On Ubuntu/Debian:

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

### Process a video

```bash
python scripts/run_video.py /path/to/video.mp4 \
  --config configs/pipeline.yaml \
  --output-dir outputs/demo
```

The exact output files depend on the enabled output settings. The default
configuration writes an annotated MP4 and a JSON result.

### Start the API

From the repository root:

```bash
uvicorn tracklab.api.main:app --host 127.0.0.1 --port 8000
```

Open the upload interface at <http://127.0.0.1:8000/> and the interactive API
documentation at <http://127.0.0.1:8000/docs>.

The main endpoints are:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/jobs` | Upload a video and start a job |
| `GET` | `/jobs/{job_id}` | Get job status and summary |
| `GET` | `/jobs/{job_id}/result` | Download JSON results |
| `GET` | `/jobs/{job_id}/preview` | Play the processed video |
| `GET` | `/jobs/{job_id}/video` | Download the processed MP4 |

The default upload limit is 500 MiB and the default video duration limit is
600 seconds (10 minutes). These are configurable with the environment variables
`TRACKLAB_MAX_UPLOAD_BYTES` and `TRACKLAB_MAX_VIDEO_DURATION_SECONDS`.

Uploaded source videos are temporary: the API deletes them after processing
succeeds or fails, and also deletes uploads rejected during validation.
Generated result files remain in `outputs/` until removed manually.

## Docker Compose

Make sure Docker Desktop is running and the model weights exist at
`models/yolo11n.pt`. From the repository root, use the current host user's IDs
so the container can write outputs to the mounted directories:

```bash
export TRACKLAB_UID="$(id -u)"
export TRACKLAB_GID="$(id -g)"
docker compose -f docker/docker-compose.yml up --build
```

When the service is healthy, open <http://127.0.0.1:8000/> or
<http://127.0.0.1:8000/docs>.

Persistent host directories:

- `data/`: uploaded videos and the SQLite database.
- `outputs/`: generated MP4 and JSON artifacts.
- `models/`: read-only model weights inside the container.

Stop the service with:

```bash
docker compose -f docker/docker-compose.yml down
```

The container uses CPU inference. Apple MPS acceleration is intended for local
Python execution on macOS and is not available inside this Linux container.

## Tests and code quality

Install development dependencies, then run:

```bash
python -m pytest -q
ruff check src/tracklab scripts tests
ruff format --check src/tracklab scripts tests
```

GitHub Actions runs the lint, format, and test checks on pushes and pull requests.

## Known limitations

This is a portfolio and demonstration project, not a hardened public
multi-user service. Jobs currently run through FastAPI background tasks rather
than a durable worker queue; work interrupted by a process or container restart
is not automatically resumed. Authentication, quotas, and production-grade
observability are not included.

## License

TrackLab source code is released under AGPL-3.0-only. Third-party packages,
model weights, and datasets can carry their own terms.
