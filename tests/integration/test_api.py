"""Integration tests for the TrackLab API."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tracklab.api.application import create_app
from tracklab.api.settings import ApiSettings
from tracklab.io.video import FrameWriter
from tracklab.pipeline import VideoPipeline
from tracklab.schemas import TrackedObject

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeTracker:
    """Predictable tracker for API integration tests."""

    def track_frame(self, frame: np.ndarray) -> list[TrackedObject]:
        return [
            TrackedObject(
                track_id=7,
                x1=5.0,
                y1=5.0,
                x2=40.0,
                y2=35.0,
                score=0.92,
                class_id=2,
            )
        ]


def create_video(path: Path, frame_count: int = 4) -> bytes:
    """Create a small video and return its bytes for multipart upload."""

    with FrameWriter(path, fps=10.0, size=(64, 48)) as writer:
        for index in range(frame_count):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, 1] = index * 25
            writer.write(frame)

    return path.read_bytes()


def _fake_transcode(source: str | Path, destination: str | Path) -> Path:
    """Copy a video for integration tests instead of invoking FFmpeg."""

    output_path = Path(destination)
    shutil.copyfile(source, output_path)
    return output_path


def make_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch | None = None,
    *,
    max_video_duration_seconds: float = 600.0,
):
    """Build an isolated app with a temporary database and fake tracker."""

    if monkeypatch is not None:
        monkeypatch.setattr("tracklab.pipeline.transcode_video_h264", _fake_transcode)

    settings = ApiSettings(
        database_url=f"sqlite:///{tmp_path / 'tracklab-test.sqlite3'}",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        config_path=PROJECT_ROOT / "configs" / "pipeline.yaml",
        max_upload_bytes=2 * 1024 * 1024,
        max_video_duration_seconds=max_video_duration_seconds,
    )

    return create_app(
        settings,
        pipeline_factory=lambda config: VideoPipeline(
            config,
            tracker=FakeTracker(),
        ),
    )


def test_upload_job_processes_video_and_persists_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_bytes = create_video(tmp_path / "input.mp4")
    app = make_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        upload_response = client.post(
            "/jobs",
            files={
                "file": (
                    "traffic.mp4",
                    video_bytes,
                    "video/mp4",
                )
            },
        )

        assert upload_response.status_code == 202
        upload_data = upload_response.json()
        job_id = upload_data["id"]
        assert upload_data["original_filename"] == "traffic.mp4"

        status_response = client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200

        status = status_response.json()
        assert status["status"] == "done"
        assert status["error"] is None
        assert status["video_available"] is True
        assert status["json_available"] is True
        assert status["summary"]["processed_frames"] == 4
        assert status["summary"]["total_detections"] == 4
        assert status["summary"]["unique_tracks"] == 1

        json_response = client.get(f"/jobs/{job_id}/result")
        assert json_response.status_code == 200
        payload = json_response.json()
        assert payload["summary"]["processed_frames"] == 4
        assert len(payload["frames"]) == 4
        assert payload["frames"][0]["objects"][0]["track_id"] == 7

        video_response = client.get(f"/jobs/{job_id}/video")
        assert video_response.status_code == 200
        assert video_response.headers["content-type"].startswith("video/mp4")
        assert len(video_response.content) > 0

        preview_response = client.get(f"/jobs/{job_id}/preview")
        assert preview_response.status_code == 200
        assert preview_response.headers["content-type"].startswith("video/mp4")
        assert preview_response.headers["content-disposition"].startswith("inline;")
        assert len(preview_response.content) > 0

        missing_response = client.get("/jobs/not-a-real-job")
        assert missing_response.status_code == 404

        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json() == {"status": "ok"}


def test_rejects_unsupported_file_extension(tmp_path: Path) -> None:
    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            files={
                "file": (
                    "notes.txt",
                    b"not a video",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 415


def test_rejects_upload_larger_than_configured_limit(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            files={
                "file": (
                    "large.mp4",
                    b"x" * (2 * 1024 * 1024 + 1),
                    "video/mp4",
                )
            },
        )

    assert response.status_code == 413


def test_empty_upload_is_rejected_without_leaving_a_file(
    tmp_path: Path,
) -> None:
    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            files={"file": ("empty.mp4", b"", "video/mp4")},
        )

    assert response.status_code == 400
    assert list((tmp_path / "uploads").iterdir()) == []


def test_job_status_persists_after_app_recreation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_bytes = create_video(tmp_path / "input.mp4")
    app = make_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            files={"file": ("traffic.mp4", video_bytes, "video/mp4")},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]
        assert client.get(f"/jobs/{job_id}").json()["status"] == "done"

    # Re-create the application against the same SQLite database.
    reopened_app = make_app(tmp_path, monkeypatch)

    with TestClient(reopened_app) as client:
        response = client.get(f"/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["status"] == "done"
    assert response.json()["json_available"] is True


def test_processing_failure_is_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingPipeline:
        def run(
            self,
            source: str | Path,
            output_dir: str | Path | None = None,
        ) -> None:
            raise RuntimeError("synthetic processing failure")

    video_bytes = create_video(tmp_path / "input.mp4")
    app = make_app(tmp_path, monkeypatch)
    app.state.job_manager.pipeline_factory = lambda _config: FailingPipeline()

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            files={"file": ("traffic.mp4", video_bytes, "video/mp4")},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]

        status_response = client.get(f"/jobs/{job_id}")
        result_response = client.get(f"/jobs/{job_id}/result")

    status = status_response.json()
    assert status_response.status_code == 200
    assert status["status"] == "failed"
    assert status["error"] == "RuntimeError: synthetic processing failure"
    assert status["completed_at"] is not None
    assert result_response.status_code == 409


def test_rejects_corrupt_video_upload(tmp_path: Path) -> None:
    app = make_app(tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            files={"file": ("corrupt.mp4", b"not a real video", "video/mp4")},
        )

    assert response.status_code == 400
    assert "valid, readable video" in response.json()["detail"]
    assert list((tmp_path / "uploads").iterdir()) == []


def test_rejects_video_exceeding_duration_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_bytes = create_video(tmp_path / "long-enough.mp4")
    app = make_app(
        tmp_path,
        monkeypatch,
        max_video_duration_seconds=0.2,
    )

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            files={"file": ("long-enough.mp4", video_bytes, "video/mp4")},
        )

    assert response.status_code == 422
    assert "maximum allowed duration" in response.json()["detail"]
    assert list((tmp_path / "uploads").iterdir()) == []


def test_api_settings_rejects_invalid_duration_limit() -> None:
    with pytest.raises(ValueError, match="max_video_duration_seconds"):
        ApiSettings(max_video_duration_seconds=0)

    with pytest.raises(ValueError, match="max_video_duration_seconds"):
        ApiSettings(max_video_duration_seconds=float("inf"))


def test_stale_queued_and_running_jobs_are_marked_failed_on_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracklab.api.models import Job, utc_now

    app1 = make_app(tmp_path, monkeypatch)
    now = utc_now()

    queued_id = "11111111-1111-1111-1111-111111111111"
    running_id = "22222222-2222-2222-2222-222222222222"
    done_id = "33333333-3333-3333-3333-333333333333"

    with app1.state.session_factory() as session:
        session.add_all(
            [
                Job(
                    id=queued_id,
                    original_filename="queued.mp4",
                    status="queued",
                    created_at=now,
                    input_path=str(tmp_path / "queued.mp4"),
                ),
                Job(
                    id=running_id,
                    original_filename="running.mp4",
                    status="running",
                    created_at=now,
                    started_at=now,
                    input_path=str(tmp_path / "running.mp4"),
                ),
                Job(
                    id=done_id,
                    original_filename="done.mp4",
                    status="done",
                    created_at=now,
                    started_at=now,
                    completed_at=now,
                    input_path=str(tmp_path / "done.mp4"),
                ),
            ]
        )
        session.commit()

    # Simulate an application restart using the same SQLite database.
    app1.state.engine.dispose()
    app2 = make_app(tmp_path, monkeypatch)

    with TestClient(app2) as client:
        queued = client.get(f"/jobs/{queued_id}").json()
        running = client.get(f"/jobs/{running_id}").json()
        done = client.get(f"/jobs/{done_id}").json()

    for recovered in (queued, running):
        assert recovered["status"] == "failed"
        assert recovered["completed_at"] is not None
        assert "yeniden başlatıldığında" in recovered["error"]

    assert done["status"] == "done"
    assert done["error"] is None


def test_upload_file_is_cleaned_up_after_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_bytes = create_video(tmp_path / "cleanup_test.mp4")
    app = make_app(tmp_path, monkeypatch)

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            files={"file": ("cleanup_test.mp4", video_bytes, "video/mp4")},
        )
        assert response.status_code == 202

    assert list((tmp_path / "uploads").iterdir()) == []


def test_error_sanitization_removes_internal_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_bytes = create_video(tmp_path / "failing.mp4")

    class PathLeakingPipeline:
        def run(self, input_path: Path, output_dir: Path):
            raise RuntimeError(
                "Secret error at /Users/secret/path/to/hidden.pt with /app/data"
            )

    settings = ApiSettings(
        database_url=f"sqlite:///{tmp_path / 'tracklab-sanitize-test.sqlite3'}",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        config_path=PROJECT_ROOT / "configs" / "pipeline.yaml",
    )
    app = create_app(
        settings,
        pipeline_factory=lambda config: PathLeakingPipeline(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            files={"file": ("failing.mp4", video_bytes, "video/mp4")},
        )
        assert response.status_code == 202
        job_id = response.json()["id"]

        status = client.get(f"/jobs/{job_id}").json()
        assert status["status"] == "failed"
        assert "/Users/secret" not in status["error"]
        assert "<path>" in status["error"]
        assert list((tmp_path / "uploads").iterdir()) == []


def test_ready_artifact_path_rejects_path_outside_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tracklab.api.models import Job, utc_now

    app = make_app(tmp_path, monkeypatch)
    with app.state.session_factory() as session:
        job = Job(
            id="22222222-2222-2222-2222-222222222222",
            original_filename="evil.mp4",
            status="done",
            created_at=utc_now(),
            input_path=str(tmp_path / "evil.mp4"),
            output_json_path="/etc/passwd",
        )
        session.add(job)
        session.commit()

    with TestClient(app) as client:
        response = client.get("/jobs/22222222-2222-2222-2222-222222222222/result")
        assert response.status_code == 403
