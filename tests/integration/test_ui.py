"""Integration tests for the TrackLab browser UI."""

from pathlib import Path

from fastapi.testclient import TestClient

from tracklab.api.application import create_app
from tracklab.api.settings import ApiSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_home_page_serves_video_upload_interface(tmp_path: Path) -> None:
    settings = ApiSettings(
        database_url=f"sqlite:///{tmp_path / 'tracklab-ui-test.sqlite3'}",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        config_path=PROJECT_ROOT / "configs" / "pipeline.yaml",
    )
    app = create_app(settings)

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "TrackLab" in response.text
    assert 'id="videoFile"' in response.text
    assert "Analizi başlat" in response.text
    assert "En fazla 10 dakika" in response.text
    assert "maxDurationSeconds" in response.text
    assert "function readVideoDuration(file)" in response.text
