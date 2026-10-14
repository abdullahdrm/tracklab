"""Environment-based settings for the TrackLab API."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """Runtime settings for uploads, output storage, and database access."""

    database_url: str = "sqlite:///./data/tracklab.sqlite3"
    upload_dir: Path = Path("data/uploads")
    output_dir: Path = Path("outputs/jobs")
    config_path: Path = Path("configs/pipeline.yaml")
    max_upload_bytes: int = 500 * 1024 * 1024
    allowed_extensions: tuple[str, ...] = (
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
    )

    def __post_init__(self) -> None:
        if self.max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be positive")

    @classmethod
    def from_env(cls) -> "ApiSettings":
        """Construct settings from environment variables."""

        return cls(
            database_url=os.getenv(
                "TRACKLAB_DATABASE_URL",
                "sqlite:///./data/tracklab.sqlite3",
            ),
            upload_dir=Path(os.getenv("TRACKLAB_UPLOAD_DIR", "data/uploads")),
            output_dir=Path(os.getenv("TRACKLAB_OUTPUT_DIR", "outputs/jobs")),
            config_path=Path(
                os.getenv(
                    "TRACKLAB_CONFIG_PATH",
                    "configs/pipeline.yaml",
                )
            ),
            max_upload_bytes=int(
                os.getenv(
                    "TRACKLAB_MAX_UPLOAD_BYTES",
                    str(500 * 1024 * 1024),
                )
            ),
        )
