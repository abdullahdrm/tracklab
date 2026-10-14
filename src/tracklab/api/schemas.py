"""Pydantic response schemas for the TrackLab HTTP API."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class JobResponse(BaseModel):
    """Public representation of a video-processing job."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    original_filename: str
    status: Literal["queued", "running", "done", "failed"]
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    summary: dict[str, Any] | None
    error: str | None
    video_available: bool
    json_available: bool
