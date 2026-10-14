"""HTTP routes for video jobs and their artifacts."""

from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse

from tracklab.api.models import Job, utc_now
from tracklab.api.schemas import JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _job_response(job: Job) -> JobResponse:
    """Map a database record to its public API representation."""

    video_exists = bool(job.output_video_path and Path(job.output_video_path).is_file())
    json_exists = bool(job.output_json_path and Path(job.output_json_path).is_file())

    return JobResponse(
        id=job.id,
        original_filename=job.original_filename,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        summary=job.summary,
        error=job.error,
        video_available=video_exists,
        json_available=json_exists,
    )


def _find_job(request: Request, job_id: str) -> Job:
    """Load a job or raise HTTP 404."""

    session_factory = request.app.state.session_factory
    with session_factory() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


def _ready_artifact_path(job: Job, field: str, label: str) -> Path:
    """Require a completed job and an existing requested output file."""

    if job.status != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not done; current status is '{job.status}'",
        )

    raw_path = getattr(job, field)
    if not raw_path:
        raise HTTPException(
            status_code=404,
            detail=f"No {label} output was produced for this job",
        )

    path = Path(raw_path)
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"The {label} output file is no longer available",
        )

    return path


@router.post("", response_model=JobResponse, status_code=202)
async def create_job(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File()],
    response: Response,
) -> JobResponse:
    """Upload a video, persist a queued job, and schedule processing."""

    settings = request.app.state.settings
    safe_filename = Path((file.filename or "").replace("\\", "/")).name
    suffix = Path(safe_filename).suffix.lower()

    if suffix not in settings.allowed_extensions:
        await file.close()
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported video extension. Allowed extensions: "
                + ", ".join(settings.allowed_extensions)
            ),
        )

    job_id = str(uuid4())
    upload_path = settings.upload_dir / f"{job_id}{suffix}"
    total_bytes = 0

    try:
        settings.upload_dir.mkdir(parents=True, exist_ok=True)

        with upload_path.open("wb") as destination:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=("Uploaded video exceeds the configured maximum size."),
                    )

                destination.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty",
            )

        job = Job(
            id=job_id,
            original_filename=safe_filename[:255],
            status="queued",
            created_at=utc_now(),
            input_path=str(upload_path.resolve()),
        )

        with request.app.state.session_factory() as session:
            session.add(job)
            session.commit()
            session.refresh(job)

    except Exception:
        upload_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    background_tasks.add_task(
        request.app.state.job_manager.process_job,
        job_id,
    )
    response.headers["Location"] = f"/jobs/{job_id}"

    return _job_response(job)


@router.get("/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request) -> JobResponse:
    """Return a job's status and summary."""

    return _job_response(_find_job(request, job_id))


@router.get("/{job_id}/result")
def download_result(job_id: str, request: Request) -> FileResponse:
    """Download the JSON result for a completed job."""

    job = _find_job(request, job_id)
    path = _ready_artifact_path(job, "output_json_path", "JSON")

    return FileResponse(
        path,
        media_type="application/json",
        filename=path.name,
    )


@router.get("/{job_id}/preview")
def preview_video(job_id: str, request: Request) -> FileResponse:
    """Stream the annotated video for in-browser playback."""

    job = _find_job(request, job_id)
    path = _ready_artifact_path(job, "output_video_path", "video")

    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )


@router.get("/{job_id}/video")
def download_video(job_id: str, request: Request) -> FileResponse:
    """Download the annotated video for a completed job."""

    job = _find_job(request, job_id)
    path = _ready_artifact_path(job, "output_video_path", "video")

    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name,
    )
