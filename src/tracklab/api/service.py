"""Job lifecycle management and pipeline execution."""

import logging
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from tracklab.api.models import Job, utc_now
from tracklab.api.settings import ApiSettings
from tracklab.config import PipelineConfig, load_config
from tracklab.pipeline import VideoPipeline

logger = logging.getLogger(__name__)

PipelineFactory = Callable[[PipelineConfig], VideoPipeline]


class JobManager:
    """Execute video jobs and persist their state transitions."""

    def __init__(
        self,
        settings: ApiSettings,
        session_factory: sessionmaker[Session],
        pipeline_factory: PipelineFactory | None = None,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.pipeline_factory = pipeline_factory or (
            lambda config: VideoPipeline(config)
        )

    def process_job(self, job_id: str) -> None:
        """Run a queued job and persist success or failure."""

        try:
            UUID(job_id)
        except ValueError:
            logger.error("Refusing to process invalid job ID: %s", job_id)
            return

        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None or job.status != "queued":
                return

            job.status = "running"
            job.started_at = utc_now()
            input_path = Path(job.input_path)
            session.commit()

        try:
            config = load_config(self.settings.config_path)
            pipeline = self.pipeline_factory(config)
            output_dir = self.settings.output_dir / job_id
            result = pipeline.run(input_path, output_dir=output_dir)
        except Exception as exc:
            logger.exception("Video job %s failed", job_id)

            with self.session_factory() as session:
                job = session.get(Job, job_id)
                if job is not None:
                    job.status = "failed"
                    job.completed_at = utc_now()
                    job.error = f"{type(exc).__name__}: {exc}"[:2000]
                    session.commit()
            return

        with self.session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                logger.error("Job %s disappeared before completion", job_id)
                return

            job.status = "done"
            job.completed_at = utc_now()
            job.summary = result.video_result.summary
            job.output_video_path = (
                str(result.video_path.resolve())
                if result.video_path is not None
                else None
            )
            job.output_json_path = (
                str(result.json_path.resolve())
                if result.json_path is not None
                else None
            )
            job.output_mot_path = (
                str(result.mot_path.resolve()) if result.mot_path is not None else None
            )
            session.commit()
