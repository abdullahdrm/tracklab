"""FastAPI application factory for TrackLab."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session, sessionmaker

from tracklab.api.database import (
    create_database_engine,
    create_session_factory,
)
from tracklab.api.models import Base
from tracklab.api.routes import router
from tracklab.api.service import JobManager
from tracklab.api.settings import ApiSettings
from tracklab.config import PipelineConfig
from tracklab.pipeline import VideoPipeline

WEB_UI_PATH = Path(__file__).resolve().parent / "static" / "index.html"


def create_app(
    settings: ApiSettings | None = None,
    pipeline_factory: Callable[[PipelineConfig], VideoPipeline] | None = None,
) -> FastAPI:
    """Build an API instance, optionally with test-specific dependencies."""

    app_settings = settings or ApiSettings.from_env()
    app_settings.upload_dir.mkdir(parents=True, exist_ok=True)
    app_settings.output_dir.mkdir(parents=True, exist_ok=True)

    engine = create_database_engine(app_settings.database_url)
    Base.metadata.create_all(bind=engine)
    session_factory: sessionmaker[Session] = create_session_factory(engine)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(
        title="TrackLab API",
        description="Video object tracking and analytics service",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.settings = app_settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.job_manager = JobManager(
        settings=app_settings,
        session_factory=session_factory,
        pipeline_factory=pipeline_factory,
    )
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        """Serve the browser-based video upload interface."""

        return FileResponse(WEB_UI_PATH, media_type="text/html")

    @app.get("/health", tags=["health"])
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app
