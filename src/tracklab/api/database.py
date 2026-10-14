"""Database engine and session factory creation."""

from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def create_database_engine(database_url: str) -> Engine:
    """Create an engine and prepare the parent directory for SQLite files."""

    kwargs: dict[str, object] = {"pool_pre_ping": True}

    if database_url.startswith("sqlite:"):
        kwargs["connect_args"] = {"check_same_thread": False}
        sqlite_path = database_url.removeprefix("sqlite:///")

        if sqlite_path not in {":memory:", ""} and not sqlite_path.startswith("file:"):
            Path(sqlite_path).expanduser().parent.mkdir(
                parents=True,
                exist_ok=True,
            )

    return create_engine(database_url, **kwargs)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create independent SQLAlchemy sessions for requests and jobs."""

    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
