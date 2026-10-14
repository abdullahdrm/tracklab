"""Database engine and session factory creation."""

from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


def create_database_engine(database_url: str) -> Engine:
    """Create an engine and prepare the parent directory for SQLite files."""

    kwargs: dict[str, object] = {"pool_pre_ping": True}

    if database_url.startswith("sqlite:"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30.0}
        if database_url.startswith("sqlite:///"):
            sqlite_path = database_url[len("sqlite:///") :].split("?")[0]
            if sqlite_path not in {":memory:", ""} and not sqlite_path.startswith(
                "file:"
            ):
                Path(sqlite_path).expanduser().parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

    engine = create_engine(database_url, **kwargs)

    if database_url.startswith("sqlite:"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, _):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create independent SQLAlchemy sessions for requests and jobs."""

    return sessionmaker(
        bind=engine,
        autoflush=False,
        expire_on_commit=False,
    )
