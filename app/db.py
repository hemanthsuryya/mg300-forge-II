"""
Database engine and session handling.

DATABASE_URL drives everything. It defaults to a local SQLite file so the app
still starts with no database server running (./run.sh keeps working); the
Docker compose stack points it at Postgres instead.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))


def _default_url() -> str:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DATA_DIR / 'tonechaser.db'}"


DATABASE_URL = os.environ.get("DATABASE_URL") or _default_url()

_kw: dict = {"pool_pre_ping": True, "future": True}
if DATABASE_URL.startswith("sqlite"):
    # The analysis runs in worker threads and touches the same SQLite file.
    _kw["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_kw)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False,
                            class_=Session)


class Base(DeclarativeBase):
    pass


@contextmanager
def session_scope():
    """Transactional scope. Commits on success, rolls back on error."""
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def _add_missing_columns() -> None:
    """
    create_all never alters a table that already exists, so a column added to a
    model after the database was created is added here. Additive only - nothing
    is dropped or changed - which keeps it safe to run on every start.
    """
    from sqlalchemy import inspect, text

    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                ddl = f"ALTER TABLE {table.name} ADD COLUMN {col.name} " \
                      f"{col.type.compile(dialect=engine.dialect)}"
                if col.default is not None and col.default.is_scalar:
                    v = col.default.arg
                    ddl += " DEFAULT " + (("TRUE" if v else "FALSE") if isinstance(v, bool)
                                          else repr(v))
                conn.execute(text(ddl))


def init_db(retries: int = 10, delay: float = 1.5) -> None:
    """
    Create tables. Retries because in Docker the app usually wins the race
    against Postgres accepting connections, even with a healthcheck.
    """
    import time
    from . import models  # noqa: F401  (registers the mappers)

    last = None
    for attempt in range(retries):
        try:
            Base.metadata.create_all(engine)
            _add_missing_columns()
            return
        except Exception as e:  # pragma: no cover - depends on external DB
            last = e
            if attempt < retries - 1:
                time.sleep(delay)
    raise RuntimeError(f"Could not initialise the database at {DATABASE_URL}: {last}")
