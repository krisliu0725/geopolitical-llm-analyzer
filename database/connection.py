"""Database engine, session factory, and initialization."""

import os

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session

from config import DATABASE_PATH

engine = create_engine(
    f"sqlite:///{DATABASE_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def init_db():
    """Create all tables and the data directory if they don't exist.

    Drops tables from the previous schema version (responses, tr_scores, bias_scores)
    since they are no longer used.
    """
    from .models import Base

    os.makedirs(DATABASE_PATH.parent, exist_ok=True)

    # Clean up tables from old schemas
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS tr_scores"))
        conn.execute(text("DROP TABLE IF EXISTS bias_scores"))
        conn.execute(text("DROP TABLE IF EXISTS responses"))
        # V2.0 schema migration: old analyses had tr_score + d1-d4 only
        conn.execute(text("DROP TABLE IF EXISTS analyses"))
        conn.commit()

    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    """Return a new SQLAlchemy session."""
    return SessionLocal()
