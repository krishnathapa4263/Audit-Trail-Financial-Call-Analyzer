"""
PostgreSQL connection pool management.
Checks/enables the pgvector extension on startup, per project spec.
"""
import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from config.settings import settings

logger = logging.getLogger(__name__)

# Engine is created once and reused (connection pool).
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,   # avoids stale-connection errors after idle periods
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def ensure_pgvector_extension() -> None:
    """
    Idempotently ensures the pgvector extension is enabled on the target database.
    Call this once at application startup, before any vector columns are used.
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
        version = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        ).scalar()
        logger.info(f"pgvector extension active (version {version})")


@contextmanager
def get_session():
    """
    Session-per-unit-of-work context manager.
    Usage:
        with get_session() as session:
            session.add(obj)
    """
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection() -> bool:
    """Simple liveness check — returns True if the DB is reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False
