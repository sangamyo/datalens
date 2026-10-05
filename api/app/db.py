from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

# create_engine is lazy: no connection is opened until the first query,
# so importing this module does not require a running database.
engine = create_engine(get_settings().database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed afterwards."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
