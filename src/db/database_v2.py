"""
src/db/database_v2.py  (promotion v2, stage 2)

Connection to the v2 database ONLY: data/processed/v2/balancegrid_v2.db. Selected by the API
when BALANCEGRID_DB=v2; the default (no variable) keeps using src/db/database.py and the v1 file.

Safeguards: the v2 path is fixed in code, must not be the v1 file and must not be under
data/raw, and init_db_v2() creates only the tables of models_v2.Base (a separate metadata).
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models_v2 import Base

ROOT = Path(__file__).resolve().parents[2]
V1_DB_PATH = (ROOT / "data" / "raw" / "balancegrid.db").resolve()
DB_PATH = (ROOT / "data" / "processed" / "v2" / "balancegrid_v2.db").resolve()

if DB_PATH == V1_DB_PATH or (ROOT / "data" / "raw").resolve() in DB_PATH.parents:
    raise RuntimeError(f"v2 database path {DB_PATH} must not be the v1 file or under data/raw")

DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
META_PATH = DB_PATH.parent / "v2_metadata.json"


def init_db():
    assert Path(engine.url.database).resolve() == DB_PATH
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
