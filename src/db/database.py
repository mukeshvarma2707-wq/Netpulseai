"""
src/db/database.py

Database connection setup. Points at a local SQLite file
(data/raw/balancegrid.db) - zero setup required.

TO MIGRATE TO AZURE SQL LATER: change only the DATABASE_URL line below.
No other file needs to change.
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "balancegrid.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()