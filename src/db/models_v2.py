"""
src/db/models_v2.py  (promotion v2, stage 2)

Schema of the v2 database (balancegrid_v2.db). It has its OWN declarative Base, so
Base.metadata.create_all() from this module only ever creates v2 tables on the v2 engine and
shares nothing with src/db/models.py.

Tables:
1. DiagnosedCase      - as v1, plus horizon, forecast_kind, model_version and segment. +1h
                        cases from diagnosis_agent_v2 (ROUTINE and ANOMALOUS), with V0 solver
                        coverage for ROUTINE cases. naive_forecast keeps v1's column name and
                        holds the model forecast, as in v1.
2. ReallocationSource - as v1 (solver_v2 moves).
3. Report             - as v1.
4. WatchFlag          - ADVISORY watch status (typical cells, margin x threshold < forecast <=
                        threshold). Never a DiagnosedCase, never counted as resolved, never
                        given to the solver.
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class DiagnosedCase(Base):
    __tablename__ = "diagnosed_cases"
    __table_args__ = (Index("ix_v2_cases_dt_class", "target_datetime", "classification"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    cell_id = Column(Integer, nullable=False, index=True)
    target_datetime = Column(String(30), nullable=False, index=True)

    naive_forecast = Column(Float)
    congestion_threshold = Column(Float)
    classification = Column(String(20))
    reason = Column(Text)

    deficit = Column(Float, nullable=True)
    covered = Column(Float, nullable=True)
    fully_resolved = Column(Boolean, nullable=True)

    horizon = Column(Integer, nullable=True)
    forecast_kind = Column(String(20), nullable=True)
    model_version = Column(String(40), nullable=True)
    segment = Column(String(10), nullable=True)

    sources = relationship("ReallocationSource", back_populates="case", cascade="all, delete-orphan")
    report = relationship("Report", back_populates="case", uselist=False, cascade="all, delete-orphan")


class ReallocationSource(Base):
    __tablename__ = "reallocation_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("diagnosed_cases.id"), nullable=False, index=True)
    source_cell_id = Column(Integer, nullable=False)
    amount_moved = Column(Float, nullable=False)

    case = relationship("DiagnosedCase", back_populates="sources")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey("diagnosed_cases.id"), nullable=False, unique=True)

    report_text = Column(Text)
    generated_at = Column(DateTime, default=datetime.utcnow)

    case = relationship("DiagnosedCase", back_populates="report")


class WatchFlag(Base):
    __tablename__ = "watch_flags"
    __table_args__ = (Index("ix_v2_watch_dt_h", "target_datetime", "horizon"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    cell_id = Column(Integer, nullable=False, index=True)
    target_datetime = Column(String(30), nullable=False)
    horizon = Column(Integer, nullable=False)
    forecast = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    margin = Column(Float, nullable=False)
    model_version = Column(String(40), nullable=False)
    forecast_kind = Column(String(20), nullable=False)
