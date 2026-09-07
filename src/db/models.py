"""
src/db/models.py

Database schema for BalanceGrid. Three tables:

1. DiagnosedCase - one row per (cell, hour) flagged by the forecasting +
   threshold pipeline and classified by the diagnosis agent as ROUTINE or
   ANOMALOUS. This is the output of diagnosis_agent.py + solver.py's
   coverage results, joined together.
2. ReallocationSource - one row per real neighbor contributing capacity to
   a routine case. A single DiagnosedCase can have MULTIPLE sources (the
   solver often pools capacity from several neighbors at once), so this is
   a separate table with a foreign key back to DiagnosedCase, not columns
   crammed onto the case itself.
3. Report - the plain-language report generated for a case, if you've
   generated one. Separate from DiagnosedCase so a case can exist without
   a report yet (not yet explained) or with one (already explained).

DESIGN DECISION (same as CounterFlag, flagging explicitly): SQLite for
local development via SQLAlchemy - zero setup, and migrating to Azure SQL
later is a one-line connection-string change in database.py, not a
rewrite of this file or any query code.
"""

from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class DiagnosedCase(Base):
    __tablename__ = "diagnosed_cases"

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