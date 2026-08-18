"""Couche d'accès aux données — PostgreSQL 16 (audit trail + registry), exigence cabinet Semaine 3.

Le schéma (customers, predictions, audit_log, model_versions) est défini ici en SQLAlchemy 2.0
et versionné par Alembic dans ``migrations/``. Les types JSON sont portables : ``JSON`` sur
SQLite (tests/CI) et ``JSONB`` natif sur PostgreSQL 16 (production), via ``with_variant``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

PortableJSON = JSON().with_variant(JSONB, "postgresql")

# BIGINT sur PostgreSQL 16 ; INTEGER auto-incrémentable sur SQLite (tests/CI).
BigIntPK = BigInteger().with_variant(Integer, "sqlite")


class Base(DeclarativeBase):
    """Base SQLAlchemy déclarative commune au schéma applicatif."""


class Customer(Base):
    """Référentiel des clients interrogés (dict: few hundred CFDs)."""

    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    cif_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    n_past_loans: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    risk_segment: Mapped[str] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Prediction(Base):
    """Chaque prédiction est journalisée atomiquement (PD, confiance, décision, version, features)."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("customers.id"), nullable=False, index=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    decision: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    policy_hit: Mapped[str] = mapped_column(String(255), nullable=True)
    risk_class: Mapped[str] = mapped_column(String(32), nullable=True)
    features: Mapped[dict[str, Any]] = mapped_column(PortableJSON, nullable=False)
    factors: Mapped[dict[str, float]] = mapped_column(PortableJSON, nullable=True)
    actor: Mapped[str | None] = mapped_column(String(64), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class AuditLog(Base):
    """Journal d'audit : chaque événement métier est tracé (traçabilité totale, principe §1)."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    event: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    actor: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    customer_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload: Mapped[dict[str, Any]] = mapped_column(PortableJSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)


class ModelVersion(Base):
    """Registry local des versions de modèle (miroir léger du Model Registry MLflow)."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(BigIntPK, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    stage: Mapped[str] = mapped_column(String(32), nullable=False, default="Staging")
    metrics: Mapped[dict[str, float]] = mapped_column(PortableJSON, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("model_name", "version"),)


def create_engine_for(url: str, *, echo: bool = False) -> Engine:
    """Fabrique l'engine SQLAlchemy. En échec (base indisponible) l'erreur remonte (fail-fast)."""
    return create_engine(url, echo=echo, pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Fabrique une fabrique de sessions liées à l'engine."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def ensure_schema(engine: Engine) -> None:
    """Crée les tables si absentes (utile pour SQLite en tests ; en prod : ``alembic upgrade head``)."""
    Base.metadata.create_all(bind=engine)
