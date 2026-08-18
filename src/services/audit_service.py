"""Service d'audit — journalisation de chaque décision, prédiction et changement (Semaine 3).

Exigence cabinet (retour.txt §1 « Traçabilité complète ») : chaque prédiction enregistrée
atomiquement (transaction unique) avec ID client, PD, confiance, décision, version du modèle,
timestamp et features utilisées. L'écriture dans la même transaction garantit l'atomicité :
si l'écriture échoue, la réponse API échoue aussi (aucune décision non tracée).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy.orm import Session

from services.database import (
    AuditLog,
    Customer,
    ModelVersion,
    Prediction,
    create_engine_for,
    create_session_factory,
    ensure_schema,
)


class AuditEvent(StrEnum):
    """Événements métier traçables dans audit_log."""

    TOKEN_ISSUED = "token_issued"
    LOGIN_FAILURE = "login_failure"
    PREDICTION_REQUESTED = "prediction_requested"
    CONFIG_CHANGED = "config_changed"
    MODEL_REGISTERED = "model_registered"
    MODEL_DEPLOYED = "model_deployed"


class AuditService:
    """Joumalise les événements dans PostgreSQL (SQLite autorisé pour les tests/CI)."""

    def __init__(self, url: str, *, echo: bool = False, create_schema: bool | None = None) -> None:
        self.url = url
        self._engine = create_engine_for(url, echo=echo)
        self._sessions = create_session_factory(self._engine)
        if create_schema is None:
            create_schema = url.startswith("sqlite")
        if create_schema:
            ensure_schema(self._engine)

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def record(
        self,
        event: AuditEvent | str,
        *,
        actor: str | None = None,
        customer_id: int | None = None,
        request_id: str | None = None,
        model_version: str | None = None,
        status: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Insère une entrée d'audit (committée immédiatement)."""
        with self._session() as session:
            session.add(
                AuditLog(
                    event=str(event.value) if isinstance(event, StrEnum) else event,
                    actor=actor,
                    customer_id=customer_id,
                    request_id=request_id,
                    model_version=model_version,
                    status=status,
                    payload=payload or {},
                    created_at=self._now(),
                )
            )
            session.commit()

    def record_prediction(
        self,
        *,
        actor: str | None,
        customer_id: int,
        request_id: str,
        model_version: str,
        probability: float,
        score: int,
        decision: str,
        confidence: float,
        policy_hit: str | None,
        risk_class: str | None,
        features: dict[str, float],
        factors: dict[str, float] | None,
    ) -> int:
        """Enregistre une prédiction et son événement d'audit dans une transaction unique.

        Returns:
            Identifiant de la prédiction créée (customer upserté dans ``customers``).
        """
        with self._session() as session:
            customer = self._upsert_customer(session, customer_id)
            prediction = Prediction(
                customer_id=customer.id,
                request_id=request_id,
                model_version=model_version,
                probability=probability,
                score=score,
                decision=decision,
                confidence=confidence,
                policy_hit=policy_hit,
                risk_class=risk_class,
                features=features,
                factors=factors or {},
                actor=actor,
                timestamp=self._now(),
            )
            session.add(prediction)
            session.add(
                AuditLog(
                    event=AuditEvent.PREDICTION_REQUESTED.value,
                    actor=actor,
                    customer_id=customer.id,
                    request_id=request_id,
                    model_version=model_version,
                    status="ok",
                    payload={"decision": decision, "probability": round(probability, 6)},
                    created_at=self._now(),
                )
            )
            session.flush()
            prediction_id = prediction.id
            session.commit()
            return int(prediction_id or 0)

    def upsert_model_version(
        self,
        *,
        model_name: str,
        version: str,
        run_id: str | None = None,
        stage: str = "Staging",
        metrics: dict[str, float] | None = None,
        notes: str | None = None,
    ) -> None:
        """Enregistre/maj une version de modèle dans model_versions (miroir du registry MLflow)."""
        with self._session() as session:
            existing = session.query(ModelVersion).filter_by(model_name=model_name, version=version).first()
            if existing is not None:
                existing.stage = stage
                existing.metrics = metrics or {}
                existing.notes = notes or existing.notes
            else:
                session.add(
                    ModelVersion(
                        model_name=model_name,
                        version=version,
                        run_id=run_id,
                        stage=stage,
                        metrics=metrics or {},
                        notes=notes,
                        created_at=self._now(),
                    )
                )
            session.commit()

    def count_predictions(self) -> int:
        with self._session() as session:
            return int(session.query(Prediction).count())

    def count_events(self) -> int:
        with self._session() as session:
            return int(session.query(AuditLog).count())

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._session() as session:
            rows = session.query(AuditLog).order_by(AuditLog.id.desc()).limit(limit).all()
            return [
                {
                    "id": row.id,
                    "event": row.event,
                    "actor": row.actor,
                    "customer_id": row.customer_id,
                    "request_id": row.request_id,
                    "model_version": row.model_version,
                    "status": row.status,
                    "payload": row.payload or {},
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

    def _upsert_customer(self, session: Session, cif_id: int) -> Customer:
        existing = session.query(Customer).filter_by(cif_id=str(cif_id)).first()
        if existing is not None:
            return existing
        customer = Customer(cif_id=str(cif_id), n_past_loans=0)
        session.add(customer)
        session.flush()
        return customer

    def _session(self) -> Session:
        return self._sessions()
