"""Model Card — documentation de gouvernance du modèle (Semaine 4).

Fichier exigé par le cabinet (retour.txt) : ``src/models/model_card.py``. Une Model Card
(référence : Mitchell et al., 2019) documente *intenté*, *mesure*, *données*, *biais*,
*limites* et *surveillance* — conditions d'une utilisation responsable en scoring de crédit
(CIF Digital Platform §65 / directives de gouvernance de l'IA).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ModelCardMetadata:
    """Métadonnées obligatoires d'une Model Card (validation stricte)."""

    model_name: str
    version: str
    model_type: str
    algorithm: str
    owner: str
    contact: str
    intended_use: str
    target_population: str
    decision_thresholds: dict[str, float]
    training_metrics: dict[str, float]
    training_date: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    features: list[str] = field(default_factory=list)
    calibration_method: str = "isotonic"
    monitoring_thresholds: dict[str, float] = field(
        default_factory=lambda: {"roc_auc_min": 0.65, "ece_max": 0.10, "psi_max": 0.25}
    )
    limitations: list[str] = field(default_factory=list)
    fairness_notes: str = ""
    data_sources: list[str] = field(default_factory=list)


class ModelCardRequirementError(ValueError):
    """Levée quand une métadonnée obligatoire manque (qualité bloquante)."""


_REQUIRED = (
    "model_name",
    "version",
    "model_type",
    "algorithm",
    "owner",
    "contact",
    "intended_use",
    "target_population",
    "decision_thresholds",
    "training_metrics",
)


class ModelCard:
    """Documentation structurée d'un modèle enregistré dans le registry."""

    def __init__(self, metadata: ModelCardMetadata) -> None:
        self.metadata = metadata
        self.validate()

    def validate(self) -> None:
        """Vérifie les champs obligatoires : tout défaut bloque la génération de la carte."""
        for field_name in _REQUIRED:
            value = getattr(self.metadata, field_name)
            if value in (None, "") or (isinstance(value, dict) and not value):
                raise ModelCardRequirementError(f"Model Card incomplète : champ obligatoire manquant « {field_name} »")

    @property
    def name_version(self) -> str:
        return f"{self.metadata.model_name}:{self.metadata.version}"

    def to_dict(self) -> dict[str, Any]:
        return {"name_version": self.name_version, **asdict(self.metadata)}

    def to_json(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def to_markdown(self, path: str | Path) -> None:
        m = self.metadata
        lines = [
            f"# Model Card — {m.model_name}",
            f"**Version** : {m.version} | **Type** : {m.model_type} | **Algorithme** : {m.algorithm}",
            f"**Responsable** : {m.owner} ({m.contact})",
            "",
            "## Usage prévu",
            m.intended_use,
            "",
            "## Population cible",
            m.target_population,
            "",
            "## Métriques d'entraînement",
            "| Métrique | Valeur |",
            "|---|---|",
        ]
        for k, v in sorted(m.training_metrics.items()):
            lines.append(f"| {k} | {v:.4f} |")
        lines += [
            "",
            "## Seuils de décision (Policy)",
            "| Seuil | Valeur |",
            "|---|---|",
        ]
        for k, v in sorted(m.decision_thresholds.items()):
            lines.append(f"| {k} | {v:.4f} |")
        lines += [
            "",
            "## Surveillance (Monitoring)",
            "| Alerte | Valeur |",
            "|---|---|",
        ]
        for k, v in sorted(m.monitoring_thresholds.items()):
            lines.append(f"| {k} | {v} |")
        lines += ["", "## Limites", *[f"- {lim}" for lim in m.limitations]]
        if m.fairness_notes:
            lines += ["", "## Considérations d'équité", m.fairness_notes]
        if m.data_sources:
            lines += ["", "## Sources de données", *[f"- {s}" for s in m.data_sources]]
        lines += ["", f"*Générée le {m.training_date}.*", ""]

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines), encoding="utf-8")


def build_model_card(
    *,
    model_name: str,
    version: str,
    algorithm: str,
    owner: str,
    contact: str,
    intended_use: str,
    target_population: str,
    decision_thresholds: dict[str, float],
    training_metrics: dict[str, float],
    features: list[str] | None = None,
    limitations: list[str] | None = None,
    fairness_notes: str = "",
    data_sources: list[str] | None = None,
) -> ModelCard:
    """Fabrique une Model Card en imposant les métadonnées de gouvernance."""
    return ModelCard(
        ModelCardMetadata(
            model_name=model_name,
            version=version,
            model_type="classification_binaire",
            algorithm=algorithm,
            owner=owner,
            contact=contact,
            intended_use=intended_use,
            target_population=target_population,
            decision_thresholds=decision_thresholds,
            training_metrics=training_metrics,
            features=features or [],
            limitations=limitations or [],
            fairness_notes=fairness_notes,
            data_sources=data_sources or [],
        )
    )
