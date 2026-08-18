"""Contrat des features — exigence cabinet : nom, version, description, bornes valides, propriétaire, SLA.

Chaque feature dispose d'un fichier dédié dans ``features/definitions/``. Le calcul reste
centralisé dans ``features/builder.py`` (aucune formule dupliquée) ; le contrat versionne le
« quoi » sans dupliquer le « comment ».
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureContract:
    """Métadonnées contractuelles d'une feature.

    Attributes:
        name: identifiant unique de la feature (colonne).
        version: version sémantique — doit être incrémentée si la formule change (non-régression).
        family: famille d'appartenance (profile_income, savings, history, context).
        description: définition métier de la feature.
        bounds: bornes valides (min, max) ; ``None`` = non bornée de ce côté.
        owner: responsable (P1=features/modèles, P2=API/services, P3=infra).
        sla: délai/condition de disponibilité de la feature.
    """

    name: str
    version: str
    family: str
    description: str
    bounds: tuple[float | None, float | None]
    owner: str
    sla: str

    def validate_bounds(self, value: float) -> bool:
        """Vérifie qu'une valeur respecte les bornes valides du contrat."""
        lo, hi = self.bounds
        return not ((lo is not None and value < lo) or (hi is not None and value > hi))
