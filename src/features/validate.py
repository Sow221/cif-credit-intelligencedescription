"""Garde anti-leakage — contrôle DÉFENSIF de la pipeline.

Valide qu'aucune variable interdite ou future n'entre dans les features
d'entraînement ni dans les requêtes de scoring. Si une variable qui encode la
cible (ex : ``p_default_true``) ou une variable post-décision réapparaît, la
validation échoue — l'entraînement s'arrête avant toute fuite.

C'est le cœur méthodologique du protocole CIF (rejeté depuis le dépôt de
référence `cifci` — see docs/). Aucune feature de fuite ne doit survivre ici.
"""

from __future__ import annotations

import pandas as pd


class LeakageError(Exception):
    """Levée lorsqu'une variable interdite ou la cible est détectée."""


# Variables qui encodent directement la cible de défaut dans le générateur
# synthétique — interdites FORMELLEMENT dans toute feature / requête.
# NB: `is_default` (la cible) n'est PAS listé ici ; elle se gère via
# `allow_target`. On interdit ici les variables de fuite "cachées".
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "p_default",
    "probability_default",
    "true_default",
)


def forbidden_features(df: pd.DataFrame) -> list[str]:
    """Retourne la liste des colonnes de fuite présentes dans ``df``."""
    hits: list[str] = []
    for col in df.columns:
        lowered = str(col).lower()
        if any(tok in lowered for tok in FORBIDDEN_SUBSTRINGS):
            hits.append(str(col))
    return hits


def assert_no_leakage(
    df: pd.DataFrame,
    *,
    feature_columns: list[str] | None = None,
    allow_target: bool = True,
) -> pd.DataFrame:
    """Valide l'absence de fuite dans un DataFrame de features (ou d'entrée).

    Args:
        df: DataFrame à valider.
        feature_columns: liste attendue des features (hors cible). Si fournie,
            vérifie leur présence et rejette les colonnes inattendues.
        allow_target: si True, la colonne cible (``is_default``) est tolérée ;
            sinon rejetée.

    Raises:
        LeakageError: si une variable de fuite est détectée, ou si les colonnes
            ne correspondent pas à la spécification.
    """
    hits = forbidden_features(df)
    if hits:
        raise LeakageError(
            f"Variables de fuite (leakage potentiel) détectées : {hits}. Supprimez-les avant tout entraînement/scoring."
        )

    if feature_columns is not None:
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise LeakageError(f"Colonnes attendues absentes des features : {missing}.")
        allowed = set(feature_columns) | {
            "customer_id",
            "loan_id",
            "is_default" if allow_target else "",
        }
        extra = [c for c in df.columns if c not in allowed]
        if extra:
            raise LeakageError(f"Colonnes inattendues dans le jeu : {extra}.")

    return df


CORRELATION_LEAKAGE_THRESHOLD = 0.75


def assert_no_correlation_leakage(
    df: pd.DataFrame,
    *,
    feature_columns: list[str],
    target: str,
    threshold: float = CORRELATION_LEAKAGE_THRESHOLD,
) -> pd.DataFrame:
    """Détecte la fuite STRUCTURELLE : une feature quasi-parfaitement corrélée à la cible.

    Complète ``assert_no_leakage`` (qui ne bloque que des noms de colonnes littéraux) :
    une variable peut encoder la cible sans jamais s'appeler ``p_default`` — c'est le cas
    d'agrégats construits, en amont, à partir d'un champ lui-même dérivé de la cible.
    Toute corrélation (Pearson, valeur absolue) au-delà de ``threshold`` est bloquante.
    """
    present = [c for c in feature_columns if c in df.columns]
    if not present or target not in df.columns:
        return df
    corr = df[present].astype(float).corrwith(df[target].astype(float))
    offenders = corr[corr.abs() > threshold]
    if not offenders.empty:
        detail = ", ".join(f"{name}={value:.3f}" for name, value in offenders.items())
        raise LeakageError(
            f"Fuite structurelle détectée : corrélation(s) avec la cible au-delà de "
            f"{threshold} : {detail}. Une feature ne doit jamais quasi-encoder la cible."
        )
    return df


def load_and_validate(
    path: str,
    *,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Charge un CSV de features et le valide contre le leakage."""
    df = pd.read_csv(path)
    return assert_no_leakage(df, feature_columns=feature_columns)
