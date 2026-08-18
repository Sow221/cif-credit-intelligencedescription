"""Configuration Pydantic centralisée — exigence du cabinet (Semaine 1, src/config/settings.py).

Classe ``Settings`` (Pydantic v2, ``pydantic-settings``) : source unique de la configuration
pour l'API, les services et l'infra. `Variables d'environnement en .env `, valeurs par
défaut strictes alignées sur le cahier des charges CIF (§M07/§64/§65/§93).

La config ML (Hydra + YAML) reste pilotée par ``config.load.load_config`` pour les pipelines ;
``Settings.from_app_config`` fournit le pont vers les dataclasses Hydra.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.load import load_config
from config.schema import AppConfig

CalibrationMethod = Literal["isotonic", "sigmoid"]


class DataSettings(BaseModel):
    """Paramètres des données synthétiques (miroir de DataConfig)."""

    n_customers: int = 10_000
    n_savings_months: int = 24
    max_past_loans: int = 5
    default_rate: float = 0.1183
    seed: int = 42


class FeatureSettings(BaseModel):
    """Paramètres du feature engineering."""

    target: str = "is_default"
    output_file: str = "features_prepared.parquet"


class ModelSettings(BaseModel):
    """Paramètres du modèle, de la calibration (§93) et du registry MLflow."""

    algorithm: str = "xgboost"
    test_size: float = 0.2
    random_state: int = 42
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 300
    calibration_enabled: bool = True
    calibration_method: CalibrationMethod = "isotonic"
    calibration_cv: int = 3
    artifacts_dir: str = "data/artifacts"
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    mlflow_experiment: str = "cif_credit_intelligence"


class DecisionSettings(BaseModel):
    """Seuils et règles du decision engine (Model ≠ Policy ≠ Workflow ≠ Decision, §64)."""

    approve_threshold: float = 0.10
    review_threshold: float = 0.25
    hard_reject_threshold: float = 0.50
    thin_file_review: bool = True
    thin_file_max_loans: int = 1


class MonitoringSettings(BaseModel):
    """Seuils d'alerte monitoring (exigence cabinet : ROC-AUC < 0.65, ECE > 0.10, PSI > 0.25)."""

    roc_auc_alert: float = 0.65
    ece_alert: float = 0.10
    psi_alert: float = 0.25
    drift_ratio_alert: float = 0.20


class ApiSettings(BaseModel):
    """Paramètres de l'API (FastAPI/JWT/rate limiting) — exigences cabinet Semaine 2."""

    host: str = "0.0.0.0"
    port: int = 8000
    request_limits_rate_per_minute: int = 120
    jwt_algorithm: Literal["HS256"] = "HS256"
    jwt_expiry_minutes: int = 60


class DatabaseSettings(BaseModel):
    """Connexion PostgreSQL 16 (audit trail, registry) — exigences cabinet Semaine 3."""

    url: str = "postgresql+psycopg2://mlflow:mlflow@localhost:5432/mlflow"


class Settings(BaseSettings):
    """Config racine : chaque variable surchargeable par environnement (préfixe CIF_)."""

    model_config = SettingsConfigDict(
        env_prefix="CIF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        validate_assignment=True,
    )

    data: DataSettings = DataSettings()
    features: FeatureSettings = FeatureSettings()
    model: ModelSettings = ModelSettings()
    decision: DecisionSettings = DecisionSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    api: ApiSettings = ApiSettings()
    database: DatabaseSettings = DatabaseSettings()

    jwt_secret: SecretStr = Field(default=SecretStr("change-me-in-production"), min_length=8)
    log_level: str = "INFO"

    @classmethod
    def from_app_config(cls, cfg: AppConfig | None = None) -> Settings:
        """Assemble un Settings depuis la config Hydra (pont avec les pipelines ML)."""
        cfg = cfg or load_config()
        return cls(
            data=DataSettings(
                n_customers=cfg.data.n_customers,
                n_savings_months=cfg.data.n_savings_months,
                max_past_loans=cfg.data.max_past_loans,
                default_rate=cfg.data.default_rate,
                seed=cfg.data.seed,
            ),
            features=FeatureSettings(target=cfg.features.target, output_file=cfg.features.output_file),
            model=ModelSettings(
                algorithm=cfg.model.algorithm,
                test_size=cfg.model.test_size,
                random_state=cfg.model.random_state,
                max_depth=int(cfg.model.xgboost.get("max_depth", 6)),
                learning_rate=float(cfg.model.xgboost.get("learning_rate", 0.1)),
                n_estimators=int(cfg.model.xgboost.get("n_estimators", 300)),
                calibration_enabled=cfg.model.calibration.enabled,
                calibration_method=cfg.model.calibration.method,
                calibration_cv=cfg.model.calibration.cv,
                artifacts_dir=cfg.model.artifacts_dir,
                mlflow_tracking_uri=cfg.model.mlflow_tracking_uri,
                mlflow_experiment=cfg.model.mlflow_experiment,
            ),
            decision=DecisionSettings(
                approve_threshold=cfg.decision.approve_threshold,
                review_threshold=cfg.decision.review_threshold,
                hard_reject_threshold=cfg.decision.hard_reject_threshold,
                thin_file_review=cfg.decision.thin_file_review,
                thin_file_max_loans=cfg.decision.thin_file_max_loans,
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retourne le Settings global en cache (lecture unique du .env)."""
    return Settings()
