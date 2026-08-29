"""Schémas de configuration du projet (dataclasses Hydra)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Configuration de la génération des données synthétiques."""

    n_customers: int = 10_000
    n_savings_months: int = 24
    max_past_loans: int = 5
    default_rate: float = 0.1183
    latent_factor_exponent: float = 3.0
    seed: int = 42
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


@dataclass
class FeatureConfig:
    """Configuration du feature engineering."""

    target: str = "is_default"
    output_file: str = "features_prepared.parquet"
    # Feature families (utilisées par l'ablation study) — 25 features au total
    families: dict[str, list[str]] = field(
        default_factory=lambda: {
            "profile_income": [
                "age",
                "gender_num",
                "sector_num",
                "location_num",
                "monthly_income",
                "income_volatility",
            ],
            "savings": [
                "current_savings",
                "avg_savings_24m",
                "savings_stability",
                "savings_volatility",
                "loan_to_savings_ratio",
                "savings_trend",
                "savings_min_24m",
            ],
            "history": [
                "n_past_loans",
                "repayment_regularity",
                "max_dpd",
                "payments_on_time_ratio",
                "loan_history_quality",
                "active_loans",
            ],
            "context": [
                "seniority_months",
                "current_loan_request",
                "current_loan_duration",
                "current_loan_purpose_num",
                "income_to_request_ratio",
                "debt_ratio",
            ],
        }
    )


@dataclass
class CalibrationConfig:
    """Configuration de la calibration des probabilités (exigence §93 du cahier des charges)."""

    enabled: bool = True
    method: str = "isotonic"
    cv: int = 3


@dataclass
class ModelConfig:
    """Configuration de l'entraînement du modèle."""

    algorithm: str = "xgboost"
    test_size: float = 0.2
    random_state: int = 42
    n_trials: int = 30
    # Hyperparamètres XGBoost — alignés sur le modèle officiel calibré (dépôt de
    # référence `cifci`) : profondeur 4, lr 0.03, 300 arbres (pas le défaut).
    xgboost: dict[str, float | int | str] = field(
        default_factory=lambda: {
            "max_depth": 4,
            "learning_rate": 0.03,
            "n_estimators": 300,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "scale_pos_weight": 1.0,
        }
    )
    artifacts_dir: str = "data/artifacts"
    mlflow_tracking_uri: str = "sqlite:///mlruns.db"
    mlflow_experiment: str = "cif_credit_intelligence"
    # Calibration des probabilités (Isotonic via CalibratedClassifierCV)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    # Familles de features utilisées pour l'ablation (M0 baseline, M1, M2, M3)
    ablation_families: list[str] = field(default_factory=lambda: ["profile_income", "savings", "history", "context"])


@dataclass
class EvaluationConfig:
    """Configuration de l'évaluation et de l'audit."""

    bootstrap_iterations: int = 1_000
    ci_percentile: float = 0.95
    go_roc_auc: float = 0.65
    nogo_roc_auc: float = 0.55
    ece_threshold: float = 0.10
    robustness_noise_levels: list[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    report_dir: str = "data/artifacts/reports"


@dataclass
class DecisionConfig:
    """Configuration du decision engine (Model ≠ Policy ≠ Workflow ≠ Decision)."""

    # Seuils sur la probabilité de défaut
    approve_threshold: float = 0.10
    review_threshold: float = 0.25
    hard_reject_threshold: float = 0.50
    # Coûts économiques (unités monétaires arbitraires)
    cost_review: float = 5_000.0
    cost_false_negative: float = 300_000.0
    cost_false_positive: float = 50_000.0
    # Human-in-the-loop
    thin_file_review: bool = True
    thin_file_max_loans: int = 1


@dataclass
class ServingConfig:
    """Configuration de l'API de scoring."""

    host: str = "0.0.0.0"
    port: int = 8000
    model_uri: str = "models:/cif_credit_official/latest"
    request_limits: dict[str, int] = field(default_factory=lambda: {"rate_per_minute": 120})


@dataclass
class AppConfig:
    """Configuration racine du projet."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
