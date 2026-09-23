"""Tests des commandes CLI — câblage (arguments → fonctions appelées), pas la logique métier
déjà couverte ailleurs. Standard Click : ``CliRunner``, dépendances lourdes (MLflow, disque)
isolées par ``monkeypatch`` plutôt que réellement invoquées.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from click.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _patch_mlflow_submodule(monkeypatch, dotted_module: str, attr: str, replacement) -> None:
    """``mlflow.sklearn`` (et les autres "flavors" ML) est chargé paresseusement : l'objet
    accessible comme attribut du paquet (``mlflow.sklearn``) n'est PAS celui enregistré dans
    ``sys.modules`` — patcher l'un ne modifie pas l'autre. Confirmé empiriquement (deux ``id()``
    différents) en écrivant ces tests. Patcher directement via ``sys.modules`` est la façon
    fiable d'intercepter ces appels ; nécessite d'avoir importé le sous-module au moins une fois."""
    import importlib

    importlib.import_module(dotted_module)
    monkeypatch.setattr(sys.modules[dotted_module], attr, replacement)


# --- cif-generate ---------------------------------------------------------------------------


def test_generate_passes_cli_overrides_to_config(runner, monkeypatch) -> None:
    from cli import generate

    captured: dict = {}

    def _fake_load_config(overrides=None):
        captured["overrides"] = overrides
        return _fake_cfg()

    monkeypatch.setattr(generate, "load_config", _fake_load_config)
    monkeypatch.setattr(
        generate, "generate_datasets", lambda cfg: SimpleNamespace(customers="c", loans="l", savings="s")
    )
    monkeypatch.setattr(generate, "save_datasets", lambda cfg, ds: {"customers": "x"})
    features = pd.DataFrame({"is_default": [0, 1, 0]})
    monkeypatch.setattr(generate, "build_features", lambda *a, **k: features)
    monkeypatch.setattr(generate, "save_features", lambda *a, **k: "out.parquet")
    monkeypatch.setattr(generate, "feature_columns", lambda cfg: ["a", "b"])

    result = runner.invoke(generate.main, ["--n-customers", "50", "--seed", "7"])

    assert result.exit_code == 0, result.output
    assert "data.n_customers=50" in captured["overrides"]
    assert "data.seed=7" in captured["overrides"]


def _fake_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(processed_dir="data/processed", n_customers=10, default_rate=0.1, seed=1),
        features=SimpleNamespace(target="is_default", output_file="f.parquet"),
    )


# --- cif-train -------------------------------------------------------------------------------


def test_train_promotes_only_when_register_flag_set(runner, monkeypatch, tmp_path) -> None:
    from cli import train

    parquet = tmp_path / "features.parquet"
    pd.DataFrame({"x": [1, 2]}).to_parquet(parquet)

    cfg = SimpleNamespace(
        data=SimpleNamespace(processed_dir=str(tmp_path), seed=1),
        features=SimpleNamespace(output_file="features.parquet"),
        model=SimpleNamespace(mlflow_tracking_uri="sqlite:///x.db", mlflow_experiment="e", algorithm="xgboost"),
    )
    monkeypatch.setattr(train, "load_config", lambda: cfg)
    monkeypatch.setattr(train.mlflow, "set_tracking_uri", lambda *a: None)
    monkeypatch.setattr(train.mlflow, "set_experiment", lambda *a: None)
    monkeypatch.setattr(
        train,
        "train_and_log",
        lambda *a, **k: SimpleNamespace(run_id="r1", metrics={"roc_auc": 0.7, "pr_auc": 0.5, "brier": 0.1}),
    )
    promote_calls: list = []
    monkeypatch.setattr(train, "latest_version", lambda client, name: "3")
    monkeypatch.setattr(
        train,
        "promote_if_better",
        lambda client, name, version, **k: promote_calls.append(version) or SimpleNamespace(promoted=True, reason="ok"),
    )

    result = runner.invoke(train.main, ["--data", str(parquet), "--no-register"])
    assert result.exit_code == 0, result.output
    assert promote_calls == []  # --no-register : jamais de promotion

    result = runner.invoke(train.main, ["--data", str(parquet), "--register"])
    assert result.exit_code == 0, result.output
    assert promote_calls == ["3"]  # --register : promotion tentée sur la version fraîche


# --- cif-evaluate ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "roc_auc,ece,expected_gate",
    [
        (0.70, 0.05, "GO"),  # au-dessus du seuil GO, ECE correct
        (0.60, 0.05, "NO-GO"),  # entre nogo et go : pas assez bon pour GO
        (0.50, 0.05, "NO-GO"),  # sous le seuil NO-GO explicite
        (0.70, 0.25, "NO-GO"),  # ECE > 2x seuil : NO-GO même avec bon AUC
    ],
)
def test_evaluate_decision_gate_matches_protocol_thresholds(
    runner, monkeypatch, tmp_path, roc_auc, ece, expected_gate
) -> None:
    from cli import evaluate

    parquet = tmp_path / "features.parquet"
    df = pd.DataFrame({"a": range(40), "b": range(40), "is_default": [0, 1] * 20})
    df.to_parquet(parquet)

    cfg = SimpleNamespace(
        data=SimpleNamespace(processed_dir=str(tmp_path)),
        features=SimpleNamespace(target="is_default", output_file="features.parquet"),
        evaluation=SimpleNamespace(
            report_dir=str(tmp_path / "reports"),
            bootstrap_iterations=10,
            robustness_noise_levels=[0.0],
            go_roc_auc=0.65,
            nogo_roc_auc=0.55,
            ece_threshold=0.10,
        ),
        model=SimpleNamespace(test_size=0.5, random_state=0),
    )
    monkeypatch.setattr(evaluate, "load_config", lambda: cfg)
    monkeypatch.setattr(evaluate, "feature_columns", lambda fc: ["a", "b"])
    monkeypatch.setattr(evaluate, "train_plain", lambda *a, **k: (object(), {"roc_auc": roc_auc, "ece": ece}))

    class _Stub:
        def predict_proba(self, X):
            import numpy as np

            return np.tile([1 - roc_auc, roc_auc], (len(X), 1))

    monkeypatch.setattr(evaluate, "train_plain", lambda *a, **k: (_Stub(), {"roc_auc": roc_auc, "ece": ece}))

    result = runner.invoke(
        evaluate.main, ["--data", str(parquet), "--no-bootstrap", "--no-robustness", "--no-fairness", "--no-ablation"]
    )
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "reports" / "evaluation_report.json").read_text())
    assert report["decision_gate"] == expected_gate


# --- cif-decision ------------------------------------------------------------------------------


def test_decision_writes_one_jsonl_line_per_customer(runner, monkeypatch, tmp_path) -> None:
    from cli import decision

    parquet = tmp_path / "features.parquet"
    pd.DataFrame({"a": [1.0, 2.0], "customer_id": [10, 20], "n_past_loans": [0, 3]}).to_parquet(parquet)
    out_path = tmp_path / "decisions.jsonl"

    cfg = SimpleNamespace(
        data=SimpleNamespace(processed_dir=str(tmp_path)),
        features=SimpleNamespace(output_file="features.parquet", target="is_default"),
        model=SimpleNamespace(mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db"),
        decision=SimpleNamespace(
            approve_threshold=0.1,
            review_threshold=0.25,
            hard_reject_threshold=0.5,
            thin_file_review=True,
            thin_file_max_loans=1,
        ),
    )
    monkeypatch.setattr(decision, "load_config", lambda: cfg)
    # set_tracking_uri s'exécute réellement (chemin isolé dans tmp_path) : le neutraliser ferait
    # retomber mlflow sur son magasin de fichiers ambiant (./mlruns, déjà peuplé par d'autres
    # runs dans ce dépôt), et casserait le test au lieu de l'isoler.
    monkeypatch.setattr(decision, "feature_columns", lambda fc: ["a"])

    class _Stub:
        def predict_proba(self, X):
            import numpy as np

            return np.array([[0.9, 0.05], [0.6, 0.4]])

    _patch_mlflow_submodule(monkeypatch, "mlflow.sklearn", "load_model", lambda uri: _Stub())

    result = runner.invoke(decision.main, ["--data", str(parquet), "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert {json.loads(line)["customer_id"] for line in lines} == {10, 20}


# --- cif-export-model --------------------------------------------------------------------------


def test_export_model_replaces_existing_output_dir(runner, monkeypatch, tmp_path) -> None:
    from cli import export_model

    out_dir = tmp_path / "model"
    out_dir.mkdir()
    (out_dir / "stale.txt").write_text("old")

    monkeypatch.setattr(export_model, "load_config", lambda overrides=None: _fake_cfg_full())
    monkeypatch.setattr(export_model, "generate_datasets", lambda cfg: SimpleNamespace(customers=1, loans=2, savings=3))
    monkeypatch.setattr(export_model, "build_features", lambda *a, **k: pd.DataFrame({"is_default": [0, 1]}))
    monkeypatch.setattr(export_model, "train_plain", lambda *a, **k: (object(), {"roc_auc": 0.9}))
    saved: dict = {}
    _patch_mlflow_submodule(monkeypatch, "mlflow.sklearn", "save_model", lambda **kw: saved.update(kw) or None)

    result = runner.invoke(export_model.main, ["--output", str(out_dir)])
    assert result.exit_code == 0, result.output
    assert not (out_dir / "stale.txt").exists()  # l'ancien dossier a bien été purgé
    assert saved["path"] == str(out_dir)


def _fake_cfg_full() -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(),
        features=SimpleNamespace(),
        model=SimpleNamespace(),
    )


# --- cif-export-lc-model -----------------------------------------------------------------------


def test_export_lc_model_defaults_threshold_when_tag_missing(runner, monkeypatch, tmp_path) -> None:
    from cli import export_lending_club_model as export_lc

    out_dir = tmp_path / "model_lc"
    monkeypatch.setattr(export_lc.mlflow, "set_tracking_uri", lambda *a: None)
    local = tmp_path / "downloaded"
    local.mkdir()
    (local / "MLmodel").write_text("x")
    monkeypatch.setattr(export_lc.mlflow.artifacts, "download_artifacts", lambda uri: str(local))

    class _Client:
        def get_model_version_by_alias(self, name, alias):
            return SimpleNamespace(tags={}, version="4")  # pas de cost_threshold posé

    monkeypatch.setattr(export_lc.mlflow.tracking, "MlflowClient", lambda: _Client())

    result = runner.invoke(export_lc.main, ["--output", str(out_dir), "--model-uri", "models:/x@champion"])
    assert result.exit_code == 0, result.output
    assert (out_dir / "cost_threshold.txt").read_text() == "0.15"  # défaut documenté, jamais inventé ailleurs


# --- cif-ingest-lc -----------------------------------------------------------------------------


def test_ingest_lc_writes_features_parquet(runner, monkeypatch, tmp_path) -> None:
    from cli import ingest_lending_club as ingest_lc

    out_path = tmp_path / "processed" / "lc.parquet"
    cfg = SimpleNamespace(lending_club=SimpleNamespace(processed_path=str(out_path)))
    monkeypatch.setattr(ingest_lc, "load_config", lambda: cfg)

    interim = pd.DataFrame({"a": [1, 2]})
    interim_path = tmp_path / "interim.parquet"
    interim.to_parquet(interim_path)
    monkeypatch.setattr(ingest_lc, "build_interim", lambda cfg, nrows=None: interim_path)
    features = pd.DataFrame({"a": [1, 2], "is_default": [0, 1]})
    monkeypatch.setattr(ingest_lc, "build_features", lambda df: features)

    result = runner.invoke(ingest_lc.main, [])
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert len(pd.read_parquet(out_path)) == 2


# --- cif-benchmark -----------------------------------------------------------------------------


def test_benchmark_registers_champion_only_when_register_flag_set(runner, monkeypatch, tmp_path) -> None:
    from cli import benchmark

    parquet = tmp_path / "lc.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(parquet)
    cfg = SimpleNamespace(
        lending_club=SimpleNamespace(processed_path=str(parquet), cost_false_negative=1.0, cost_false_positive=1.0),
        model=SimpleNamespace(mlflow_tracking_uri="sqlite:///x.db"),
    )
    monkeypatch.setattr(benchmark, "load_config", lambda: cfg)
    monkeypatch.setattr(benchmark.mlflow, "set_tracking_uri", lambda *a: None)
    monkeypatch.setattr(benchmark.mlflow, "set_experiment", lambda *a: None)

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(benchmark.mlflow, "start_run", lambda **k: _Ctx())
    monkeypatch.setattr(benchmark.mlflow, "log_metrics", lambda *a, **k: None)
    monkeypatch.setattr(benchmark.mlflow, "log_params", lambda *a, **k: None)
    monkeypatch.setattr(benchmark.mlflow, "log_artifacts", lambda *a, **k: None)

    champion = SimpleNamespace(name="logistic")
    result_obj = SimpleNamespace(
        metrics={
            "test_metrics": {"logistic": {"roc_auc": 0.7, "ece": 0.05}},
            "champion": "logistic",
            "decision": {"cost_threshold": 0.2},
        },
        champion=champion,
        test=pd.DataFrame({"x": [1]}),
    )
    monkeypatch.setattr(benchmark, "run_benchmark", lambda *a, **k: result_obj)
    monkeypatch.setattr(benchmark, "write_artifacts", lambda *a, **k: tmp_path)

    register_calls: list = []
    monkeypatch.setattr(benchmark, "register_champion", lambda *a, **k: register_calls.append(1) or "2")

    result = runner.invoke(benchmark.main, ["--no-register"])
    assert result.exit_code == 0, result.output
    assert register_calls == []

    result = runner.invoke(benchmark.main, ["--register"])
    assert result.exit_code == 0, result.output
    assert register_calls == [1]


# --- cif-promote / cif-rollback -----------------------------------------------------------------


def test_promote_exits_nonzero_when_refused(runner, monkeypatch) -> None:
    from cli import promote as promote_cli

    monkeypatch.setattr(promote_cli, "_client", lambda uri: object())
    monkeypatch.setattr(promote_cli, "latest_version", lambda client, name: "5")
    monkeypatch.setattr(
        promote_cli,
        "promote_if_better",
        lambda *a, **k: SimpleNamespace(promoted=False, version="5", reason="AUC insuffisant"),
    )

    result = runner.invoke(promote_cli.promote, [])
    assert result.exit_code == 1
    assert "REFUSÉE" in result.output


def test_promote_exits_zero_when_accepted(runner, monkeypatch) -> None:
    from cli import promote as promote_cli

    monkeypatch.setattr(promote_cli, "_client", lambda uri: object())
    monkeypatch.setattr(promote_cli, "latest_version", lambda client, name: "5")
    monkeypatch.setattr(
        promote_cli, "promote_if_better", lambda *a, **k: SimpleNamespace(promoted=True, version="5", reason="ok")
    )

    result = runner.invoke(promote_cli.promote, [])
    assert result.exit_code == 0
    assert "PROMUE" in result.output


def test_rollback_echoes_restored_version(runner, monkeypatch) -> None:
    from cli import promote as promote_cli

    monkeypatch.setattr(promote_cli, "_client", lambda uri: object())
    monkeypatch.setattr(promote_cli, "rollback", lambda client, name: "3")

    result = runner.invoke(promote_cli.rollback_cmd, [])
    assert result.exit_code == 0
    assert "v3" in result.output


# --- cif-replay-monitoring ----------------------------------------------------------------------


def test_replay_monitoring_logs_one_metric_step_per_month(runner, monkeypatch, tmp_path) -> None:
    from cli import replay_monitoring

    parquet = tmp_path / "lc.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(parquet)
    cfg = SimpleNamespace(
        lending_club=SimpleNamespace(processed_path=str(parquet)),
        model=SimpleNamespace(mlflow_tracking_uri="sqlite:///x.db"),
    )
    monkeypatch.setattr(replay_monitoring, "load_config", lambda: cfg)
    monkeypatch.setattr(replay_monitoring, "temporal_partition", lambda df, cfg: (df, df, df))
    monkeypatch.setattr(replay_monitoring.mlflow, "set_tracking_uri", lambda *a: None)
    monkeypatch.setattr(replay_monitoring.mlflow, "set_experiment", lambda *a: None)

    loaded = SimpleNamespace(unwrap_python_model=lambda: SimpleNamespace(model="champion-stub"))
    monkeypatch.setattr(replay_monitoring.mlflow.pyfunc, "load_model", lambda uri: loaded)

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(replay_monitoring.mlflow, "start_run", lambda **k: _Ctx())
    logged: list = []
    monkeypatch.setattr(
        replay_monitoring.mlflow, "log_metric", lambda name, value, step=None: logged.append((name, step))
    )
    monkeypatch.setattr(replay_monitoring.mlflow, "log_artifacts", lambda *a, **k: None)

    drift_series = pd.DataFrame({"month": ["2014-01", "2014-02"], "psi": [0.1, 0.2]})
    perf_series = pd.DataFrame({"month": ["2015-01"], "roc_auc": [0.7], "ece": [0.05]})
    result_obj = SimpleNamespace(drift_series=drift_series, performance_series=perf_series, any_alert=False)
    monkeypatch.setattr(replay_monitoring, "run_replay", lambda *a, **k: result_obj)
    monkeypatch.setattr(replay_monitoring, "write_replay_artifacts", lambda *a, **k: tmp_path)

    result = runner.invoke(replay_monitoring.main, [])
    assert result.exit_code == 0, result.output
    psi_steps = [step for name, step in logged if name == "psi"]
    assert psi_steps == [201401, 201402]  # un point par mois, ordonné


# --- cif-register-model -------------------------------------------------------------------------


def test_register_model_fails_cleanly_when_artifact_missing(runner, monkeypatch) -> None:
    from cli import register

    monkeypatch.setattr(register, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(register, "find_official_artifact", lambda: None)

    result = runner.invoke(register.main, [])
    assert result.exit_code != 0
    assert "introuvable" in result.output


def test_register_model_fails_cleanly_when_version_already_exists(runner, monkeypatch) -> None:
    from cli import register

    monkeypatch.setattr(register, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(register, "find_official_artifact", lambda: "/tmp/x.joblib")
    monkeypatch.setattr(register, "version_exists", lambda name, version: True)

    result = runner.invoke(register.main, ["--version", "1.0.0"])
    assert result.exit_code != 0
    assert "déjà enregistrée" in result.output


def test_register_model_uses_explicit_metrics_json_without_touching_synthetic_data(runner, monkeypatch) -> None:
    from cli import register

    cfg = SimpleNamespace(
        features=SimpleNamespace(target="is_default"),
        decision=SimpleNamespace(approve_threshold=0.1, review_threshold=0.25, hard_reject_threshold=0.5),
    )
    monkeypatch.setattr(register, "load_config", lambda: cfg)
    monkeypatch.setattr(register, "find_official_artifact", lambda: "/tmp/x.joblib")
    monkeypatch.setattr(register, "version_exists", lambda name, version: False)
    monkeypatch.setattr(register, "feature_columns", lambda fc: ["a", "b"])

    # Si le chemin "métriques calculées sur le synthétique" était emprunté par erreur, ceci
    # planterait (fichier inexistant) — confirme que --metrics-json le court-circuite bien.
    monkeypatch.setattr(
        register,
        "_load_synthetic_features",
        lambda cfg: (_ for _ in ()).throw(AssertionError("ne doit pas être appelé")),
    )

    captured: dict = {}
    monkeypatch.setattr(
        register,
        "register_from_joblib",
        lambda joblib_path, **kw: (
            captured.update(kw) or SimpleNamespace(model_name="cif_credit_official", version="1.0.0", stage="Staging")
        ),
    )

    result = runner.invoke(register.main, ["--metrics-json", json.dumps({"roc_auc": 0.91})])
    assert result.exit_code == 0, result.output
    assert captured["metrics"] == {"roc_auc": 0.91}
