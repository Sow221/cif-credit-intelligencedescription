"""Tests du rapport de monitoring — PSI, alertes cabinet, drift Evidently (Semaine 4)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from monitoring.drift_report import (
    check_alert,
    check_monitoring_alerts,
    generate_psi_report,
    psi_score,
)

RNG = np.random.default_rng(7)


def _dataset(n: int, shift: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": RNG.normal(40 + shift, 10, n),
            "income": RNG.lognormal(mean=13 + shift, sigma=0.5, size=n),
        }
    )


class TestPSI:
    def test_psi_close_to_zero_for_same_distribution(self):
        ref = _dataset(3000)
        cur = _dataset(3000)
        assert psi_score(ref["age"].to_numpy(), cur["age"].to_numpy()) < 0.05

    def test_psi_high_for_drifted_distribution(self):
        ref = _dataset(500)
        cur = _dataset(500, shift=6.0)
        assert psi_score(ref["age"].to_numpy(), cur["age"].to_numpy()) > 0.25

    def test_psi_empty_inputs_returns_zero(self):
        assert psi_score(np.array([]), np.array([1.0])) == 0.0

    def test_generate_psi_report_writes_json(self, tmp_path):
        ref = _dataset(400, shift=0.0)
        cur = _dataset(400, shift=4.0)
        out = tmp_path / "psi.json"
        values = generate_psi_report(ref, cur, out, columns=["age", "income"])
        assert set(values) == {"age", "income"}
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved == values


class TestAlerts:
    def test_check_alert_min_threshold(self):
        assert check_alert("roc_auc_min", 0.60)
        assert not check_alert("roc_auc_min", 0.70)
        assert check_alert("ece_max", 0.15)
        assert check_alert("psi_max", 0.30)
        assert not check_alert("psi_max", 0.10)

    def test_check_monitoring_alerts_defaults(self):
        alerts = check_monitoring_alerts({"roc_auc": 0.60, "ece": 0.12, "psi": 0.30, "drift_ratio": 0.4})
        assert alerts["roc_auc"]["alerted"]
        assert alerts["ece"]["alerted"]
        assert alerts["psi"]["alerted"]
        assert alerts["drift_ratio"]["alerted"]
        assert alerts["roc_auc"]["threshold"] == 0.65

    def test_monitoring_alerts_no_false_positive(self):
        alerts = check_monitoring_alerts({"roc_auc": 0.83, "ece": 0.02, "psi": 0.05, "drift_ratio": 0.1})
        assert not any(a["alerted"] for a in alerts.values())

    def test_unknown_metric_ignored(self):
        alerts = check_monitoring_alerts({"roc_auc": 0.9})
        assert set(alerts) == {"roc_auc"}
        assert not alerts["roc_auc"]["alerted"]
        assert alerts["roc_auc"]["threshold"] == 0.65
