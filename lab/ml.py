"""Run with python -m lab.ml. Reports are based exclusively on held-out data."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate, demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import ClassificationMetric


def dataset():
    rng = np.random.default_rng(42)
    group = rng.integers(0, 2, 2400)
    skills = rng.normal(size=2400)
    experience = rng.normal(size=2400)
    access = rng.normal(0.9 * group, 0.7)
    X = pd.DataFrame(dict(skills=skills, experience=experience, access_proxy=access))
    # Deliberately biased historical labels, not a measure of human merit.
    p = 1 / (1 + np.exp(-(1.4 * skills + 0.6 * experience + 1.6 * group - 0.8)))
    return X, rng.binomial(1, p), group


def aif_dataset(X, y, group):
    df = X.reset_index(drop=True).assign(label=y, group=group)
    return BinaryLabelDataset(df=df, label_names=["label"], protected_attribute_names=["group"])


def metrics(X, y, group, pred):
    frame = MetricFrame(metrics={"accuracy": accuracy_score, "selection_rate": selection_rate,
                                "TPR": true_positive_rate, "FPR": false_positive_rate},
                        y_true=y, y_pred=pred, sensitive_features=group)
    truth = aif_dataset(X, y, group)
    classified = truth.copy(deepcopy=True)
    classified.labels = np.asarray(pred).reshape(-1, 1)
    audit = ClassificationMetric(truth, classified, unprivileged_groups=[{"group": 0}], privileged_groups=[{"group": 1}])
    return {"accuracy": accuracy_score(y, pred), "by_group": frame.by_group.to_dict(),
            "group_counts": {str(g): int(sum(group == g)) for g in (0, 1)},
            "demographic_parity_gap": demographic_parity_difference(y, pred, sensitive_features=group),
            "equalized_odds_gap": equalized_odds_difference(y, pred, sensitive_features=group),
            "aif360_statistical_parity_difference": audit.statistical_parity_difference(),
            "aif360_disparate_impact": audit.disparate_impact()}


def run(output=Path("artifacts")):
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    X, y, group = dataset()
    train, rest = train_test_split(np.arange(len(y)), test_size=0.4, random_state=42, stratify=group * 2 + y)
    calibration, test = train_test_split(rest, test_size=0.5, random_state=42, stratify=(group * 2 + y)[rest])
    base = LogisticRegression().fit(X.iloc[train], y[train])
    weighted_data = Reweighing(unprivileged_groups=[{"group": 0}], privileged_groups=[{"group": 1}]).fit_transform(aif_dataset(X.iloc[train], y[train], group[train]))
    weighted = LogisticRegression().fit(X.iloc[train], y[train], sample_weight=weighted_data.instance_weights)
    threshold = ThresholdOptimizer(estimator=base, constraints="demographic_parity", objective="accuracy_score", prefit=True, predict_method="predict_proba")
    threshold.fit(X.iloc[calibration], y[calibration], sensitive_features=group[calibration])
    predictions = {"baseline": base.predict(X.iloc[test]), "aif360_reweighed": weighted.predict(X.iloc[test]),
                   "fairlearn_threshold": threshold.predict(X.iloc[test], sensitive_features=group[test], random_state=42)}
    report = {name: metrics(X.iloc[test], y[test], group[test], pred) for name, pred in predictions.items()}
    background = X.iloc[train].to_numpy()
    explanation = shap.LinearExplainer(base, background)(X.iloc[test[:40]].to_numpy())
    lime = LimeTabularExplainer(background, feature_names=list(X.columns), class_names=["not selected", "selected"], random_state=42)
    local = lime.explain_instance(X.iloc[test[0]].to_numpy(), lambda rows: base.predict_proba(pd.DataFrame(rows, columns=X.columns)), num_features=3)
    report["explainability"] = {"model": "baseline logistic regression only", "shap_units": "log odds",
        "shap_base_value": float(explanation.base_values[0]), "shap_first_row": dict(zip(X.columns, explanation.values[0].tolist())),
        "shap_mean_absolute": dict(zip(X.columns, np.abs(explanation.values).mean(axis=0).tolist())),
        "first_row_log_odds": float(base.decision_function(X.iloc[test[:1]])[0]),
        "lime_probability_contributions": local.as_list(), "lime_surrogate_r_squared": float(local.score)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "ml_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    local.save_to_file(str(output / "lime_explanation.html"))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
