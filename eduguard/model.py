import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score, precision_recall_fscore_support, brier_score_loss, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from .data import NUM, CAT, FEATURES

def preprocessing():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), NUM),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT)
    ], sparse_threshold=0)

def splits(df, seed=42):
    groups = df.id_student.astype(str).to_numpy()
    tr, hold = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed).split(df, df.at_risk, groups=groups))
    rest = df.iloc[hold]
    val_i, remain_i = next(GroupShuffleSplit(n_splits=1, test_size=.6, random_state=seed+1).split(rest, rest.at_risk, groups=rest.id_student.astype(str)))
    remain = rest.iloc[remain_i]
    cal_i, test_i = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed+2).split(remain, remain.at_risk, groups=remain.id_student.astype(str)))
    return df.iloc[tr], rest.iloc[val_i], remain.iloc[cal_i], remain.iloc[test_i]

def quantile(scores, alpha):
    n = len(scores)
    level = min(1, np.ceil((n+1)*(1-alpha))/n)
    return float(np.quantile(scores, level, method="higher"))

def predict_sets(prob, threshold):
    return np.column_stack([prob <= threshold, 1-prob <= threshold])

def metrics(y, p, threshold):
    pred = (p >= .5).astype(int)
    pr, re, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    sets = predict_sets(p, threshold)
    return {"n": int(len(y)), "positive_rate": float(np.mean(y)), "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "precision": float(pr),
            "recall": float(re), "f1": float(f1),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
            "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else None,
            "brier": float(brier_score_loss(y, p)),
            "coverage": float(np.mean(sets[np.arange(len(y)), np.asarray(y, dtype=int)])),
            "avg_set_size": float(sets.sum(axis=1).mean()),
            "empty_set_rate": float(np.mean(sets.sum(axis=1) == 0)),
            "ambiguous_rate": float(np.mean(sets.sum(axis=1) == 2)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}

def train(df, directory="models", seed=42, alpha=.1, demo=False):
    if len(df) < 100 or df.at_risk.nunique() < 2:
        raise ValueError("Need at least 100 rows and both risk classes.")
    train_df, val_df, cal_df, test_df = splits(df, seed)
    for name, frame in [("training", train_df), ("validation", val_df), ("calibration", cal_df), ("test", test_df)]:
        if frame.at_risk.nunique() < 2:
            raise ValueError(f"{name} split contains only one class; use more data.")
    models = {
        "Logistic Regression": Pipeline([("prep", preprocessing()), ("model", LogisticRegression(max_iter=1500, class_weight="balanced"))]),
        "Random Forest": Pipeline([("prep", preprocessing()), ("model", RandomForestClassifier(n_estimators=180, min_samples_leaf=5, max_depth=15, class_weight="balanced_subsample", random_state=seed, n_jobs=-1))])
    }
    report = {}
    for label, model in models.items():
        model.fit(train_df[FEATURES], train_df.at_risk)
        pc = model.predict_proba(cal_df[FEATURES])[:, 1]
        scores = 1 - np.where(cal_df.at_risk.to_numpy() == 1, pc, 1-pc)
        threshold = quantile(scores, alpha)
        pt = model.predict_proba(test_df[FEATURES])[:, 1]
        report[label] = {"test": metrics(test_df.at_risk.to_numpy(), pt, threshold), "conformal_threshold": threshold}
    def validation_f1(label):
        p = models[label].predict_proba(val_df[FEATURES])[:, 1]
        return precision_recall_fscore_support(val_df.at_risk, (p >= .5).astype(int), average="binary", zero_division=0)[2]
    chosen = max(models, key=validation_f1)
    bundle = {"model": models[chosen], "threshold": report[chosen]["conformal_threshold"],
              "alpha": alpha, "feature_columns": FEATURES, "selected": chosen, "demo": demo, "seed": seed}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, directory / "eduguard.joblib")
    result = {"dataset": "synthetic demo" if demo else "OULAD", "cutoff_day_exclusive": 28, "seed": seed, "alpha": alpha,
              "sizes": {"train": len(train_df), "validation": len(val_df), "calibration": len(cal_df), "test": len(test_df)},
              "selected_model": chosen, "selection": "validation F1 at 0.5 threshold", "models": report,
              "notes": "Only first 28 days used. Withdrawn/Fail = risk. Synthetic results are not research findings."}
    (directory / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    test_df.sample(min(100, len(test_df)), random_state=seed).to_csv(directory / "examples.csv", index=False)
    return result

def load_bundle(directory="models"):
    return joblib.load(Path(directory) / "eduguard.joblib")

def infer(bundle, rows):
    prob = bundle["model"].predict_proba(rows[FEATURES])[:, 1]
    return prob, predict_sets(prob, bundle["threshold"])

def explanations(bundle, row):
    pipe = bundle["model"]
    pre = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    x = pre.transform(row[FEATURES])
    names = pre.get_feature_names_out()
    if isinstance(model, RandomForestClassifier):
        import shap
        exp = shap.TreeExplainer(model)(x)
        vals = exp.values
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        values = vals[0]
    else:
        values = x[0] * model.coef_[0]
    return pd.DataFrame({"feature": names, "contribution": values}).assign(abs_contribution=lambda d: d.contribution.abs()).sort_values("abs_contribution", ascending=False)
