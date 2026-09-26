"""Retrain and re-save the two models the notebook selected.

Scope
-----
Model *selection* happened in the notebook: sections ML.10-ML.12 compared eight
regressors and ML.15-ML.16 compared seven classifiers, each under identical
repeated cross-validation, and picked a winner by the primary metric declared
in advance (cross-validated MAE for regression, cross-validated macro F1 for
classification). That comparison, together with the diagnostics, explainability
and statistical testing, stays in ``notebooks/machine_learning.ipynb`` — it is
the analysis, and duplicating it here would only create two versions to keep in
step.

This script does the narrower job that is genuinely useful outside the
notebook: it **regenerates the selected artefacts** by re-running the same two
hyperparameter searches on the same split with the same seed, so that
``models/`` can be rebuilt from the CSV in one command. Every value that
controls the outcome is imported from :mod:`src.preprocessing` or defined below
with the same value the notebook used, and ``--verify`` checks the result
against the metrics recorded in ``models/model_metadata.json``.

Usage
-----
Rebuild the artefacts in place::

    python -m src.train

Rebuild into a scratch directory and compare against the committed metrics
without overwriting anything::

    python -m src.train --output-dir /tmp/models --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.preprocessing import (  # noqa: E402
        CATEGORICAL_FEATURES,
        DATA_PATH,
        ENGINEERED_NUMERIC,
        LIKERT_MAX,
        LIKERT_MIN,
        MODEL_DIR,
        MODEL_FEATURES,
        NUMERIC_FEATURES,
        STRESS_BAND_EDGES,
        STRESS_BAND_LABELS,
        TARGET,
        build_modelling_frame,
        to_stress_band,
    )
else:
    from .preprocessing import (
        CATEGORICAL_FEATURES,
        DATA_PATH,
        ENGINEERED_NUMERIC,
        LIKERT_MAX,
        LIKERT_MIN,
        MODEL_DIR,
        MODEL_FEATURES,
        NUMERIC_FEATURES,
        STRESS_BAND_EDGES,
        STRESS_BAND_LABELS,
        TARGET,
        build_modelling_frame,
        to_stress_band,
    )

# --------------------------------------------------------------------------- #
# Configuration - identical to section ML.1 of the notebook
# --------------------------------------------------------------------------- #

RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_SPLITS, CV_REPEATS = 5, 3

#: Search space for the regression winner (notebook section ML.12).
FOREST_SEARCH_SPACE = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [2, 3, 4, 6, None],
    "model__min_samples_leaf": [1, 2, 4, 8],
    "model__max_features": ["sqrt", "log2", 0.5, 1.0],
}

#: Search space for the classification winner (notebook section ML.16).
SVM_SEARCH_SPACE = {
    "model__C": [0.1, 0.5, 1.0, 3.0, 10.0, 30.0],
    "model__gamma": ["scale", 0.01, 0.05, 0.1, 0.3],
}

SEARCH_ITERATIONS = 25


# --------------------------------------------------------------------------- #
# Pipeline construction - identical to section ML.8
# --------------------------------------------------------------------------- #


def build_preprocessor(numeric_features, categorical_features) -> ColumnTransformer:
    """Build the ``ColumnTransformer`` holding every *learned* preprocessing step.

    Median imputation and standardisation for the numeric branch, most-frequent
    imputation and one-hot encoding for the categorical branch. Because this is
    a pipeline step, cross-validation and the hyperparameter searches re-fit it
    from scratch inside every fold, on that fold's training portion only.
    """
    numeric_branch = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_branch = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_branch, numeric_features),
            ("cat", categorical_branch, categorical_features),
        ]
    )


def build_model_pipeline(estimator) -> Pipeline:
    """Attach an estimator to a freshly built preprocessor."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
            ("model", estimator),
        ]
    )


# --------------------------------------------------------------------------- #
# Data preparation - identical to section ML.7
# --------------------------------------------------------------------------- #


def build_split(data_path: Path | str = DATA_PATH):
    """Load the data and reproduce the notebook's stratified train/test split.

    The split is stratified by stress band so that the rare Low band keeps
    comparable proportions in both partitions, and the same split serves both
    the regression and the classification problem.

    Returns:
        ``(X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test)``.
    """
    frame = build_modelling_frame(data_path)

    features = frame[MODEL_FEATURES]
    y_regression = frame[TARGET]
    y_band = to_stress_band(y_regression)

    X_train, X_test, y_reg_train, y_reg_test, y_band_train, y_band_test = train_test_split(
        features,
        y_regression,
        y_band,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_band,
    )

    def to_codes(bands, index):
        return pd.Series(
            pd.Categorical(bands, categories=STRESS_BAND_LABELS, ordered=True).codes,
            index=index,
            name="Stress band",
        )

    return (
        X_train,
        X_test,
        y_reg_train,
        y_reg_test,
        to_codes(y_band_train, X_train.index),
        to_codes(y_band_test, X_test.index),
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train_regressor(X_train, y_train):
    """Re-run the tuned Random Forest search selected in section ML.12.

    Returns:
        ``(fitted_pipeline, cv_mae)`` where ``cv_mae`` is the cross-validated
        mean absolute error of the best configuration.
    """
    search = RandomizedSearchCV(
        build_model_pipeline(RandomForestRegressor(random_state=RANDOM_STATE)),
        param_distributions=FOREST_SEARCH_SPACE,
        n_iter=SEARCH_ITERATIONS,
        scoring="neg_mean_absolute_error",
        cv=RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, -search.best_score_


def train_classifier(X_train, y_train):
    """Re-run the tuned RBF SVM search selected in section ML.16.

    Balanced class weights counter the 5.7 : 1 band imbalance, and macro F1 is
    the search objective so that all three bands count equally.

    Returns:
        ``(fitted_pipeline, cv_f1_macro)``.
    """
    search = RandomizedSearchCV(
        build_model_pipeline(
            SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE)
        ),
        param_distributions=SVM_SEARCH_SPACE,
        n_iter=SEARCH_ITERATIONS,
        scoring="f1_macro",
        cv=RepeatedStratifiedKFold(
            n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
        ),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_score_


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate_regressor(model, cv_mae, X_train, y_train, X_test, y_test) -> dict:
    """Score the fitted regressor on the hold-out set."""
    test_prediction = model.predict(X_test)
    return {
        "cv_mae": float(cv_mae),
        "test_mae": float(mean_absolute_error(y_test, test_prediction)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_prediction))),
        "test_r2": float(r2_score(y_test, test_prediction)),
        "test_explained_variance": float(explained_variance_score(y_test, test_prediction)),
        "train_r2": float(r2_score(y_train, model.predict(X_train))),
    }


def evaluate_classifier(model, cv_f1, X_test, y_test) -> dict:
    """Score the fitted classifier on the hold-out set."""
    prediction = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    binarised = label_binarize(y_test, classes=range(len(STRESS_BAND_LABELS)))
    return {
        "cv_f1_macro": float(cv_f1),
        "test_accuracy": float(accuracy_score(y_test, prediction)),
        "test_precision_macro": float(
            precision_score(y_test, prediction, average="macro", zero_division=0)
        ),
        "test_recall_macro": float(
            recall_score(y_test, prediction, average="macro", zero_division=0)
        ),
        "test_f1_macro": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
        "test_f1_weighted": float(
            f1_score(y_test, prediction, average="weighted", zero_division=0)
        ),
        "test_roc_auc_ovr_macro": float(
            roc_auc_score(binarised, probabilities, multi_class="ovr", average="macro")
        ),
    }


def baseline_scores(X_train, y_reg_train, y_cls_train) -> tuple[float, float]:
    """Cross-validated scores of the two 'no model at all' baselines."""
    regression_cv = RepeatedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )
    classification_cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )
    mean_mae = -cross_val_score(
        build_model_pipeline(DummyRegressor(strategy="mean")),
        X_train,
        y_reg_train,
        cv=regression_cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    ).mean()
    majority_f1 = cross_val_score(
        build_model_pipeline(DummyClassifier(strategy="most_frequent")),
        X_train,
        y_cls_train,
        cv=classification_cv,
        scoring="f1_macro",
        n_jobs=-1,
    ).mean()
    return float(mean_mae), float(majority_f1)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def build_metadata(regression_metrics, classification_metrics, baselines, rows) -> dict:
    """Assemble the metadata document saved alongside the artefacts."""
    labelled, train_rows, test_rows = rows
    mean_mae, majority_f1 = baselines
    try:
        import xgboost

        xgboost_version = xgboost.__version__
    except ImportError:
        xgboost_version = None

    return {
        "project": "Academic Stress Analysis - predictive modelling",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source_dataset": "data/Academic Stress Dataset.csv",
        "rows_labelled": labelled,
        "rows_train": train_rows,
        "rows_test": test_rows,
        "random_state": RANDOM_STATE,
        "feature_names": MODEL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "engineered_features": ENGINEERED_NUMERIC,
        "excluded_features": {
            "temporal": [
                "Response Month",
                "Response Day",
                "Response DayOfWeek",
                "Response Hour",
                "Days Since First Response",
            ],
            "reason": "no cross-validated improvement; reflects survey timing, not the student",
        },
        "regression": {
            "target": TARGET,
            "target_scale": [LIKERT_MIN, LIKERT_MAX],
            "model": "Random Forest (tuned)",
            "artefact": "best_stress_regression_model.pkl",
            "primary_metric": "mean absolute error",
            "metrics": regression_metrics,
            "baseline_cv_mae": mean_mae,
        },
        "classification": {
            "target": f"{TARGET} banded",
            "band_edges": STRESS_BAND_EDGES,
            "band_labels": STRESS_BAND_LABELS,
            "band_rule": "rating 1-2 -> Low, 3 -> Moderate, 4-5 -> High",
            "class_codes": {label: index for index, label in enumerate(STRESS_BAND_LABELS)},
            "model": "SVM (RBF kernel) (tuned)",
            "artefact": "best_stress_classification_model.pkl",
            "primary_metric": "macro-averaged F1",
            "metrics": classification_metrics,
            "baseline_cv_f1_macro": majority_f1,
        },
        "library_versions": {
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost_version,
        },
        "intended_use": (
            "Educational / portfolio demonstration of a predictive modelling workflow on "
            "self-reported survey data. Not a clinical, diagnostic or screening instrument."
        ),
    }


def verify_against_committed(metadata: dict, reference_path: Path, tolerance: float = 5e-3) -> bool:
    """Compare freshly computed metrics with the committed ones.

    Args:
        metadata: The metadata document just built.
        reference_path: Path to the committed ``model_metadata.json``.
        tolerance: Absolute tolerance. Small differences are expected across
            library versions; a larger gap means the run did not reproduce.

    Returns:
        ``True`` if every metric matches within tolerance.
    """
    if not reference_path.exists():
        print(f"No reference metadata at {reference_path} - nothing to verify against.")
        return True

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    ok = True
    for problem in ("regression", "classification"):
        if reference[problem]["model"] != metadata[problem]["model"]:
            print(
                f"  ! {problem}: selected model differs - "
                f"{reference[problem]['model']!r} committed, {metadata[problem]['model']!r} now"
            )
            ok = False
        for metric, value in metadata[problem]["metrics"].items():
            expected = reference[problem]["metrics"].get(metric)
            if expected is None:
                continue
            delta = abs(value - expected)
            flag = "ok " if delta <= tolerance else "DIFF"
            if delta > tolerance:
                ok = False
            print(f"  {flag} {problem:<15} {metric:<26} {expected:.4f} -> {value:.4f}")
    return ok


def main(argv=None) -> int:
    """Entry point for ``python -m src.train``."""
    parser = argparse.ArgumentParser(
        description="Retrain and save the two models selected in the notebook.",
    )
    parser.add_argument(
        "--data", type=Path, default=DATA_PATH, metavar="CSV", help="survey CSV to train on"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR,
        metavar="DIR",
        help="where to write the artefacts (default: models/)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="compare the resulting metrics with models/model_metadata.json",
    )
    args = parser.parse_args(argv)

    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = build_split(args.data)
    print(f"Training rows {len(X_train)} | hold-out rows {len(X_test)} | features {len(MODEL_FEATURES)}")

    print("\nTuning the Random Forest regressor ...")
    regressor, cv_mae = train_regressor(X_train, y_reg_train)
    regression_metrics = evaluate_regressor(
        regressor, cv_mae, X_train, y_reg_train, X_test, y_reg_test
    )
    print(f"  CV MAE {regression_metrics['cv_mae']:.3f} | hold-out MAE {regression_metrics['test_mae']:.3f}")

    print("\nTuning the RBF SVM classifier ...")
    classifier, cv_f1 = train_classifier(X_train, y_cls_train)
    classification_metrics = evaluate_classifier(classifier, cv_f1, X_test, y_cls_test)
    print(
        f"  CV macro F1 {classification_metrics['cv_f1_macro']:.3f} | "
        f"hold-out macro F1 {classification_metrics['test_f1_macro']:.3f}"
    )

    print("\nScoring the baselines ...")
    baselines = baseline_scores(X_train, y_reg_train, y_cls_train)
    print(f"  mean predictor CV MAE {baselines[0]:.3f} | majority class CV macro F1 {baselines[1]:.3f}")

    metadata = build_metadata(
        regression_metrics,
        classification_metrics,
        baselines,
        (len(X_train) + len(X_test), len(X_train), len(X_test)),
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(regressor, output_dir / "best_stress_regression_model.pkl")
    joblib.dump(classifier, output_dir / "best_stress_classification_model.pkl")
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"\nSaved 3 artefacts to {output_dir.name}/")

    if args.verify:
        print("\nVerifying against the committed metrics:")
        if verify_against_committed(metadata, MODEL_DIR / "model_metadata.json"):
            print("\nAll metrics reproduce within tolerance.")
        else:
            print("\nSome metrics differ - see the DIFF lines above.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
