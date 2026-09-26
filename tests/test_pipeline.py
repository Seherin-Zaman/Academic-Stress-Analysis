"""Checks that `src/` reproduces the notebook's data preparation and results.

These are regression tests in the software sense: they pin the behaviour the
saved model artefacts depend on, so that a change to `src/preprocessing.py` or a
dependency upgrade that would silently alter predictions fails loudly instead.

Run with pytest, or directly::

    python -m pytest tests/ -v
    python tests/test_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import EXAMPLE_RECORDS, load_models, predict  # noqa: E402
from src.preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    DATA_PATH,
    MODEL_DIR,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    STRESS_BAND_LABELS,
    TARGET,
    build_modelling_frame,
    clean_dataset,
    engineer_features,
    load_raw_dataset,
    prepare_records_for_model,
    to_stress_band,
)

# Values established by the notebook run that produced the committed artefacts.
EXPECTED_RAW_ROWS = 140
EXPECTED_LABELLED_ROWS = 129
EXPECTED_BAND_COUNTS = {"Low": 14, "Moderate": 35, "High": 80}
EXPECTED_DEMO_SCORES = [2.64, 3.57, 4.64]
EXPECTED_DEMO_BANDS = ["Low", "Moderate", "High"]


# --------------------------------------------------------------------------- #
# Data loading and cleaning
# --------------------------------------------------------------------------- #


def test_dataset_is_present_and_has_expected_shape():
    assert DATA_PATH.exists(), f"dataset missing at {DATA_PATH}"
    raw = load_raw_dataset()
    assert len(raw) == EXPECTED_RAW_ROWS
    assert TARGET in raw.columns, "column rename did not produce the target name"


def test_cleaning_standardises_the_study_environment_labels():
    """The raw file mixes 'disrupted' with 'Peaceful'/'Noisy'; cleaning must fix it."""
    raw = load_raw_dataset()
    assert "disrupted" in set(raw["Study Environment"].dropna())

    cleaned = clean_dataset(raw)
    assert set(cleaned["Study Environment"].dropna()) == {"Peaceful", "Noisy", "Disrupted"}


def test_cleaning_does_not_impute_or_drop_rows():
    """Imputation belongs inside the pipeline; cleaning must not pre-empt it."""
    raw = load_raw_dataset()
    cleaned = clean_dataset(raw)
    assert len(cleaned) == len(raw), "cleaning must not drop rows"
    assert cleaned[TARGET].isna().sum() == 11, "cleaning must not impute the target"


def test_modelling_frame_keeps_only_labelled_rows():
    frame = build_modelling_frame()
    assert len(frame) == EXPECTED_LABELLED_ROWS
    assert frame[TARGET].notna().all()


def test_stress_bands_match_the_documented_distribution():
    frame = build_modelling_frame()
    counts = to_stress_band(frame[TARGET]).value_counts().to_dict()
    assert {k: int(v) for k, v in counts.items()} == EXPECTED_BAND_COUNTS


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


def test_engineered_features_are_row_wise():
    """A row's features must not depend on the other rows present.

    This is what makes it safe to engineer features before splitting: if the
    transformation were fitted on the dataset as a whole it would leak
    information between the training and hold-out partitions.
    """
    frame = build_modelling_frame()
    full = engineer_features(frame)
    subset = engineer_features(frame.head(10))

    engineered = ["Overall Pressure Score", "Peer x Home Pressure", "Environment Disturbance"]
    np.testing.assert_allclose(
        subset[engineered].to_numpy(dtype=float),
        full[engineered].head(10).to_numpy(dtype=float),
        err_msg="feature engineering depends on which other rows are present",
    )


def test_overall_pressure_score_is_the_mean_of_the_three_items():
    frame = build_modelling_frame()
    expected = frame[["Peer pressure", "Home Academic Pressure", "Academic Competition"]].mean(axis=1)
    np.testing.assert_allclose(frame["Overall Pressure Score"], expected)


def test_no_target_derived_column_reaches_the_feature_set():
    """The leakage guard that the notebook asserts at split time."""
    leaky = [c for c in MODEL_FEATURES if "stress" in c.lower()]
    assert not leaky, f"target-derived column in the feature set: {leaky}"


def test_feature_list_matches_the_saved_metadata():
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    assert metadata["feature_names"] == MODEL_FEATURES
    assert metadata["numeric_features"] == NUMERIC_FEATURES
    assert metadata["categorical_features"] == CATEGORICAL_FEATURES


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #


def test_raw_survey_records_are_prepared_without_manual_preprocessing():
    """Verbose survey wording and the lower-case 'disrupted' must both survive."""
    records = [dict(record) for record in EXAMPLE_RECORDS]
    for record in records:
        record.pop("Profile", None)

    features = prepare_records_for_model(records)
    assert list(features.columns) == MODEL_FEATURES
    assert len(features) == len(records)
    # The high-pressure profile submits "disrupted" - the highest disturbance level.
    assert features["Environment Disturbance"].iloc[2] == 2


@pytest.mark.skipif(
    not (MODEL_DIR / "best_stress_regression_model.pkl").exists(),
    reason="saved artefacts not present; run `python -m src.train` first",
)
def test_saved_models_reproduce_the_notebook_predictions():
    result = predict(EXAMPLE_RECORDS)

    np.testing.assert_allclose(
        result["Predicted stress score (1-5)"].to_numpy(dtype=float),
        EXPECTED_DEMO_SCORES,
        atol=0.01,
    )
    assert list(result["Predicted band"]) == EXPECTED_DEMO_BANDS


@pytest.mark.skipif(
    not (MODEL_DIR / "best_stress_regression_model.pkl").exists(),
    reason="saved artefacts not present; run `python -m src.train` first",
)
def test_band_probabilities_are_a_valid_distribution():
    result = predict(EXAMPLE_RECORDS)
    probabilities = result[[f"P({band})" for band in STRESS_BAND_LABELS]].to_numpy(dtype=float)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=0.01)
    assert (probabilities >= 0).all()


@pytest.mark.skipif(
    not (MODEL_DIR / "best_stress_regression_model.pkl").exists(),
    reason="saved artefacts not present; run `python -m src.train` first",
)
def test_metadata_metrics_are_present_and_in_range():
    _, _, metadata = load_models()
    regression = metadata["regression"]["metrics"]
    classification = metadata["classification"]["metrics"]

    # The selected model must beat the 'no model at all' baselines.
    assert regression["cv_mae"] < metadata["regression"]["baseline_cv_mae"]
    assert classification["cv_f1_macro"] > metadata["classification"]["baseline_cv_f1_macro"]

    assert 0.0 <= classification["test_roc_auc_ovr_macro"] <= 1.0
    assert 0.0 <= classification["test_f1_macro"] <= 1.0


def test_no_absolute_path_leaked_into_the_metadata():
    """The metadata is committed, so it must not carry a machine-specific path."""
    text = (MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8")
    for marker in ("C:\\\\", "C:/", "/home/", "/Users/", "OneDrive"):
        assert marker not in text, f"absolute path fragment {marker!r} in model_metadata.json"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
