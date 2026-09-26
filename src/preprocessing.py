"""Data loading, cleaning and feature engineering for the academic stress dataset.

This module is a faithful extraction of sections ML.2, ML.4 and ML.6 of
``notebooks/machine_learning.ipynb``. It is deliberately limited to the
*deterministic, row-wise* part of the pipeline:

* renaming and type-standardising the raw survey export;
* mapping the free-text survey labels onto short canonical categories;
* deriving the composite, interaction and ordinal features.

Everything that has to be *learned from data* — imputation, scaling and one-hot
encoding — lives inside the fitted scikit-learn ``Pipeline`` stored in
``models/``, so it is fitted on training folds only and can never leak
information from the hold-out set. Keeping that boundary is the whole reason
this module stops where it does.

Because every transformation here depends only on a row's own values, the same
functions serve the training data and a single new survey response.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Project layout
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Academic Stress Dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"

# --------------------------------------------------------------------------- #
# Column vocabulary and scale definition
# --------------------------------------------------------------------------- #

#: Verbose survey headers mapped to the short names used throughout the project.
COLUMN_RENAME = {
    "Your Academic Stage": "Academic Stage",
    "Academic pressure from your home": "Home Academic Pressure",
    "What coping strategy you use as a student?": "Coping Strategy",
    "Do you have any bad habits like smoking, drinking on a daily basis?": "Bad Habits",
    "What would you rate the academic  competition in your student life": "Academic Competition",
    "Rate your academic stress index": "Academic Stress Index",
}

TARGET = "Academic Stress Index"
LIKERT_MIN, LIKERT_MAX = 1, 5
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M"

#: The three predictor rating items (the fourth Likert item is the target).
RATING_ITEMS = ["Peer pressure", "Home Academic Pressure", "Academic Competition"]

#: Lower-cased survey text mapped to the canonical analysis label.
CATEGORY_CANONICAL_MAP = {
    "Academic Stage": {
        "high school": "High School",
        "undergraduate": "Undergraduate",
        "post-graduate": "Post-Graduate",
    },
    "Study Environment": {
        "peaceful": "Peaceful",
        "noisy": "Noisy",
        "disrupted": "Disrupted",  # fixes the lower-case inconsistency in the source file
    },
    "Coping Strategy": {
        "analyze the situation and handle it with intellect": "Analytical",
        "emotional breakdown (crying a lot)": "Emotional Breakdown",
        "social support (friends, family)": "Social Support",
    },
    "Bad Habits": {
        "no": "No",
        "yes": "Yes",
        "prefer not to say": "Prefer not to say",
    },
}

#: Study environments in order of increasing disturbance.
ENVIRONMENT_DISTURBANCE_ORDER = {"Peaceful": 0, "Noisy": 1, "Disrupted": 2}

#: Coping strategies treated as distress-driven rather than adaptive.
MALADAPTIVE_COPING = {"Emotional Breakdown"}

# --------------------------------------------------------------------------- #
# Feature inventory (must match models/model_metadata.json)
# --------------------------------------------------------------------------- #

ENGINEERED_NUMERIC = [
    "Overall Pressure Score",
    "Peer x Home Pressure",
    "Competition x Home Pressure",
    "Environment Disturbance",
    "Maladaptive Coping",
]

NUMERIC_FEATURES = RATING_ITEMS + ENGINEERED_NUMERIC

CATEGORICAL_FEATURES = [
    "Academic Stage",
    "Study Environment",
    "Coping Strategy",
    "Bad Habits",
]

#: The exact feature list, in order, that the saved pipelines expect.
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: Classification band definition (section ML.14): 1-2 Low, 3 Moderate, 4-5 High.
STRESS_BAND_EDGES = [LIKERT_MIN - 1, 2, 3, LIKERT_MAX]
STRESS_BAND_LABELS = ["Low", "Moderate", "High"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_raw_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the survey export and apply header-level tidying only.

    No row-level cleaning happens here, so a data-quality assessment run on the
    result describes the data exactly as it arrives from the source file.

    Args:
        path: Location of the survey CSV. Defaults to the copy in ``data/``.

    Returns:
        The raw responses with whitespace stripped from the headers and the
        short column names applied.
    """
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    return frame.rename(columns=COLUMN_RENAME)


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #


def clean_dataset(frame: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Standardise types, category labels and value ranges.

    Deliberately does **not** impute, deduplicate or trim outliers:

    * imputation is a learned transformation and belongs inside the
      cross-validated pipeline, fitted on training folds only;
    * identical answers to a short Likert survey are unremarkable, so repeated
      answer patterns are genuine observations rather than errors;
    * every rating is a valid point on a 1-5 opinion scale, so no value is an
      out-of-range measurement error to remove.

    Only columns actually present are touched, so the same function serves the
    full training frame and a single unlabelled record at prediction time.

    Args:
        frame: Raw survey data with the short column names applied.
        verbose: Print a line for each unrecognised label or out-of-scale value.

    Returns:
        A cleaned copy. The input frame is not modified.
    """
    cleaned = frame.copy()

    if "Timestamp" in cleaned.columns:
        cleaned["Timestamp"] = pd.to_datetime(
            cleaned["Timestamp"], format=TIMESTAMP_FORMAT, errors="coerce"
        )

    for column, mapping in CATEGORY_CANONICAL_MAP.items():
        if column not in cleaned.columns:
            continue
        standardised = cleaned[column].astype("string").str.strip().str.lower()
        known = standardised.dropna()
        unmapped = sorted(set(known[~known.isin(mapping)]))
        if verbose and unmapped:
            print(f"  ! {column}: unrecognised labels set to NaN -> {unmapped}")
        cleaned[column] = standardised.map(mapping)

    for column in RATING_ITEMS + [TARGET]:
        if column not in cleaned.columns:
            continue
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        out_of_scale = ~cleaned[column].between(LIKERT_MIN, LIKERT_MAX) & cleaned[column].notna()
        if verbose and int(out_of_scale.sum()):
            print(f"  ! {column}: {int(out_of_scale.sum())} out-of-scale value(s) set to NaN")
        cleaned.loc[out_of_scale, column] = np.nan

    return cleaned


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the composite, interaction and ordinal features used by the models.

    Every feature is a row-wise transformation of that row's own values, so the
    function learns nothing from the dataset as a whole and cannot leak
    information between the training and hold-out partitions.

    The features are:

    ``Overall Pressure Score``
        Mean of the three pressure ratings — a total pressure-load index that is
        more stable than any single item because rating noise partly cancels.
    ``Peer x Home Pressure`` / ``Competition x Home Pressure``
        Interaction terms testing whether pressure sources compound.
    ``Environment Disturbance``
        Ordinal encoding that preserves the natural Peaceful < Noisy < Disrupted
        ordering a one-hot encoding would discard.
    ``Maladaptive Coping``
        Binary indicator isolating the distress-driven coping response.

    Args:
        frame: A cleaned frame (see :func:`clean_dataset`).

    Returns:
        A copy with the engineered columns appended.
    """
    out = frame.copy()

    out["Overall Pressure Score"] = out[RATING_ITEMS].mean(axis=1)
    out["Peer x Home Pressure"] = out["Peer pressure"] * out["Home Academic Pressure"]
    out["Competition x Home Pressure"] = (
        out["Academic Competition"] * out["Home Academic Pressure"]
    )
    out["Environment Disturbance"] = out["Study Environment"].map(ENVIRONMENT_DISTURBANCE_ORDER)
    out["Maladaptive Coping"] = out["Coping Strategy"].map(
        lambda value: np.nan if pd.isna(value) else float(value in MALADAPTIVE_COPING)
    )

    return out


def to_stress_band(values: pd.Series) -> pd.Series:
    """Map the 1-5 stress rating onto the ordered Low / Moderate / High bands."""
    return pd.cut(values, bins=STRESS_BAND_EDGES, labels=STRESS_BAND_LABELS, ordered=True)


# --------------------------------------------------------------------------- #
# End-to-end preparation
# --------------------------------------------------------------------------- #


def prepare_records_for_model(records) -> pd.DataFrame:
    """Turn raw survey-format records into the exact input the pipelines expect.

    This is the single code path used for both training data and new responses:
    ``raw record -> rename -> clean -> engineer -> select MODEL_FEATURES``.
    Imputation, scaling and encoding are then applied by the fitted pipeline
    itself, so they can never drift out of sync with the model.

    Args:
        records: Anything ``pandas.DataFrame`` accepts — typically a list of
            dicts keyed by the original, verbose survey questions. Short column
            names are also accepted, since the rename is a no-op for them.

    Returns:
        A frame containing exactly ``MODEL_FEATURES``, in order.
    """
    frame = pd.DataFrame(records).rename(columns=COLUMN_RENAME)
    frame = clean_dataset(frame)
    frame = engineer_features(frame)
    return frame[MODEL_FEATURES]


def build_modelling_frame(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load, clean and engineer the dataset, keeping only rows that carry a label.

    Rows with no stress rating are dropped rather than imputed: fabricating a
    target would inflate every metric computed against it. This reproduces the
    129-row modelling frame used in the notebook.

    Args:
        path: Location of the survey CSV.

    Returns:
        The labelled, cleaned and feature-engineered modelling frame.
    """
    frame = load_raw_dataset(path)
    frame = clean_dataset(frame)
    frame = frame.dropna(subset=[TARGET]).reset_index(drop=True)
    return engineer_features(frame)
