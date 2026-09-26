"""Load the saved pipelines and predict academic stress for new survey responses.

The artefacts in ``models/`` are complete scikit-learn ``Pipeline`` objects, not
bare estimators: each one carries its own fitted imputer, scaler and one-hot
encoder. A reloaded artefact therefore accepts raw survey records directly and
cannot drift out of sync with a separately stored preprocessing step.

Command line
------------
Predict for the three example students used in section ML.22 of the notebook::

    python -m src.predict --demo

Predict for your own responses (a JSON file holding one object or a list of
objects, keyed by either the verbose survey questions or the short names)::

    python -m src.predict --input my_responses.json

Scope
-----
The models estimate a *self-reported* stress rating from seven survey answers.
They are a portfolio demonstration of a modelling workflow, not a clinical,
diagnostic or screening instrument, and no output should be used to make a
consequential decision about an individual.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

# Support both `python -m src.predict` and a direct `python src/predict.py` run.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.preprocessing import (  # noqa: E402
        MODEL_DIR,
        STRESS_BAND_LABELS,
        prepare_records_for_model,
    )
else:
    from .preprocessing import MODEL_DIR, STRESS_BAND_LABELS, prepare_records_for_model

REGRESSION_ARTEFACT = MODEL_DIR / "best_stress_regression_model.pkl"
CLASSIFICATION_ARTEFACT = MODEL_DIR / "best_stress_classification_model.pkl"
METADATA_ARTEFACT = MODEL_DIR / "model_metadata.json"

#: The three profiles demonstrated in section ML.22, in raw survey format.
EXAMPLE_RECORDS = [
    {
        "Profile": "Low-pressure student",
        "Your Academic Stage": "undergraduate",
        "Peer pressure": 1,
        "Academic pressure from your home": 2,
        "Study Environment": "Peaceful",
        "What coping strategy you use as a student?": "Social support (friends, family)",
        "Do you have any bad habits like smoking, drinking on a daily basis?": "No",
        "What would you rate the academic  competition in your student life": 2,
    },
    {
        "Profile": "Moderate-pressure student",
        "Your Academic Stage": "undergraduate",
        "Peer pressure": 3,
        "Academic pressure from your home": 3,
        "Study Environment": "Noisy",
        "What coping strategy you use as a student?": (
            "Analyze the situation and handle it with intellect"
        ),
        "Do you have any bad habits like smoking, drinking on a daily basis?": "No",
        "What would you rate the academic  competition in your student life": 3,
    },
    {
        "Profile": "High-pressure student",
        "Your Academic Stage": "high school",
        "Peer pressure": 5,
        "Academic pressure from your home": 5,
        "Study Environment": "disrupted",
        "What coping strategy you use as a student?": "Emotional breakdown (crying a lot)",
        "Do you have any bad habits like smoking, drinking on a daily basis?": "Yes",
        "What would you rate the academic  competition in your student life": 5,
    },
]


def load_models(model_dir: Path | str = MODEL_DIR):
    """Load both saved pipelines and their metadata.

    Args:
        model_dir: Directory holding the ``.pkl`` artefacts and metadata file.

    Returns:
        A ``(regressor, classifier, metadata)`` tuple.

    Raises:
        FileNotFoundError: If an artefact is missing, with a pointer to the
            notebook section that regenerates it.
    """
    model_dir = Path(model_dir)
    paths = {
        "regression model": model_dir / REGRESSION_ARTEFACT.name,
        "classification model": model_dir / CLASSIFICATION_ARTEFACT.name,
        "metadata": model_dir / METADATA_ARTEFACT.name,
    }
    missing = [f"{label} ({path.name})" for label, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing artefact(s) in "
            f"{model_dir}: {', '.join(missing)}. "
            "Run notebooks/machine_learning.ipynb through section ML.23 to regenerate them."
        )

    regressor = joblib.load(paths["regression model"])
    classifier = joblib.load(paths["classification model"])
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    return regressor, classifier, metadata


def predict(records, model_dir: Path | str = MODEL_DIR) -> pd.DataFrame:
    """Predict a stress score and a stress band for each raw survey record.

    Args:
        records: One dict, or a list of dicts, in raw survey format.
        model_dir: Directory holding the saved artefacts.

    Returns:
        A frame with the predicted 1-5 score, the predicted band, the
        confidence in that band, and the full per-band probabilities.
    """
    if isinstance(records, dict):
        records = [records]

    records = [dict(record) for record in records]
    profiles = [record.pop("Profile", f"Record {i + 1}") for i, record in enumerate(records)]

    regressor, classifier, _ = load_models(model_dir)
    features = prepare_records_for_model(records)

    scores = regressor.predict(features)
    bands = classifier.predict(features)
    probabilities = classifier.predict_proba(features)

    # The reported confidence is the probability of the band that was actually
    # predicted, not the largest probability in the row. For an SVC those can
    # differ: `predict` uses libsvm's voting over the decision function while
    # `predict_proba` uses a separately fitted Platt calibration, so near a band
    # boundary they can disagree. Surfacing the most probable band alongside the
    # prediction makes that disagreement visible instead of hiding it in a max().
    result = pd.DataFrame(
        {
            "Predicted stress score (1-5)": scores.round(2),
            "Predicted band": [STRESS_BAND_LABELS[code] for code in bands],
            "Confidence in predicted band": [
                f"{row[code]:.0%}" for row, code in zip(probabilities, bands)
            ],
            "Most probable band": [
                STRESS_BAND_LABELS[row.argmax()] for row in probabilities
            ],
        },
        index=pd.Index(profiles, name="Profile"),
    )
    for position, band in enumerate(STRESS_BAND_LABELS):
        result[f"P({band})"] = probabilities[:, position].round(3)
    return result


def main(argv=None) -> int:
    """Entry point for ``python -m src.predict``."""
    parser = argparse.ArgumentParser(
        description="Predict academic stress from survey responses using the saved models.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--demo",
        action="store_true",
        help="predict for the three example students from notebook section ML.22",
    )
    source.add_argument(
        "--input",
        type=Path,
        metavar="FILE",
        help="JSON file holding one survey record, or a list of records",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=MODEL_DIR,
        metavar="DIR",
        help="directory holding the saved artefacts (default: models/)",
    )
    args = parser.parse_args(argv)

    if args.demo:
        records = EXAMPLE_RECORDS
    else:
        records = json.loads(args.input.read_text(encoding="utf-8"))

    _, _, metadata = load_models(args.models)
    print(f"Regression model     : {metadata['regression']['model']}")
    print(f"Classification model : {metadata['classification']['model']}")
    print(f"Band rule            : {metadata['classification']['band_rule']}\n")
    print(predict(records, args.models).to_string())
    print(f"\n{metadata['intended_use']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
