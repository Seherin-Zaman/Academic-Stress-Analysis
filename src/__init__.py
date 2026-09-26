"""Reusable components of the Academic Stress predictive-modelling project.

The notebook in ``notebooks/machine_learning.ipynb`` remains the primary
implementation: it contains the exploratory analysis, the model comparison, the
hyperparameter searches and the explainability work. This package holds only the
parts that are genuinely useful outside the notebook — the data-preparation code
that a saved pipeline needs at prediction time, and a thin loader around the
saved artefacts.

Both modules mirror the notebook logic exactly; ``tests/test_pipeline.py``
asserts that they reproduce the notebook's predictions.
"""

__all__ = ["preprocessing", "predict"]
