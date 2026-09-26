# Academic Stress Analysis

*Predicting self-reported student stress from a wellbeing survey — an
end-to-end machine learning project.*

An end-to-end supervised learning project on a 140-response student wellbeing
survey: exploratory analysis, leakage-safe preprocessing, feature engineering,
a regression and a classification problem, model comparison against baselines,
hyperparameter tuning, explainability, formal statistical testing, and saved
inference-ready pipelines.

The headline result is deliberately modest and stated as measured: the survey
items carry a **real but moderate** signal about a student's self-reported
stress. Much of the value in this project is in the parts that establish *how
much* signal there is and *why it is not more* — baselines, cross-validation
variance, a learning curve, and significance testing that overturns one of the
project's own earlier descriptive conclusions.

---

## Overview

**The problem.** Academic stress affects student wellbeing and attainment, but
the factors around it — peer pressure, pressure from home, competition, study
environment, coping style, lifestyle habits — are usually discussed
qualitatively. This project asks a quantitative question: given a short survey,
**how much of a student's self-reported stress level can actually be predicted,
and which factors carry that signal?**

**Why it matters.** A well-calibrated answer tells a student services team where
limited intervention budget is best directed — and, just as usefully, where the
data does *not* support a confident claim. A project that reports "study
environment is the only categorical factor whose effect survives correction for
multiple comparisons" is more actionable than one that reports six unqualified
correlations.

**What the system does.** From seven survey answers it produces two predictions:
a continuous stress score on the 1–5 scale, and a Low / Moderate / High band with
calibrated per-band probabilities. Both are served by complete scikit-learn
`Pipeline` objects that carry their own fitted imputer, scaler and encoder, so a
raw survey record can be scored directly with no manual preprocessing.

---

## Key Features

* **Data quality assessment** — a reusable profiling function run *before* and
  *after* cleaning, surfacing missingness, duplicate answer patterns,
  out-of-scale ratings and inconsistent category labels.
* **Cleaning as an idempotent function** — the same `clean_dataset()` serves the
  training data and a single new record, so training and inference cannot drift
  apart.
* **Feature engineering with stated rationale** — five features (a composite
  pressure score, two interaction terms, an ordinal environment encoding and a
  coping-style risk flag), each justified, plus an explicit list of features
  deliberately *not* created and why.
* **An empirical feature-selection decision** — six timestamp features were
  built, measured against fold-to-fold noise, found not to earn their place, and
  dropped, with the reasoning recorded.
* **Leakage-safe design** — imputation, scaling and encoding live inside the
  `Pipeline` and are re-fitted per CV fold; target-derived columns are excluded
  and guarded by a runtime `assert`.
* **Two complementary ML problems** — regression on the 1–5 rating and
  three-band classification, sharing one stratified split so they stay
  comparable.
* **Baselines first** — a mean predictor and a majority-class predictor, so every
  model is judged against "no model at all" rather than in the abstract.
* **Model comparison** — 8 regressors and 7 classifiers under identical repeated
  cross-validation, plus hyperparameter tuning of the leading families.
* **Explainability** — impurity importance, permutation importance and SHAP
  (global beeswarm plus an individual waterfall), with the collinearity caveat
  applied to their interpretation.
* **Statistical testing** — Spearman correlations and Kruskal–Wallis group
  comparisons with epsilon-squared effect sizes and a Bonferroni correction.
* **Robustness checks** — generalisation gaps, cross-validation variance and a
  learning curve, used to argue that *data volume*, not algorithm choice, is the
  binding constraint.
* **Saved, verified artefacts** — both pipelines persisted with `joblib` beside a
  metadata file, and reloaded in-notebook to assert the predictions round-trip.
* **A Tkinter dashboard** — an interactive statistical summary view (Section 5),
  opened on demand so it never blocks an automated notebook run.

---

## Dataset

**Student Academic Stress — Real World Dataset**, published on Kaggle by
`poushal02`
([link](https://www.kaggle.com/datasets/poushal02/student-academic-stress-real-world-dataset)).

| | |
| --- | --- |
| Responses | 140 (129 usable — 11 carry no stress rating) |
| Columns | 9 (1 timestamp, 4 Likert ratings, 4 categorical) |
| Collected | 24 July – 18 August 2025 |
| Target | `Academic Stress Index`, self-reported 1–5 |
| Missing cells | 55 (~4.4%) |

**Predictors:** peer pressure, home academic pressure and academic competition
(each 1–5); academic stage, study environment, coping strategy and bad habits
(categorical).

**Class balance** for the banded target: High 80, Moderate 35, Low 14 — an
imbalance of roughly 5.7 : 1.

Full column documentation, data-quality notes and an important **licensing
caveat about redistributing the CSV** are in [`data/README.md`](data/README.md).

---

## Machine Learning Pipeline

```
data/Academic Stress Dataset.csv
  → data quality assessment        profile before/after, no assumptions
  → cleaning                       types, canonical labels, scale validation
                                   (no imputation here — that would leak)
  → feature engineering            5 row-wise features; 6 temporal features
                                   built, tested, and dropped on the evidence
  → drop unlabelled rows           140 → 129 (targets are never imputed)
  → stratified train/test split    96 train / 33 hold-out, stratified by band
  → Pipeline preprocessing         median + most-frequent imputation,
                                   StandardScaler, OneHotEncoder
                                   ── re-fitted inside every CV fold ──
  → baselines                      mean predictor / majority class
  → model training & comparison    8 regressors, 7 classifiers, repeated CV
  → hyperparameter tuning          GridSearchCV + RandomizedSearchCV
  → diagnostics & explainability   residuals, confusion matrix, ROC, PR,
                                   permutation importance, SHAP
  → statistical testing            Spearman, Kruskal–Wallis, ε², Bonferroni
  → robustness checks              learning curve, CV variance, overfit gaps
  → final hold-out evaluation      test set used exactly once, at the end
  → saved pipelines                models/*.pkl + model_metadata.json
```

Cross-validation is **5-fold repeated 3 times** (45 fits per model), chosen
because a single split on 96 rows is dominated by fold-assignment luck. The
hold-out set takes no part in any selection decision.

---

## Models

### Regression — predicting the 1–5 stress score

Selection metric: **cross-validated MAE** (declared before fitting).

| Model | CV MAE ↓ | CV MAE (sd) | CV R² | Test MAE | Test RMSE | Test R² | Train R² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Random Forest (tuned)** | **0.637** | — | — | **0.697** | **0.840** | **0.408** | 0.654 |
| Ridge Regression (tuned) | 0.637 | — | — | — | — | — | — |
| Ridge Regression | 0.647 | 0.109 | 0.229 | 0.712 | 0.951 | 0.241 | 0.553 |
| Lasso Regression | 0.651 | 0.103 | 0.274 | 0.750 | 0.916 | 0.295 | 0.491 |
| Linear Regression *(baseline)* | 0.660 | 0.114 | 0.161 | 0.721 | 0.982 | 0.190 | 0.557 |
| Random Forest | 0.681 | 0.080 | 0.189 | 0.729 | 0.862 | 0.376 | 0.907 |
| Gradient Boosting | 0.712 | 0.087 | 0.030 | 0.744 | 0.904 | 0.315 | 0.906 |
| XGBoost | 0.726 | 0.082 | −0.025 | 0.700 | 0.854 | 0.388 | 0.917 |
| Decision Tree | 0.770 | 0.156 | −0.348 | 0.894 | 1.125 | −0.061 | 0.995 |
| Mean predictor *(baseline)* | 0.823 | 0.144 | −0.131 | 0.878 | 1.093 | −0.002 | 0.000 |

### Classification — Low / Moderate / High band

Selection metric: **cross-validated macro F1** (accuracy is reported but
explicitly not used for selection — see Results).

| Model | CV macro F1 ↑ | CV F1 (sd) | Test acc. | Test prec. (macro) | Test recall (macro) | Test macro F1 | Test ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **SVM RBF (tuned)** | **0.610** | — | **0.545** | **0.578** | **0.535** | **0.539** | **0.720** |
| XGBoost (tuned) | 0.603 | — | — | — | — | — | — |
| XGBoost | 0.603 | 0.111 | 0.636 | 0.667 | 0.591 | 0.617 | 0.674 |
| Random Forest | 0.593 | 0.134 | 0.576 | 0.578 | 0.578 | 0.578 | 0.739 |
| SVM (RBF kernel) | 0.592 | 0.089 | 0.606 | 0.622 | 0.589 | 0.586 | 0.715 |
| Gradient Boosting | 0.585 | 0.091 | 0.576 | 0.557 | 0.557 | 0.556 | 0.742 |
| Decision Tree | 0.544 | 0.117 | 0.515 | 0.549 | 0.544 | 0.543 | 0.628 |
| Logistic Regression *(baseline)* | 0.531 | 0.100 | 0.636 | 0.629 | 0.631 | 0.630 | 0.695 |
| Majority class *(baseline)* | 0.256 | 0.003 | 0.606 | 0.202 | 0.333 | 0.252 | 0.500 |

---

## Results

### Regression — Random Forest (tuned)

| Metric | Value |
| --- | --- |
| Cross-validated MAE | **0.637** stress points |
| Mean-predictor baseline (CV MAE) | 0.823 |
| **Error reduction vs baseline** | **22.6%** |
| Hold-out MAE | 0.697 |
| Hold-out RMSE | 0.840 |
| Hold-out R² | 0.408 |
| Train R² | 0.654 |

Tuning capped tree depth at 3, which cut the training R² from 0.91 to 0.65 while
*improving* the cross-validated score — removing memorisation, not capacity.

**A caveat that belongs next to the headline.** The tuned Ridge model scores
0.6372 against the forest's 0.6367 — a gap of **0.0005**, against fold-to-fold
standard deviation of about **0.11**. The selection rule picks the top of the
list deterministically, but the honest reading is that a regularised linear
model and a depth-constrained forest are **equivalent here**. Six of the ten
regression candidates sit within one CV standard deviation of the winner.

### Classification — SVM with RBF kernel (tuned)

| Metric | Value | Majority-class baseline |
| --- | --- | --- |
| Cross-validated macro F1 | **0.610** | 0.256 |
| Hold-out macro F1 | 0.539 | 0.252 |
| Hold-out ROC-AUC (OvR) | 0.720 | 0.500 |
| Hold-out accuracy | 0.545 | 0.606 |
| Hold-out weighted F1 | 0.564 | — |

**On that accuracy figure.** The selected classifier's hold-out accuracy (0.545)
is *below* the majority-class baseline (0.606). That is not a failure hidden in
the table — it is the trade the design makes on purpose. The baseline achieves
its accuracy by never identifying a single non-High student, which would be
useless for any plausible application. Balanced class weights and a macro-F1
objective ask the model to find the Low and Moderate students instead, and on a
5.7 : 1 imbalance those two goals genuinely conflict. Macro F1 (0.61 vs 0.26)
and ROC-AUC (0.72 vs 0.50) are the metrics that reflect what was asked for.

Errors are also the benign kind for an ordinal target: **14 of the 15 hold-out
misclassifications are between adjacent bands**, with almost no Low↔High
confusion.

### What predicts stress

Consistent across impurity importance, permutation importance and SHAP:

1. **The pressure block** — the composite `Overall Pressure Score` first, then
   the individual pressure ratings and their interactions. Because these are
   collinear by construction, the defensible claim is that the *block as a whole*
   drives predictions, not any single question.
2. **Study environment** — a consistent but modest contribution, and the only
   categorical factor whose group differences survive Bonferroni correction
   (Kruskal–Wallis p ≈ 0.010, ε² ≈ 0.06). Mean stress runs 4.03 in disrupted
   environments against 3.43 in peaceful ones.
3. **Everything else** — academic stage, coping strategy and reported habits sit
   at or near zero.

**A finding that corrects the project's own earlier analysis.** The descriptive
section (Part I) reported that stress differs by academic stage — high school
3.83 > post-graduate 3.73 > undergraduate 3.67. Formal testing shows that
ordering is **not statistically detectable** (p ≈ 0.75, ε² ≈ 0), and the models
independently assign academic stage almost no weight. With only 11 post-graduate
respondents, the apparent pattern is what sampling variation produces. The
notebook keeps both results and explains the correction rather than quietly
deleting the earlier claim.

**Association, not causation.** Every result comes from one cross-sectional
self-report survey. A disrupted environment may raise stress; stressed students
may perceive their surroundings as more disruptive; or a third factor may drive
both. The design cannot distinguish these.

---

## Project Structure

```
academic-stress-analysis/
├── README.md                     This file
├── requirements.txt              Pinned dependencies (see the scikit-learn note)
├── LICENSE                       MIT — covers the code, not the dataset
├── .gitignore
│
├── data/
│   ├── README.md                 Columns, quality notes, licensing caveat
│   └── Academic Stress Dataset.csv
│
├── notebooks/
│   └── machine_learning.ipynb    Primary implementation (202 cells)
│
├── models/
│   ├── best_stress_regression_model.pkl       Random Forest (tuned), ~789 KB
│   ├── best_stress_classification_model.pkl   SVM RBF (tuned), ~25 KB
│   └── model_metadata.json                    Features, metrics, versions
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          Loading, cleaning, feature engineering
│   ├── train.py                  Regenerate the saved artefacts
│   └── predict.py                Load the pipelines and score new records
│
├── tests/
│   └── test_pipeline.py          Checks src reproduces the notebook's results
│
├── reports/
│   └── figures/                  Key charts as PNG
│
└── docs/
    └── project_notes.md          Condensed technical write-up
```

**On the split between the notebook and `src/`.** The notebook is the primary
implementation and deliberately stays that way — the model comparison, tuning,
diagnostics and explainability are the analysis, and duplicating them into
scripts would create two versions to keep in step. `src/` holds only what is
genuinely useful outside a notebook: the deterministic data preparation an
inference call needs, a loader for the saved pipelines, and a training script
that regenerates the artefacts. `src/train.py --verify` reproduces all 13
committed metrics exactly.

---

## Installation

```bash
git clone https://github.com/Seherin-Zaman/academic-stress-analysis.git
cd academic-stress-analysis

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**Python 3.10–3.13.** Developed and verified on Python 3.13.5.

> **scikit-learn is pinned to 1.6.1 on purpose.** The pipelines in `models/`
> were serialised with that version, and a pickled scikit-learn object is only
> guaranteed to load under the version that wrote it — loading them under 1.8.0,
> for instance, fails with
> `AttributeError: Can't get attribute '_RemainderColsList'`. If you prefer a
> newer scikit-learn, install it and regenerate the artefacts with
> `python -m src.train`; the notebook itself runs fine either way.

The Tkinter dashboard in Section 5 needs Tk, which ships with standard Python on
Windows and macOS. On Debian/Ubuntu install `python3-tk`. The dashboard is
disabled by default (`LAUNCH_GUI = False`) so it never blocks a top-to-bottom run.

---

## Running the Project

### 1. Run the notebook (the full analysis)

```bash
jupyter lab notebooks/machine_learning.ipynb
```

The notebook is committed with its outputs, so it can be read without running
anything. It is ~1.9 MB (mostly embedded plot images), which GitHub's inline
viewer sometimes declines to render — in that case open it via
[nbviewer](https://nbviewer.org/github/Seherin-Zaman/academic-stress-analysis/blob/main/notebooks/machine_learning.ipynb).

Run every cell from the top on a fresh kernel — roughly **3 minutes** end to end.
Paths are resolved by locating the repository root at runtime, so the notebook
works from any working directory after a clone. It writes to `models/` and
`reports/figures/`; the source CSV is only ever read.

### 2. Predict with the saved models

```bash
python -m src.predict --demo                   # the three example students
python -m src.predict --input my_records.json  # your own survey records
```

`my_records.json` holds one object, or a list of objects, keyed by either the
original survey questions or the short column names:

```json
[
  {
    "Your Academic Stage": "undergraduate",
    "Peer pressure": 4,
    "Academic pressure from your home": 5,
    "Study Environment": "Noisy",
    "What coping strategy you use as a student?": "Social support (friends, family)",
    "Do you have any bad habits like smoking, drinking on a daily basis?": "No",
    "What would you rate the academic  competition in your student life": 4
  }
]
```

Or from Python:

```python
import joblib
from src.preprocessing import prepare_records_for_model

model = joblib.load("models/best_stress_regression_model.pkl")
model.predict(prepare_records_for_model([record]))
```

The saved object is the whole pipeline, so it accepts raw survey values —
including the lower-case `disrupted` spelling and the full-length coping-strategy
wording — and imputes, scales and encodes them itself.

### 3. Retrain the models

```bash
python -m src.train                                   # rebuild models/
python -m src.train --output-dir /tmp/m --verify      # rebuild elsewhere and compare
```

### 4. Run the tests

```bash
python -m pytest tests/ -v      # or: python tests/test_pipeline.py
```

---

## Technologies

| Tool | Role |
| --- | --- |
| **Python 3.13** | Language |
| **pandas**, **NumPy** | Data handling and numerical work |
| **scikit-learn** | Pipelines, preprocessing, models, CV, tuning, metrics |
| **XGBoost** | Gradient-boosted trees in both comparisons *(optional)* |
| **SHAP** | Per-prediction feature attribution *(optional)* |
| **SciPy** | Spearman correlation, Kruskal–Wallis testing |
| **Matplotlib**, **seaborn** | Visualisation |
| **joblib** | Pipeline serialisation |
| **Tkinter** | Interactive statistics dashboard (Section 5) |
| **requests** | External-module demonstration (Section 2) |
| **Jupyter** | Notebook environment |

---

## Results / Visualisations

Key charts are saved to [`reports/figures/`](reports/figures/) when the notebook
runs:

| File | Shows |
| --- | --- |
| `eda_correlation_and_target_distribution.png` | Spearman matrix across the ordinal items; target distribution |
| `regression_model_comparison.png` | CV MAE per model with error bars; train-vs-CV R² overfitting check |
| `regression_diagnostics.png` | Actual vs predicted, residuals vs fitted, error distribution |
| `stress_band_class_distribution.png` | The 5.7 : 1 class imbalance |
| `classification_diagnostics.png` | Confusion matrix, ROC curves, precision–recall curves |
| `stress_by_categorical_factor.png` | Stress distribution per categorical factor with test statistics |
| `robustness_learning_curve_and_cv_spread.png` | Learning curve and per-model CV spread |

The SHAP beeswarm and waterfall plots are rendered inline in the notebook.

---

## Limitations

Stated plainly, because they bound every number above.

1. **Sample size.** 129 labelled responses, split 96 / 33. Every metric carries
   wide uncertainty. Only **four low-stress students** are in the hold-out set,
   so per-band results for that class are indicative, not measured — one
   prediction moves its recall by 25 percentage points.
2. **The learning curve has not flattened.** Validation error still falls as rows
   are added (0.68 at 22 rows → 0.63 at 76). **More data, not a better
   algorithm, is the binding constraint.**
3. **Model differences are inside the noise.** Fold-to-fold standard deviation is
   about 0.11 MAE while the spread across leading models is about 0.01. The
   leaderboard identifies a *group of equivalent models*, not a decisive winner.
4. **Self-reported target.** A single subjective 1–5 item with no external
   validation, subject to the usual self-report biases. The models predict a
   *reported rating*, not a measured psychological state.
5. **Cross-sectional design.** One moment in time; no causal claim is possible.
6. **Sampling and coverage.** ~4 weeks, one distribution channel, dominated by
   undergraduates (100 of 140) with only 11 post-graduates. Findings may not
   transfer to other institutions, countries or education systems.
7. **Narrow feature set.** Seven items cannot capture workload, finances, sleep,
   physical health, mental-health history or social support quality. Large
   unexplained variance is expected — and observed (hold-out R² = 0.41).
8. **Class imbalance handled, not eliminated.** With 14 low-stress students in
   total, the Low band remains intrinsically hard to learn. SMOTE was considered
   and rejected: interpolating synthetic students between a handful of real ones
   on a coarse ordinal scale would manufacture response patterns nobody gave.
9. **Collinear engineered features.** The composite pressure score overlaps its
   components by construction, which helps prediction and complicates
   attribution. Rankings *within* the pressure block should not be over-read.
10. **Retained duplicates.** Five identical answer patterns were kept as genuine
    responses. If any were accidental double submissions, a small optimistic bias
    remains.

**Not a diagnostic tool.** These models estimate a self-reported survey rating.
They cannot diagnose, screen for, or measure psychological distress, and must not
be used to make consequential decisions about an individual.

---

## Future Improvements

**Data** — a larger, more balanced sample across institutions and stages;
repeated measurement across an academic year so within-student change can be
modelled; validation of the stress target against an established instrument
rather than a single ad-hoc item.

**Modelling** — **ordinal regression** (proportional-odds or ordered-probit) is
a better structural fit than either plain regression or unordered multiclass
classification, since it uses the band ordering instead of discarding it.
**Nested cross-validation** would give an unbiased estimate of the whole
select-and-tune procedure rather than of the selected model alone. On a larger
sample, probability calibration and threshold analysis would matter more than
raw accuracy — the SVC's `predict`/`predict_proba` disagreement documented in
section ML.22 is a concrete instance.

**Analysis** — multivariate models adjusting for confounders; mediation analysis
of the coping-strategy pathway; mixed-effects models if institution-level
grouping becomes available.

**Engineering** — a small inference API around the saved pipelines; a data
validation contract on incoming records; drift monitoring between cohorts; and
CI running `src/train.py --verify` so a dependency bump that changes the results
fails loudly instead of silently.

**Ethics and governance** — informed consent for this specific use, a documented
data-protection assessment, human review of any consequential decision, and clear
communication that a predicted band is a coarse statistical signal about a
*response pattern*, not a judgement about a person.

---

## Author

**Seherin Zaman**

Built as a portfolio project. The analysis, modelling decisions and write-up are
my own; the dataset is third-party and credited in
[`data/README.md`](data/README.md).

---

## Licence

Code and documentation: [MIT](LICENSE).
The dataset is third-party and is **not** covered by that licence — see
[`data/README.md`](data/README.md) before publishing this repository.
