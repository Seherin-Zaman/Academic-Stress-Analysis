# Project Notes — Academic Stress Predictive Modelling

Condensed technical documentation. Every figure here is taken from the executed
notebook (`notebooks/machine_learning.ipynb`) or from
`models/model_metadata.json`; nothing is restated from memory.

---

## 1. Problem statement

Predict a student's self-reported academic stress from a short survey, and
identify which of the surveyed factors carry that predictive signal.

The problem is framed twice, deliberately, because the two framings answer
different questions and fail in different ways:

| Framing | Target | Question |
| --- | --- | --- |
| Regression | `Academic Stress Index`, 1–5 continuous | *How intense is a student's reported stress likely to be?* |
| Classification | Low / Moderate / High band | *Which stress band is a student most likely to fall into?* |

Both use the same stratified split, so their results are directly comparable.

**Scope.** This models a *self-reported rating*, not a clinical measure. It
identifies statistical association, never causation, and is a portfolio
demonstration rather than a diagnostic or screening instrument.

---

## 2. Dataset

Student Academic Stress — Real World Dataset (Kaggle, `poushal02`).

| Property | Value |
| --- | --- |
| Raw responses | 140 |
| Usable (target present) | 129 |
| Training / hold-out | 96 / 33 |
| Columns | 9 raw → 12 model features |
| Collection window | 24 Jul – 18 Aug 2025 |
| Missing cells | 55 (~4.4%) |

Full column-level documentation is in [`../data/README.md`](../data/README.md).

**Composition risks worth stating.** 100 of 140 respondents are undergraduates
and only 11 are post-graduates; the survey ran ~4 weeks through a single
distribution channel. Subgroup conclusions for the smaller stages are
unreliable, and nothing here should be assumed to transfer to another cohort.

---

## 3. Target variable

`Academic Stress Index` — a single self-reported item, 1 (lowest) to 5 (highest).

* Distribution: `1`→5, `2`→9, `3`→35, `4`→50, `5`→30, missing→11.
* Mean 3.71, sd 1.03 — strongly skewed towards high stress.

**11 rows with no target are dropped, never imputed.** Imputing a target
fabricates ground truth and inflates every metric measured against it.

**Banding (section ML.14).** Two rules were considered:

| Rule | Verdict |
| --- | --- |
| Tertile split at the 33rd/67th percentiles | **Rejected** — the distribution is concentrated on values 3, 4 and 5, so a percentile boundary lands *inside* a single rating and two students who both answered "4" could receive different labels. |
| Scale-based: 1–2 → Low, 3 → Moderate, 4–5 → High | **Adopted** — cut-points fall on the natural midpoint and upper half, identical answers always get identical labels, and the bands are self-explanatory. |

Resulting counts: **Low 14, Moderate 35, High 80** — imbalance ≈ **5.7 : 1**.
The edges are fixed once as a configuration constant and never adjusted
afterwards to improve a metric.

---

## 4. Feature engineering

Five features are derived, each for a stated reason:

| Feature | Definition | Rationale |
| --- | --- | --- |
| `Overall Pressure Score` | Mean of the three pressure ratings | Same 1–5 scale and moderately inter-correlated, so the mean is a meaningful total pressure-load index; item-level noise partly cancels |
| `Peer x Home Pressure` | Peer × home pressure | Tests whether the two *social* pressure sources compound; a linear model cannot express this without an explicit product term |
| `Competition x Home Pressure` | Competition × home pressure | Same logic for institutional competition combined with family expectation |
| `Environment Disturbance` | Peaceful 0, Noisy 1, Disrupted 2 | Preserves a real ordering that one-hot encoding discards; a single monotone trend is far more sample-efficient on 129 rows. The one-hot version is kept too, so no information is lost |
| `Maladaptive Coping` | 1 if `Emotional Breakdown`, else 0 | Groups the two adaptive strategies and isolates the distress response as one risk indicator — a cheaper signal than three separate one-hot columns |

**Deliberately not created:** a bad-habits flag (an exact duplicate of the
`Bad Habits_Yes` one-hot column); a hand-weighted risk score (the weights would
be invented rather than learned); and **anything derived from the target** — a
stress category, a high-stress flag, a stress-to-pressure ratio — which would be
the target in disguise.

### The temporal-feature decision

Six timestamp features were built (`Response Year/Month/Day/DayOfWeek/Hour`,
`Days Since First Response`) and then **tested rather than assumed**:

* `Response Year` is constant (all 2025) and was dropped automatically by a
  zero-variance check.
* The rest changed CV MAE by ~0.02 against fold-to-fold sd of 0.08–0.11 — about
  a fifth of the noise — and inconsistently in direction (the linear model's MAE
  improved while its R² got worse).

They also describe **survey administration, not the student**: two-thirds of
responses arrived in the first three days, so any signal is largely a proxy for
"answered early", which cannot transfer to a new cohort.

**Decision: dropped.** Where the statistical evidence is a tie, discard the
feature that cannot plausibly generalise.

**Final feature set: 12 columns** — 3 raw ratings + 5 engineered numeric + 4
categorical.

---

## 5. Leakage control

Five mechanisms, worth listing because they are the methodological core:

1. **Split before any fitting.** The hold-out set is created first and used once,
   at the very end.
2. **All learned preprocessing inside the `Pipeline`.** Median/most-frequent
   imputation, `StandardScaler` and `OneHotEncoder(handle_unknown="ignore")` are
   steps of a `ColumnTransformer` that `cross_validate` and both searches re-fit
   inside every fold, on that fold's training portion only.
3. **Feature engineering is row-wise only**, so applying it before the split
   cannot transfer information between rows. (`tests/test_pipeline.py` asserts
   this by re-engineering a 10-row subset and comparing.)
4. **Model selection uses training-set CV only.** The hold-out never influences
   which model or hyperparameters are chosen.
5. **A runtime leakage guard.** An `assert` fails the notebook if any
   target-derived column reaches the feature list, rather than letting inflated
   metrics through silently.

Part I of the notebook imputes on the full dataset and derives `*_Category`
columns *from the target*. Part II therefore **re-loads the raw CSV** rather
than reusing Part I's frame — the reason is documented at the top of Part II.

---

## 6. Evaluation strategy

| Aspect | Choice | Why |
| --- | --- | --- |
| Split | 75/25, stratified by band | Only 14 Low students exist; a plain random split could leave the band unrepresented |
| Cross-validation | 5-fold × 3 repeats (45 fits) | A single split on 96 rows is dominated by fold-assignment luck |
| Regression metric | **CV MAE** | On the target's natural scale, the most stable of the four, and not distorted by low target variance the way R² is |
| Classification metric | **CV macro F1** | Treats all three bands equally; accuracy is misleading because always predicting "High" already scores ~62% |
| Baselines | Mean predictor, majority class, plus simple linear/logistic models | Establishes "no model at all" and "simplest real model" before anything complex |

Both primary metrics were **declared before fitting** and the winner selected on
cross-validation only — never on the hold-out score.

**Class imbalance** is handled with stratification, `class_weight="balanced"`
and macro averaging. **SMOTE was considered and rejected**: with 14 low-stress
students, interpolating synthetic students between a handful of real ones on a
coarse ordinal scale manufactures response patterns nobody gave, and risks
optimistic CV scores more than it helps.

---

## 7. Models tested

**Regression (8):** mean predictor (baseline), Linear Regression (baseline),
Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost.
Tuned: Ridge (`GridSearchCV` over `alpha`), Random Forest (`RandomizedSearchCV`,
25 draws over depth / leaf size / feature subsampling / forest size).

**Classification (7):** majority class (baseline), Logistic Regression
(baseline), Decision Tree, Random Forest, Gradient Boosting, SVM (RBF), XGBoost.
Tuned: XGBoost and the RBF SVM (`RandomizedSearchCV`, 25 draws each).

Searches were kept small on purpose: 25 configurations × 15 folds is already 375
fits, and a larger search on 96 rows would mostly be selecting on validation-fold
noise.

The unconstrained Decision Tree was included as a **diagnostic** — it reaches
train R² 0.995 with CV R² −0.348, making the overfitting mechanism visible
rather than hypothetical.

---

## 8. Best models

### Regression — Random Forest (tuned)

| Metric | Value |
| --- | --- |
| CV MAE | **0.637** |
| Baseline CV MAE (mean predictor) | 0.823 |
| Error reduction | **22.6%** |
| Hold-out MAE | 0.697 |
| Hold-out RMSE | 0.840 |
| Hold-out R² | 0.408 |
| Hold-out explained variance | 0.410 |
| Train R² | 0.654 |

Tuning capped depth at 3: train R² fell 0.91 → 0.65 while the CV score
*improved* — removing memorisation, not capacity.

### Classification — SVM (RBF kernel, tuned)

| Metric | Value | Majority baseline |
| --- | --- | --- |
| CV macro F1 | **0.610** | 0.256 |
| Hold-out macro F1 | 0.539 | 0.252 |
| Hold-out weighted F1 | 0.564 | — |
| Hold-out accuracy | 0.545 | 0.606 |
| Hold-out macro precision | 0.578 | 0.202 |
| Hold-out macro recall | 0.535 | 0.333 |
| Hold-out ROC-AUC (OvR) | 0.720 | 0.500 |

Per-band hold-out: Low P 0.67 / R 0.50 (n=4), Moderate P 0.33 / R 0.56 (n=9),
High P 0.73 / R 0.55 (n=20).

### The caveat that must travel with these numbers

Tuned Ridge scores **0.6372** against the forest's **0.6367** — a gap of 0.0005
against fold-to-fold sd of ~0.11. Six of ten regression candidates fall within
one CV standard deviation of the winner, and six of nine classification
candidates likewise. **The leaderboards identify a group of equivalent models,
not a decisive winner.** The selection rule breaks the tie deterministically;
it does not establish superiority.

---

## 9. Key findings

1. **The pressure block dominates.** `Overall Pressure Score` is the strongest
   single correlate (Spearman ρ ≈ 0.56) and the largest contributor under
   impurity importance, permutation importance and SHAP alike. Because the
   composite is collinear with its components by construction, the defensible
   claim is that the *block as a whole* drives predictions — permutation
   importance concentrating on the composite is a collinearity artefact, not a
   finding about that one question.
2. **Study environment matters, modestly.** The only categorical factor
   surviving Bonferroni correction (Kruskal–Wallis p ≈ 0.010, ε² ≈ 0.06 —
   ~6% of rank variance). Mean stress 4.03 disrupted vs 3.43 peaceful.
3. **Academic stage does not differ significantly** (p ≈ 0.75, ε² ≈ 0). This
   **corrects a claim made in Part I**, which read the group means (High School
   3.83 > Post-Graduate 3.73 > Undergraduate 3.67) as a real ordering. With 11
   post-graduate respondents it is well within sampling variation, and the models
   agree — academic stage receives almost no weight. Both results are kept in the
   notebook and the correction explained.
4. **Bad habits: suggestive, unresolved.** The largest raw difference (4.30 vs
   3.67) but not significant (p ≈ 0.13) with only 10 and 6 students in the
   smaller groups. A larger sample would be needed to settle it.
5. **Errors are the benign kind.** 14 of 15 hold-out misclassifications are
   between adjacent bands; Low↔High confusion is almost absent. On an ordinal
   target that is the failure mode you want.
6. **Predictions compress towards the scale centre.** Hold-out students span
   1–5 while predictions span ~1.2–4.6. Standard behaviour for a regularised
   model on a noisy self-report target, and preferable to confidently predicting
   extremes it cannot identify.
7. **Data volume is the binding constraint, not algorithm choice.** The learning
   curve is still descending at 96 rows (validation MAE 0.68 at 22 rows → 0.63
   at 76), complex models overfit before they outperform, and the leading models
   are separated by less than the CV noise.

---

## 10. Limitations

1. **129 labelled rows**, 33 in hold-out, **4 of them Low** — per-band metrics
   for Low are indicative, not measured.
2. **Model differences sit inside the noise** (spread ~0.01 MAE vs sd ~0.11).
3. **Self-reported, single-item target** with no external validation; subject to
   mood, social desirability and acquiescence bias.
4. **Cross-sectional** — no causal claim is possible in either direction, and a
   third factor could drive both sides of any association.
5. **Coverage skew** — undergraduate-dominated, one channel, ~4 weeks.
6. **Seven survey items** cannot capture workload, finances, sleep, health
   history or social support quality; large unexplained variance is expected and
   observed.
7. **Imbalance handled, not eliminated** — 14 Low students total.
8. **Collinear engineered features** complicate within-block attribution.
9. **Five duplicate answer patterns retained** as genuine; if any were
   accidental double submissions a small optimistic bias remains.
10. **`SVC.predict` and `SVC.predict_proba` can disagree** near a band boundary
    (libsvm voting vs a separately fitted Platt calibration). Section ML.22 now
    reports the probability of the *predicted* band and shows the most probable
    band beside it, so the disagreement is visible rather than hidden by a
    `max()`. A profile where the two differ is one whose band is not settled.

---

## 11. Potential improvements

**Data.** A larger, more balanced sample across institutions and stages;
repeated measurement across an academic year so within-student change (and
therefore temporal ordering) can be modelled; validation of the target against an
established instrument.

**Modelling.** **Ordinal regression** (proportional-odds / ordered-probit) fits
the structure better than either plain regression or unordered multiclass, since
it uses the band ordering rather than discarding it. **Nested cross-validation**
would give an unbiased estimate of the whole select-and-tune procedure rather
than of the selected model alone. Probability calibration and threshold analysis
would matter more than raw accuracy on a larger sample.

**Analysis.** Multivariate models adjusting for confounders; mediation analysis
of the coping-strategy pathway; mixed-effects models given institution-level
grouping.

**Engineering.** A small inference API over the saved pipelines; a data
validation contract on incoming records; drift monitoring between cohorts; CI
running `python -m src.train --verify` so a dependency bump that changes the
results fails loudly rather than silently.

**Ethics and governance.** Informed consent for this specific use, a documented
data-protection assessment, human review of any consequential decision, and clear
communication that a predicted band is a coarse statistical signal about a
*response pattern*, not a judgement about a person.

---

## 12. Reproducibility notes

* `RANDOM_STATE = 42` threaded through the split, CV folds, every stochastic
  model and both searches.
* Reference environment: **Python 3.13.5**, pandas 2.2.3, numpy 2.1.3,
  scikit-learn 1.6.1, xgboost 3.2.0, shap 0.51.0.
* **`scikit-learn` is pinned to 1.6.1** in `requirements.txt` because the
  committed pickles only load under the version that wrote them — 1.8.0 raises
  `AttributeError: Can't get attribute '_RemainderColsList'`. Regenerate with
  `python -m src.train` to use a different version.
* `python -m src.train --verify` reproduces all 13 committed metrics exactly.
* A full top-to-bottom notebook run takes roughly 3 minutes and regenerates
  `models/` and `reports/figures/`.
* The Tkinter dashboard is gated behind `LAUNCH_GUI = False` so
  `mainloop()` never blocks an automated run.
