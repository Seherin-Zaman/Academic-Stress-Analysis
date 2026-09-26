# Dataset

## Name and source

**Student Academic Stress — Real World Dataset**, published on Kaggle by user
`poushal02`:

<https://www.kaggle.com/datasets/poushal02/student-academic-stress-real-world-dataset>

This URL is the source cited in the original notebook. Everything below is
derived by inspecting the CSV in this folder — nothing about the collection
process is asserted beyond what the file itself shows.

## File

| | |
| --- | --- |
| Filename | `Academic Stress Dataset.csv` |
| Format | UTF-8 CSV, comma-separated, one header row |
| Size | ~13 KB |
| Records | **140 survey responses** |
| Columns | **9** |
| Response timestamps | 24 July 2025 to 18 August 2025 |

The CSV is the only copy kept. No Excel duplicate is stored: the code reads the
CSV exclusively, so a second copy in another format would be an unsynchronised
duplicate rather than an asset.

**This file is read-only to the project.** Every cleaning and feature step in
the notebook operates on in-memory copies; nothing writes back to it.

## Columns

Column names in the raw file are the full survey questions. The notebook renames
them to short analysis names, shown in the second column.

| Raw column (survey question) | Analysis name | Type | Missing | Values |
| --- | --- | --- | --- | --- |
| `Timestamp` | — | text → datetime | 0 | Submission time, `DD/MM/YYYY HH:MM` |
| `Your Academic Stage` | `Academic Stage` | categorical | 0 | undergraduate (100), high school (29), post-graduate (11) |
| `Peer pressure` | `Peer pressure` | ordinal 1–5 | 10 | Likert rating |
| `Academic pressure from your home` | `Home Academic Pressure` | ordinal 1–5 | 10 | Likert rating |
| `Study Environment` | `Study Environment` | categorical | 8 | Peaceful (66), disrupted (34), Noisy (32) |
| `What coping strategy you use as a student?` | `Coping Strategy` | categorical | 7 | Analytical (80), Emotional breakdown (32), Social support (21) |
| `Do you have any bad habits like smoking, drinking on a daily basis?` | `Bad Habits` | categorical | 0 | No (123), Yes (10), prefer not to say (7) |
| `What would you rate the academic  competition in your student life` | `Academic Competition` | ordinal 1–5 | 9 | Likert rating (note the double space in the raw header) |
| `Rate your academic stress index ` | `Academic Stress Index` | ordinal 1–5 | 11 | **Target** (note the trailing space in the raw header) |

Two raw headers contain stray whitespace, which is why the loader calls
`str.strip()` on the column index before anything else.

## Target variable

**`Academic Stress Index`** — the student's own rating of their academic stress
on a 1 (lowest) to 5 (highest) scale.

It is used in two forms:

| Task | Target | Distribution |
| --- | --- | --- |
| Regression | The 1–5 rating as a continuous value | mean 3.71, sd 1.03 |
| Classification | Three ordered bands: 1–2 → **Low**, 3 → **Moderate**, 4–5 → **High** | Low 14, Moderate 35, High 80 |

Raw rating counts: `1` → 5, `2` → 9, `3` → 35, `4` → 50, `5` → 30, missing → 11.

The distribution is skewed towards high stress, and the three-band version is
imbalanced at roughly **5.7 : 1** between the largest and smallest class. This
drives several design decisions in the notebook — stratified splitting, balanced
class weights, and macro-averaged F1 rather than accuracy as the headline metric.

## Data quality as it arrives

* **55 missing cells** (~4.4% of the grid), spread across six columns.
* **11 responses carry no stress rating.** These are dropped rather than imputed
  — fabricating a target would inflate every metric measured against it. That
  leaves **129 labelled rows** for supervised learning.
* **Inconsistent capitalisation**: `Study Environment` contains `disrupted`
  lower-case alongside `Peaceful` and `Noisy`. Left unfixed this would encode as
  a fourth category.
* **0 exact duplicate rows**; **5 duplicate answer patterns** once `Timestamp` is
  ignored. These are retained — with three Likert items and four short
  categorical items, distinct students can legitimately give identical answers,
  and the repeats consist of the most common response on every item, which is
  what chance agreement looks like.
* All rating items lie within the valid 1–5 range; there are no out-of-scale
  values to correct.

## Preprocessing applied

Performed in the notebook and in `src/preprocessing.py`, never on this file:

1. Strip whitespace from column headers, then rename to short analysis names.
2. Parse `Timestamp` with an explicit `%d/%m/%Y %H:%M` format.
3. Lower-case and map category text onto canonical labels (fixes
   `disrupted`/`Disrupted`; shortens verbose coping-strategy wording).
4. Coerce rating items to numeric; set any value outside 1–5 to `NaN`.
5. Drop rows with no target value (140 → 129).
6. Engineer five features: `Overall Pressure Score`, `Peer x Home Pressure`,
   `Competition x Home Pressure`, `Environment Disturbance`, `Maladaptive Coping`.

**Imputation, scaling and one-hot encoding are deliberately *not* in this list.**
They are learned transformations, so they live inside the scikit-learn
`Pipeline` and are fitted on training folds only. Applying them to the whole
dataset first would leak information from the hold-out set into training.

## Licensing and redistribution — please read

The dataset's licence **could not be verified programmatically** while preparing
this repository (the Kaggle dataset page renders its licence field via
JavaScript and is not readable without a browser or an authenticated API call).
No licence is asserted here, and none should be inferred from this repository's
MIT `LICENSE`, which covers the code only.

**Before making this repository public, confirm the licence on the Kaggle page
above** (or via `kaggle datasets metadata poushal02/student-academic-stress-real-world-dataset`)
and act accordingly:

* If it permits redistribution (e.g. CC0, CC BY, ODbL), record the licence name
  in this file and add the attribution the licence requires.
* If it does not, remove `Academic Stress Dataset.csv` from the repository, add
  `data/*.csv` to `.gitignore`, and replace it with download instructions so the
  notebook stays runnable.

## Privacy

The file contains **no direct identifiers** — no names, email addresses, student
numbers, IP addresses or free-text responses. Each row is a submission timestamp
plus eight closed-response survey answers.

Two residual considerations are worth naming honestly rather than dismissing:

* The data concerns **student wellbeing**, a sensitive topic, and consists of
  self-reported psychological state.
* Combining an exact submission timestamp with a rare attribute combination
  (for instance, one of the 11 post-graduate respondents) is a theoretical
  re-identification route for anyone who knows the original cohort.

The project's use of the data is analytical and aggregate. The models are a
demonstration of a modelling workflow and are explicitly **not** a clinical,
diagnostic or screening instrument — see the limitations in the root `README.md`.
