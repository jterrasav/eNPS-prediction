# Predicting Individual eNPS Responses with Deep Learning

Bachelor's Thesis (Grado en Ciencia de Datos, Universitat de València, 2025).

Predicting the individual score (1–10) that each employee will give to the eNPS question — *"How likely are you to recommend your company as a good place to work?"* — from their historical workplace-climate data. Built on real, multi-company data from the [Happyforce](https://www.myhappyforce.com/) employee-experience platform.

**Best model: optimized LSTM — R² 0.65 · MAE 0.82 · MSE 1.41** on a 1–10 scale, a 16% MSE reduction over the non-temporal baseline.

---

## Results

Six models compared under identical evaluation conditions:

| Model | MSE | MAE | R² |
|---|---|---|---|
| MLP — independent observations | 1.6751 | 0.9192 | 0.6163 |
| MLP — independent, tuned | 1.6676 | 0.9167 | 0.6181 |
| MLP — flattened sequences | 1.5298 | 0.9024 | 0.6244 |
| MLP — flattened sequences, tuned | 1.4819 | 0.8486 | 0.6362 |
| LSTM — sequences | 1.4121 | 0.8239 | 0.6533 |
| **LSTM — sequences, tuned** | **1.4081** | **0.8180** | **0.6543** |

**Main finding:** every model that receives temporal information outperforms every model that does not. Sequence modelling is worth more here than architecture tuning — hyperparameter search moved R² by under 0.02, while adding temporal structure moved it by 0.04.

A mean absolute error of 0.82 points on a 1–10 scale is precise enough to flag meaningful shifts in individual employee sentiment before they surface in aggregate metrics.

---

## Dataset

Real anonymized data from the Happyforce platform, spanning **2017–2024** across multiple client companies.

| | |
|---|---|
| Happiness Index records (daily mood votes) | 6,338,171 |
| Individual eNPS responses | 134,525 |
| Source tables unified | 4 |
| Final modelling dataset | 110,167 observations × 113 features |
| Temporal sequences (4 steps) | 31,527 × 4 × 110 |

Target distribution: 55,234 promoters · 34,999 passives · 19,934 detractors.

Three input families:

- **Happiness Index (HI)** — daily 4-level emotional self-report (sad → very happy), averaged over the window
- **Scores** — structured survey responses grouped into workplace factors (Relationships, Intrinsic Motivation, Feedback, Alignment, Wellbeing, Reward & Recognition)
- **Calendar and interval features** — month/quarter dummies and `delta_days` between consecutive eNPS responses

Gender, birth date and hiring date were available but deliberately excluded to minimise personal data in the analysis.

---

## Method

**Temporal alignment.** eNPS is a quarterly metric, so each target response draws only on HI and score records from the **90 days preceding it**. Nothing recorded after the target vote enters the feature set — the causal direction of the problem is preserved by construction.

**Sequence construction.** Sliding windows of length 4 per employee, with an iterative filter discarding any window where consecutive steps sit more than 180 days apart: beyond six months, a previous state no longer reliably describes the current one. `delta_days` is added as an explicit feature so the model can distinguish tightly spaced from widely spaced histories.

**Leakage prevention.** The 70/15/15 split is made **by `employeeId`, not by row**. No employee appears in more than one partition, so the models are evaluated on people they have never seen — not on unseen quarters of people they already know. This is the single most important design decision in the project: a row-level split would have inflated every metric reported above.

**Preprocessing.** Deduplication, null removal, filtering of employees with insufficient history, Min-Max scaling, and pivoting of long-format score records into a wide feature matrix with explicit `_has_value` indicators for missingness.

**Optimization.** Keras Tuner `RandomSearch`, 50 trials per architecture over layer count, units, dropout rates and dense-layer width; best configuration by validation loss, then fully retrained.

---

## Repository

| File | Contents |
|---|---|
| `Tratamientos_Datos.ipynb` | Cleaning, temporal alignment, feature engineering, final dataset construction |
| `MLP_1Observacion.ipynb` | Non-sequential MLP: baseline and tuned |
| `LSTM_MLP_Secuencias.ipynb` | Sequence models: LSTM and flattened MLP, baseline and tuned |
| `Comparaciones.ipynb` | Cross-model comparison and figures |
| `eNPS_TFG_JorgeTerrasaVidal.pdf` | Full thesis (Spanish), including learning curves and scatter plots per model |

Source data is not included in this repository and is not publicly redistributable. Notebook outputs have been cleared: they contained production identifiers from the source platform.

**Stack:** Python · TensorFlow/Keras · Keras Tuner · Pandas · NumPy · Scikit-learn · Matplotlib

---

## Known limitations

- The original runs did not fix a random seed in `train_test_split`, so re-executing reproduces the reported metrics only approximately. Any rerun should set `random_state`.
- Sequence-model results rest on 31,527 windows drawn from employees with at least four aligned eNPS responses, which skews the sample toward longer-tenured, more consistently engaged participants. The metrics should be read as conditional on that population.
- Predictions compress slightly at the extremes of the scale — mild overestimation in the 1–4 range, mild saturation near 10 — a known consequence of optimising squared error on a bounded ordinal target.

---

## Author

**Jorge Terrasa Vidal** — [LinkedIn](https://www.linkedin.com/in/jorge-terrasa-vidal/)

Thesis supervised by Valero Laparra Pérez-Muelas (ETSE-UV), in collaboration with Happyforce.
