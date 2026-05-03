# Data Engineering Output Guide for Data Science

This document explains the datasets produced by the Data Engineering pipeline and how they should be used by the Data Science team for the selected Assist v10 use cases:

- **HIS-05:** saturation / wait-time analysis.
- **HIS-10:** no-show prediction for medical appointments.

The goal of this pipeline is not to train models. Its purpose is to clean raw hospital tables, standardize key fields, validate important assumptions, and generate base datasets that can be consumed by the Data Science pipelines.

---

## What Data Science should use

After running the Data Engineering pipeline, the main datasets for Data Science are:

| Use case | Dataset | Path | Modeling goal |
|---|---|---|---|
| HIS-10 | `preprocessed_data_his10` | `data/03_primary/preprocessed_data_his10.parquet` | Predict appointment no-show. |
| HIS-05 | `master_table_his05` | `data/03_primary/master_table_his05.parquet` | Analyze / forecast emergency saturation and wait-time behavior. |

---

## How to generate the datasets

From the project root, activate the virtual environment and run:

```bash
kedro run --pipeline data_engineering
```

Optional syntax checks:

```bash
python -m py_compile src/assist_v10/pipelines/data_engineering/nodes.py
python -m py_compile src/assist_v10/pipelines/data_engineering/pipeline.py
```

The raw `.parquet` files must exist locally in:

```text
data/01_raw/
```

Raw data files must not be committed to GitHub.

---

## Input tables used by Data Engineering

The pipeline uses five raw hospital tables:

| Raw dataset | Meaning | Main use |
|---|---|---|
| `hospac` | Patient encounter / hospital episode table. | Bridge between appointments and patient information. |
| `hosagd` | Appointment schedule table. | Main source for HIS-10 because it contains appointment data and `asistencia`. |
| `hosmpi` | Master patient index. | Provides demographic information for patients. |
| `notamedicaurg` | Emergency medical notes. | Main source for HIS-05 demand and wait-time proxy. |
| `triage` | Emergency triage records. | Enriches HIS-05 with triage severity information. |

In simple terms:

```text
HOSAGD = appointment schedule
HOSPAC = hospital encounter / episode
HOSMPI = patient demographics
NOTAMEDICAURG = emergency medical notes
TRIAGE = emergency severity classification
```

---

## Main processed outputs

The pipeline also creates intermediate cleaned tables:

| Dataset | Path | Description |
|---|---|---|
| `processed_hospac` | `data/02_intermediate/processed_hospac.parquet` | Cleaned HOSPAC table. |
| `processed_hosagd` | `data/02_intermediate/processed_hosagd.parquet` | Cleaned appointment schedule table. |
| `processed_hosmpi` | `data/02_intermediate/processed_hosmpi.parquet` | Cleaned patient master table. |
| `processed_triage` | `data/02_intermediate/processed_triage.parquet` | Cleaned triage table. |
| `processed_notamedicaurg` | `data/02_intermediate/processed_notamedicaurg.parquet` | Cleaned emergency notes table. |

For modeling, Data Science should mainly use:

| Dataset | Path |
|---|---|
| `preprocessed_data_his10` | `data/03_primary/preprocessed_data_his10.parquet` |
| `master_table_his05` | `data/03_primary/master_table_his05.parquet` |

---

# HIS-10: No-Show Prediction

## Dataset

```text
data/03_primary/preprocessed_data_his10.parquet
```

## Dataset construction

The HIS-10 dataset is created from:

```text
HOSAGD + HOSPAC + HOSMPI
```

- `HOSAGD` provides appointment information and the attendance field.
- `HOSPAC` links appointment keys with the patient encounter / hospital episode.
- `HOSMPI` adds demographic information from the patient master table.

## Grain

```text
1 row = 1 appointment with known attendance label
```

## Target variable

The target variable is:

```text
no_show
```

It is created from `asistencia` in `HOSAGD`:

| Raw value | Meaning | Encoded value |
|---|---|---|
| `A` | Attended appointment | `no_show = 0` |
| `I` | No-show / missed appointment | `no_show = 1` |

Missing or invalid `asistencia` values are not imputed, because the true outcome is unknown. Those records are excluded from the initial labeled modeling dataset.

## Latest local validation

```text
preprocessed_data_his10: 28,551 labeled appointments

21,325 attended appointments
7,226 no-shows
```

Approximate target distribution:

```text
74.69% attended
25.31% no-show
```

This means the target is moderately imbalanced. Data Science should not rely only on accuracy.

## Suggested metrics for HIS-10

Recommended metrics:

- Recall.
- Precision.
- F1-score.
- ROC-AUC.
- PR-AUC.
- Confusion matrix.

The positive class should be:

```text
no_show = 1
```

because the business problem is identifying appointments likely to become no-shows.

## Important modeling notes for HIS-10

- `no_show` is already created.
- Numeric missing values were intentionally preserved for the modeling stage.
- Categorical missing values are represented as `UNKNOWN`.
- The dataset contains candidate variables, not final selected features.
- Final feature selection belongs to the Data Science stage.
- Review possible leakage variables before training.

Potential leakage warning:

```text
p_status
```

This variable should be reviewed before modeling. If it represents a post-appointment or post-event status, it should be removed from model features.

## HIS-10 join logic

`HOSAGD` is joined with `HOSPAC` using:

```text
area_key + cve_num_key + cve_mbo_key
```

This avoids duplicating appointments when `cve_num` and `cve_mbo` repeat across areas.

Then the result is joined with `HOSMPI` using:

```text
p_num_exp_key = m_num_exp_key
```

---

# HIS-05: Saturation / Wait-Time Analysis

## Dataset

```text
data/03_primary/master_table_his05.parquet
```

## Dataset construction

The HIS-05 dataset is created from:

```text
NOTAMEDICAURG + TRIAGE
```

- `NOTAMEDICAURG` provides emergency arrivals and the wait-time proxy.
- `TRIAGE` provides hourly triage severity counts.

## Grain

```text
1 row = 1 hour
```

This is important: HIS-05 is not appointment-level or patient-level. It is an hourly operational dataset.

## Main variables

| Variable | Meaning |
|---|---|
| `timestamp` | Hourly timestamp. |
| `pacientes_llegando` | Number of emergency arrivals per hour. |
| `tiempo_espera` | Average wait-time proxy per hour. |
| `tiempo_espera_mediana` | Median wait-time proxy per hour. |
| `wait_proxy_valid_count` | Number of valid wait-time proxy records in that hour. |
| `triage_events` | Number of triage records in that hour. |
| `triage_*` | Hourly counts by triage category. |
| `hour` | Hour of day. |
| `day_of_week` | Day of week. |
| `day` | Day of month. |
| `month` | Month. |
| `is_weekend` | Weekend indicator. |

## Wait-time proxy

Since `AtMed_Hora` was not available/useful in the raw data, the pipeline creates:

```text
wait_proxy_min = note_datetime - arrival_datetime
```

This is an operational proxy, not the official clinical waiting time.

Invalid proxy values are marked using:

```text
valid_wait_proxy = 1 if 0 <= wait_proxy_min <= 1440 minutes
```

For the final hourly dataset:

```text
tiempo_espera = average valid wait-time proxy per hour
```

## Latest local validation

```text
master_table_his05: 3,132 hourly records
```

The latest local validation showed:

```text
pacientes_llegando:
- min: 1
- median: 6
- mean: 7.29
- max: 49

tiempo_espera:
- median: 69.67 minutes
- mean: 110.80 minutes
- max: 1349.90 minutes
```

## Important modeling notes for HIS-05

- This is a time-dependent dataset.
- Avoid random train/test splits.
- Prefer chronological splits.
- `pacientes_llegando` can be used as a demand/saturation target.
- `tiempo_espera` can be used as a wait-time proxy target, but it is not the official clinical waiting time.
- Consider lag features and rolling averages.
- Review outliers before training, especially high wait-time values.
- `triage_*` variables can be used as exogenous predictors.

Suggested feature engineering ideas:

- Lag features for `pacientes_llegando`.
- Lag features for `tiempo_espera`.
- Rolling means by hour.
- Rolling medians.
- Hour-of-day effects.
- Day-of-week effects.
- Weekend effects.
- Triage severity counts.

---

## Cleaning and validation rules already applied

The Data Engineering pipeline already applies the following rules:

- Strip spaces from text fields.
- Remove non-printable/control characters from text fields.
- Normalize categorical/code-like columns.
- Convert empty strings to missing values.
- Parse date and time fields into real datetime columns.
- Validate appointment duration.
- Validate patient age.
- Clean and validate `asistencia`.
- Clean and validate triage categories.
- Replace non-informative codes such as `000`, `00000`, `NA`, `NULL` with missing values.
- Preserve numeric missing values for the Data Science/modeling stage.
- Fill categorical missing values with `UNKNOWN` only in the modeling base dataset.

---

## What Data Science still needs to decide

The following decisions are intentionally left for the Data Science stage:

| Area | Pending decision |
|---|---|
| Feature selection | Decide final model features. |
| Numeric missing values | Decide imputation strategy. |
| Categorical variables | Decide encoding strategy. |
| Outliers | Decide whether to cap, transform, remove or keep. |
| Class imbalance | Decide sampling strategy, class weights or threshold tuning for HIS-10. |
| Time split | Decide train/validation/test split strategy for HIS-05. |
| Model choice | Choose and evaluate models. |
| Metrics | Define final evaluation metrics. |
| Threshold | Select operating threshold for HIS-10 if needed. |

---

## Suggested immediate next steps for Data Science

### For HIS-10

1. Load:

```python
import pandas as pd

his10 = pd.read_parquet("data/03_primary/preprocessed_data_his10.parquet")
```

2. Use `no_show` as the target.
3. Separate predictors from target.
4. Review possible leakage variables, especially `p_status`.
5. Decide categorical encoding strategy.
6. Decide numeric imputation strategy.
7. Train a baseline classification model.
8. Evaluate with recall, precision, F1-score, ROC-AUC and PR-AUC.
9. Tune the threshold based on the business objective.

### For HIS-05

1. Load:

```python
import pandas as pd

his05 = pd.read_parquet("data/03_primary/master_table_his05.parquet")
```

2. Sort by `timestamp`.
3. Use chronological train/test split.
4. Decide target:
   - `pacientes_llegando` for demand/saturation.
   - `tiempo_espera` for wait-time proxy.
5. Create lag and rolling features.
6. Train a baseline forecasting/regression model.
7. Evaluate with metrics such as MAE, RMSE and MAPE, depending on the target.

---

## Important notes

- Do not commit real `.parquet` data files.
- The `data/` folder is ignored by Git except for `.gitkeep` files.
- The Data Wrangling notebook documents the analysis and reasoning.
- This pipeline implements the reproducible cleaning and dataset creation steps.
- Final feature selection, numeric imputation, encoding strategy and model training belong to the Data Science stage.