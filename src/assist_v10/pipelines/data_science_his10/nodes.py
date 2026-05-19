"""
Data Science nodes for HIS-10 No-Show Guard.

This module implements the modelling pipeline for predicting appointment
no-shows using LightGBM with Optuna hyperparameter optimisation and MLflow
experiment tracking.

Leakage fix
-----------
The original notebook fitted ``KNNImputer`` and ``StandardScaler`` on the
**full** dataset before splitting.  This module defers those transformations
to the training node so they are fitted **only on training data** and then
applied to both train and test sets.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Tuple

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# =============================================================
# Node 1 — Feature Engineering (leakage-free)
# =============================================================

def preprocess_features(preprocessed_data: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering that does **not** cause data leakage.

    Numeric scaling / imputation are deferred to the training node so that
    the transformers are fitted only on the training partition.

    Parameters
    ----------
    preprocessed_data : pd.DataFrame
        Raw output of the Data Engineering pipeline (29 cols, ~28 551 rows).

    Returns
    -------
    pd.DataFrame
        OHE-encoded DataFrame.  ``duration_min`` and ``lead_time_days``
        remain in their original (unscaled) form.
    """
    df = preprocessed_data.copy()

    # 1. Drop leakage / redundant / no-variability columns ---------------
    cols_to_drop = [
        "lead_time_invalid",   # will impute lead_time_days instead
        "p_status",            # data leakage (99.83 % corr with no_show)
        "p_tpo_cita",          # duplicate of tpo_cita
        "m_sexo",              # near-duplicate of p_sexo
        "m_edad_num",          # almost entirely null
        "m_status",            # data leakage
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # 2. Binary flag transformations -------------------------------------
    df["conflicto"] = np.where(df["conflicto"] == "C", 1, 0)
    df["agregada"] = np.where(df["agregada"] == "A", 1, 0)
    df["ultimahora"] = np.where(df["ultimahora"] == "U", 1, 0)

    # 3. String cleaning, zero-padding and frequency masking -------------
    def _clean_pad_and_mask(
        col_name: str,
        pad_len: int,
        freq_mask_val: str,
        replace_dict: dict | None = None,
    ) -> None:
        df[col_name] = (
            df[col_name]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        if replace_dict:
            df[col_name] = df[col_name].replace(replace_dict)
        if pad_len > 0:
            mask_unknown = df[col_name].isin(["UNKNOWN"])
            df.loc[~mask_unknown, col_name] = (
                df.loc[~mask_unknown, col_name].str.zfill(pad_len)
            )
        counts = df[col_name].value_counts()
        freq_1_vals = counts[counts == 1].index
        df.loc[df[col_name].isin(freq_1_vals), col_name] = freq_mask_val

    _clean_pad_and_mask("med", pad_len=6, freq_mask_val="OTHERS")

    _clean_pad_and_mask(
        "m_ciu", pad_len=3, freq_mask_val="OTR",
        replace_dict={
            "NA": "UNKNOWN", "N/A": "UNKNOWN", "N": "UNKNOWN",
            "A": "UNKNOWN", "NA+": "UNKNOWN", "AN": "UNKNOWN",
        },
    )

    _clean_pad_and_mask(
        "m_col", pad_len=4, freq_mask_val="OTRA",
        replace_dict={
            "NA": "UNKNOWN", "N/A": "UNKNOWN",
            "A": "UNKNOWN", "N": "UNKNOWN",
        },
    )

    _clean_pad_and_mask(
        "m_cp", pad_len=5, freq_mask_val="OTROS",
        replace_dict={"A": "UNKNOWN"},
    )

    _clean_pad_and_mask(
        "m_edo", pad_len=2, freq_mask_val="OTRO",
        replace_dict={"N/": "UNKNOWN", "A": "UNKNOWN"},
    )

    _clean_pad_and_mask(
        "m_pai", pad_len=3, freq_mask_val="OTR",
        replace_dict={
            "NAN": "UNKNOWN", "A": "UNKNOWN", "N": "UNKNOWN",
            "MÉX": "MEX", "MX": "MEX",
        },
    )

    # 4. One-Hot Encoding ------------------------------------------------
    cols_to_ohe = [
        "area", "med", "esp", "buffer", "appointment_hour", "tpo_cita",
        "appointment_day_of_week", "appointment_day", "appointment_month",
        "p_sexo", "p_tpo_pac", "m_ciu", "m_col", "m_cp", "m_edo", "m_pai",
    ]
    df = pd.get_dummies(df, columns=cols_to_ohe, dtype=int)

    # Remove non-informative OHE columns
    for col in ["p_sexo_N", "tpo_cita_P"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    logger.info(
        "Feature engineering complete: %d rows × %d columns",
        df.shape[0], df.shape[1],
    )
    return df


# =============================================================
# Node 2 — Train / Test Split
# =============================================================

def split_data(
    model_input: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Split the dataset into training and test sets with stratification.

    Parameters
    ----------
    model_input : pd.DataFrame
        Output of ``preprocess_features``.
    parameters : dict
        Must contain ``his10.split.test_size`` and ``his10.split.random_state``.

    Returns
    -------
    dict
        ``{"X_train", "X_test", "y_train", "y_test"}``
    """
    split_params = parameters["split"]
    test_size = split_params["test_size"]
    random_state = split_params["random_state"]

    X = model_input.drop(columns=["no_show"])
    y = model_input["no_show"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logger.info(
        "Data split: train=%d, test=%d, features=%d",
        len(X_train), len(X_test), X_train.shape[1],
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


# =============================================================
# Helpers (private)
# =============================================================

def _fit_numeric_transformers(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, KNNImputer, StandardScaler, StandardScaler]:
    """Fit KNNImputer / StandardScalers on **train only**, transform both.

    Returns the transformed DataFrames and the fitted transformer objects
    (so they can be bundled with the model for inference).
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # KNN Imputation for lead_time_days (fit on train only)
    knn_imputer = KNNImputer(n_neighbors=5)
    X_train[["lead_time_days"]] = knn_imputer.fit_transform(
        X_train[["lead_time_days"]]
    )
    X_test[["lead_time_days"]] = knn_imputer.transform(
        X_test[["lead_time_days"]]
    )

    # Log1p transformation (deterministic — no leakage)
    X_train["lead_time_days"] = np.log1p(
        X_train["lead_time_days"].clip(lower=0)
    )
    X_test["lead_time_days"] = np.log1p(
        X_test["lead_time_days"].clip(lower=0)
    )

    # StandardScaler for duration_min (fit on train only)
    scaler_duration = StandardScaler()
    X_train[["duration_min"]] = scaler_duration.fit_transform(
        X_train[["duration_min"]]
    )
    X_test[["duration_min"]] = scaler_duration.transform(
        X_test[["duration_min"]]
    )

    # StandardScaler for lead_time_days (fit on train only)
    scaler_lead_time = StandardScaler()
    X_train[["lead_time_days"]] = scaler_lead_time.fit_transform(
        X_train[["lead_time_days"]]
    )
    X_test[["lead_time_days"]] = scaler_lead_time.transform(
        X_test[["lead_time_days"]]
    )

    return X_train, X_test, knn_imputer, scaler_duration, scaler_lead_time


def _evaluate_predictions(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Calculate all classification metrics."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
    }


# =============================================================
# Node 3 — Train with Optuna + MLflow
# =============================================================

def train_model_with_optuna(
    split_data: Dict[str, Any],
    parameters: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Train a LightGBM model using Optuna + MLflow.

    Numeric transformers (``KNNImputer``, ``StandardScaler``) are fitted
    **only** on the training partition to prevent data leakage.

    Parameters
    ----------
    split_data : dict
        ``{"X_train", "X_test", "y_train", "y_test"}``
    parameters : dict
        Pipeline parameters (optuna, training, split sections).

    Returns
    -------
    model_bundle : dict
        Contains the trained LGBMClassifier, fitted transformers,
        feature column list and best hyper-parameters.
        Persisted as ``trained_model_his10.pkl`` via the Kedro catalog.
    test_data : dict
        ``{"X_test": <transformed>, "y_test": <series>}`` for the
        evaluate node.
    """
    import warnings
    warnings.filterwarnings("ignore")

    optuna_params = parameters["optuna"]
    training_params = parameters["training"]
    split_params = parameters["split"]

    n_trials = optuna_params["n_trials"]
    experiment_name = optuna_params["experiment_name"]
    threshold = training_params["threshold"]
    random_state = split_params["random_state"]

    # ── Fit numeric transformers on train, transform both ────────────
    X_train, X_test, knn_imputer, scaler_dur, scaler_lt = (
        _fit_numeric_transformers(
            split_data["X_train"], split_data["X_test"],
        )
    )
    y_train = split_data["y_train"]
    y_test = split_data["y_test"]

    # ── Class imbalance weight ──────────────────────────────────────
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    scale_weight = negatives / positives

    logger.info(
        "Training distribution: 0s=%d, 1s=%d, scale_pos_weight=%.4f",
        negatives, positives, scale_weight,
    )

    # ── Configure MLflow ────────────────────────────────────────────
    # kedro run is expected to be executed from the project root,
    # so os.getcwd() points there.
    project_root = os.getcwd()
    mlruns_dir = os.path.join(project_root, "mlruns")
    tracking_uri = "file:///" + mlruns_dir.replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info("MLflow tracking URI: %s", tracking_uri)

    # ── Optuna objective ────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.2, log=True,
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", 5, 50,
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0,
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True,
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", 1e-8, 10.0, log=True,
            ),
            "scale_pos_weight": scale_weight,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }

        with mlflow.start_run(
            nested=True, run_name=f"trial_{trial.number:03d}",
        ):
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = _evaluate_predictions(y_test, y_proba, threshold)

            mlflow.log_params(
                {k: v for k, v in params.items() if k != "verbose"},
            )
            mlflow.log_metrics({
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "accuracy": metrics["accuracy"],
            })

            logger.info(
                "Trial %3d | ROC-AUC: %.4f | F1: %.4f | "
                "Recall: %.4f | Precision: %.4f",
                trial.number, metrics["roc_auc"], metrics["f1_score"],
                metrics["recall"], metrics["precision"],
            )

        return metrics["roc_auc"]

    # ── Run Optuna optimisation ─────────────────────────────────────
    logger.info(
        "Starting Optuna optimisation (%d trials) …", n_trials,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", study_name=experiment_name,
    )

    run_name = f"Optuna_HPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name):
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # ── Best configuration ──────────────────────────────────────
        best_params = study.best_params.copy()
        best_params["scale_pos_weight"] = scale_weight
        best_params["random_state"] = random_state
        best_params["n_jobs"] = -1
        best_params["verbose"] = -1

        logger.info("Best ROC-AUC in search: %.4f", study.best_value)
        logger.info("Best params: %s", best_params)

        # ── Train final model ───────────────────────────────────────
        final_model = lgb.LGBMClassifier(**best_params)
        final_model.fit(X_train, y_train)

        y_proba_final = final_model.predict_proba(X_test)[:, 1]
        final_metrics = _evaluate_predictions(
            y_test, y_proba_final, threshold,
        )

        # Log best results to parent run
        mlflow.log_params({
            "best_" + k: v
            for k, v in best_params.items()
            if k != "verbose"
        })
        mlflow.log_metrics({
            "best_" + k: v
            for k, v in final_metrics.items()
            if isinstance(v, float)
        })
        mlflow.log_metric("n_trials", n_trials)
        mlflow.lightgbm.log_model(final_model, "best_lightgbm_model")

        # Save trials summary as MLflow artifact
        trials_csv_path = os.path.join(
            project_root, "data", "09_tracking",
            "optuna_trials_his10.csv",
        )
        os.makedirs(os.path.dirname(trials_csv_path), exist_ok=True)
        study.trials_dataframe().to_csv(trials_csv_path, index=False)
        mlflow.log_artifact(trials_csv_path, artifact_path="optuna_results")

        logger.info("Trials summary saved to %s", trials_csv_path)

    # ── Build model bundle (includes transformers for inference) ─────
    model_bundle = {
        "model": final_model,
        "knn_imputer": knn_imputer,
        "scaler_duration": scaler_dur,
        "scaler_lead_time": scaler_lt,
        "feature_columns": list(X_train.columns),
        "best_params": best_params,
        "best_roc_auc": study.best_value,
    }

    test_data = {
        "X_test": X_test,
        "y_test": y_test,
    }

    return model_bundle, test_data


# =============================================================
# Node 4 — Evaluate Model
# =============================================================

def evaluate_model(
    model_bundle: Dict[str, Any],
    test_data: Dict[str, Any],
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate the trained model and return metrics.

    The output dict is persisted as ``evaluation_metrics_his10.json``
    via the Kedro catalog.
    """
    threshold = parameters["training"]["threshold"]
    model = model_bundle["model"]
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = _evaluate_predictions(y_test, y_proba, threshold)

    evaluation_output = {
        "experiment_name": parameters["optuna"]["experiment_name"],
        "timestamp": datetime.now().isoformat(),
        "n_trials": parameters["optuna"]["n_trials"],
        "best_params": {
            k: (
                float(v)
                if isinstance(v, (int, float, np.integer, np.floating))
                else str(v)
            )
            for k, v in model_bundle["best_params"].items()
            if k != "verbose"
        },
        "metrics": {
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "accuracy": metrics["accuracy"],
        },
        "confusion_matrix": {
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        },
        "training_info": {
            "test_size": len(X_test),
            "n_features": int(X_test.shape[1]),
            "threshold": threshold,
        },
    }

    logger.info(
        "Final evaluation — ROC-AUC: %.4f | PR-AUC: %.4f | "
        "F1: %.4f | Precision: %.4f | Recall: %.4f",
        metrics["roc_auc"], metrics["pr_auc"], metrics["f1_score"],
        metrics["precision"], metrics["recall"],
    )

    return evaluation_output
