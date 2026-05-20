"""
HIS-05 · Monitor de Saturación — Nodos del pipeline de Data Science
====================================================================
Flujo:
  master_table_his05
      │
      ▼
  build_features_node        →  his05_features, his05_feature_cols
      │
      ▼
  hyperparameter_tuning_node →  his05_best_params
      │
      ▼
  train_model_node           →  trained_model_his05, his05_cv_results,
                                his05_oof_preds, his05_y_array
      │
      ▼
  evaluate_model_node        →  evaluation_metrics_his05
"""

from __future__ import annotations

import logging
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_features_node(
    master: pd.DataFrame,
    params: dict,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Construye lags, rolling stats, features cíclicas y exógenas de triage
    sobre master_table_his05.

    Parámetros leídos de conf/base/parameters/his05.yml:
        target          : columna objetivo (pacientes_llegando)
        lags            : lista de horas de lag
        rolling_windows : lista de ventanas para rolling mean/std
    """
    target  = params["target"]
    lags    = params["lags"]
    windows = params["rolling_windows"]

    df = master.copy().sort_values("timestamp").reset_index(drop=True)

    # ── Variables temporales (crearlas si no vienen en la tabla) ─────────────
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = df["timestamp"].dt.hour
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["timestamp"].dt.dayofweek
    if "month" not in df.columns:
        df["month"] = df["timestamp"].dt.month
    if "is_weekend" not in df.columns:
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    if "week_of_year" not in df.columns:
        df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)

    # ── Lags ──────────────────────────────────────────────────────────────────
    for lag in lags:
        df[f"lag_{lag}h"] = df[target].shift(lag)

    # ── Rolling stats ─────────────────────────────────────────────────────────
    for w in windows:
        df[f"roll_mean_{w}h"] = df[target].shift(1).rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}h"]  = (
            df[target].shift(1).rolling(w, min_periods=1).std().fillna(0)
        )

    # ── Diferencias ───────────────────────────────────────────────────────────
    df["diff_1h"]  = df[target].diff(1)
    df["diff_24h"] = df[target].diff(24)

    # ── Cíclicas ──────────────────────────────────────────────────────────────
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ── Espera rezagada ───────────────────────────────────────────────────────
    if "tiempo_espera" in df.columns:
        df["lag_wait_1h"]  = df["tiempo_espera"].shift(1)
        df["lag_wait_24h"] = df["tiempo_espera"].shift(24)

    # ── Lista final de features ───────────────────────────────────────────────
    triage_cols   = [c for c in df.columns if c.startswith("triage_")]
    temporal_cols = [
        "hour_of_day", "day_of_week", "month", "is_weekend", "week_of_year",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    ]
    lag_cols  = [f"lag_{l}h"       for l in lags]
    roll_cols = (
        [f"roll_mean_{w}h" for w in windows] +
        [f"roll_std_{w}h"  for w in windows]
    )
    diff_cols = ["diff_1h", "diff_24h"]
    wait_cols = (
        ["lag_wait_1h", "lag_wait_24h"] if "lag_wait_1h" in df.columns else []
    )

    feature_cols = temporal_cols + lag_cols + roll_cols + diff_cols + triage_cols + wait_cols

    df_model = (
        df[["timestamp", target] + feature_cols]
        .dropna()
        .reset_index(drop=True)
    )

    log.info("Features: %d  |  Filas tras dropna: %d", len(feature_cols), len(df_model))
    return df_model, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# 2. BÚSQUEDA DE HIPERPARÁMETROS
# ─────────────────────────────────────────────────────────────────────────────

def hyperparameter_tuning_node(
    df_model: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
) -> dict[str, Any]:
    """
    Grid search sobre TimeSeriesSplit.
    Prueba todas las combinaciones de params['param_grid'] y devuelve
    la combinación con menor MAE promedio en CV.

    El grid se configura en conf/base/parameters/his05.yml:

        param_grid:
          n_estimators:     [300, 500]
          learning_rate:    [0.03, 0.05, 0.10]
          max_depth:        [4, 6]
          subsample:        [0.8]
          colsample_bytree: [0.7, 0.8]
          min_child_weight: [3, 5]
    """
    target     = params["target"]
    n_splits   = params["n_splits"]
    grid       = params["param_grid"]
    fixed      = params.get("fixed_params", {})

    X = df_model[feature_cols].values
    y = df_model[target].values

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    keys   = list(grid.keys())
    combos = list(product(*grid.values()))

    log.info("Tuning: %d combinaciones × %d folds", len(combos), n_splits)

    results = []
    for combo in combos:
        combo_params = dict(zip(keys, combo))
        xgb_p = {
            **fixed,
            **combo_params,
            "early_stopping_rounds": 30,
            "eval_metric": "mae",
            "random_state": 42,
            "n_jobs": -1,
        }

        fold_maes = []
        for train_idx, val_idx in tscv.split(X):
            m = XGBRegressor(**xgb_p)
            m.fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False,
            )
            preds = np.clip(m.predict(X[val_idx]), 0, None)
            fold_maes.append(mean_absolute_error(y[val_idx], preds))

        mean_mae = float(np.mean(fold_maes))
        std_mae  = float(np.std(fold_maes))
        results.append({**combo_params, "cv_mae_mean": mean_mae, "cv_mae_std": std_mae})
        log.info("  %s → MAE=%.4f ± %.4f", combo_params, mean_mae, std_mae)

    best = (
        pd.DataFrame(results)
        .sort_values("cv_mae_mean")
        .iloc[0]
        .to_dict()
    )

    # Convertir tipos numpy a Python nativos para que el JSON funcione
    best_params: dict[str, Any] = {
        k: (int(v)   if isinstance(v, (np.integer,))  else
            float(v) if isinstance(v, (np.floating,)) else v)
        for k, v in best.items()
    }

    log.info("Mejores parámetros: %s", best_params)
    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# 3. ENTRENAMIENTO DEL MODELO FINAL
# ─────────────────────────────────────────────────────────────────────────────

def train_model_node(
    df_model: pd.DataFrame,
    feature_cols: list[str],
    best_params: dict,
    params: dict,
) -> tuple[XGBRegressor, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Entrena con los mejores hiperparámetros sobre TimeSeriesSplit para obtener
    OOF predictions, determina n_estimators óptimo y entrena el modelo final
    sobre todos los datos.
    """
    target   = params["target"]
    n_splits = params["n_splits"]
    fixed    = params.get("fixed_params", {})

    X = df_model[feature_cols].values
    y = df_model[target].values

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Params que XGBoost requiere como int (al venir de JSON quedan float)
    int_keys = {"n_estimators", "max_depth", "min_child_weight"}
    clean_best = {
        k: (int(v) if k in int_keys else v)
        for k, v in best_params.items()
        if k not in ("cv_mae_mean", "cv_mae_std")
    }

    train_params = {
        **fixed,
        **clean_best,
        "early_stopping_rounds": 30,
        "eval_metric": "mae",
        "random_state": 42,
        "n_jobs": -1,
        "n_estimators": 500,
    }

    cv_rows    = []
    oof_preds  = np.full(len(y), np.nan)
    best_iters = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        m = XGBRegressor(**train_params)
        m.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        preds = np.clip(m.predict(X[val_idx]), 0, None)
        oof_preds[val_idx] = preds
        best_iters.append(m.best_iteration)

        cv_rows.append({
            "fold":    fold,
            "n_train": len(train_idx),
            "n_val":   len(val_idx),
            "MAE":     float(mean_absolute_error(y[val_idx], preds)),
            "RMSE":    float(root_mean_squared_error(y[val_idx], preds)),
            "R2":      float(r2_score(y[val_idx], preds)),
        })
        log.info(
            "Fold %d | MAE=%.4f  RMSE=%.4f  R²=%.4f",
            fold, cv_rows[-1]["MAE"], cv_rows[-1]["RMSE"], cv_rows[-1]["R2"],
        )

    # Modelo final sin early stopping
    best_n = max(int(np.mean(best_iters)), 50)
    final_params = {
        k: v for k, v in train_params.items()
        if k not in ("early_stopping_rounds", "eval_metric")
    }
    final_params["n_estimators"] = best_n

    final_model = XGBRegressor(**final_params)
    final_model.fit(X, y, verbose=False)

    log.info("Modelo final listo. n_estimators=%d", best_n)
    return final_model, pd.DataFrame(cv_rows), oof_preds, y


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUACIÓN Y MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model_node(
    final_model: XGBRegressor,
    cv_results_df: pd.DataFrame,
    oof_preds: np.ndarray,
    y_array: np.ndarray,
    df_model: pd.DataFrame,
    feature_cols: list[str],
    best_params: dict,
    params: dict,
) -> dict:
    """
    Consolida métricas OOF y de CV en el diccionario evaluation_metrics_his05
    que Kedro guarda en data/09_tracking/evaluation_metrics_his05.json.
    """
    target = params["target"]

    oof_mask   = ~np.isnan(oof_preds)
    y_true_oof = y_array[oof_mask]
    y_pred_oof = oof_preds[oof_mask]

    mae_oof  = mean_absolute_error(y_true_oof, y_pred_oof)
    rmse_oof = root_mean_squared_error(y_true_oof, y_pred_oof)
    r2_oof   = r2_score(y_true_oof, y_pred_oof)

    top_features = (
        pd.DataFrame({
            "feature":    feature_cols,
            "importance": final_model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .head(10)["feature"]
        .tolist()
    )

    used_params = {
        k: v for k, v in best_params.items()
        if k not in ("cv_mae_mean", "cv_mae_std")
    }

    metrics = {
        "model":              "XGBRegressor",
        "target":             target,
        "n_features":         len(feature_cols),
        "n_samples":          len(df_model),
        "cv_folds":           int(cv_results_df["fold"].max()),
        "cv_MAE_mean":        round(float(cv_results_df["MAE"].mean()),  4),
        "cv_MAE_std":         round(float(cv_results_df["MAE"].std()),   4),
        "cv_RMSE_mean":       round(float(cv_results_df["RMSE"].mean()), 4),
        "cv_RMSE_std":        round(float(cv_results_df["RMSE"].std()),  4),
        "cv_R2_mean":         round(float(cv_results_df["R2"].mean()),   4),
        "cv_R2_std":          round(float(cv_results_df["R2"].std()),    4),
        "oof_MAE":            round(mae_oof,  4),
        "oof_RMSE":           round(rmse_oof, 4),
        "oof_R2":             round(r2_oof,   4),
        "best_params":        used_params,
        "best_n_estimators":  int(final_model.n_estimators),
        "top10_features":     top_features,
        "feature_list":       feature_cols,
    }

    log.info("── evaluation_metrics_his05 ──")
    for k, v in metrics.items():
        if k not in ("feature_list", "top10_features"):
            log.info("  %-22s: %s", k, v)

    return metrics