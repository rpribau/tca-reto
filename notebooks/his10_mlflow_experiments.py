# ============================================================
# Experimentación con Optuna + MLflow para HIS-10 (No-Show Guard)
# ============================================================
#
# Este código está diseñado para pegarse como celda(s) al final
# del notebook models_his10.ipynb, DESPUÉS de las celdas que
# definen:
#   - his10_for_model = feature_engineering_his10(his10)
#   - X = his10_for_model.drop(columns=['no_show'])
#   - y = his10_for_model['no_show']
#
# Utiliza Optuna para buscar los mejores hiperparámetros de
# LightGBM y MLflow para registrar cada experimento.
# Al final guarda:
#   - El mejor modelo como .pkl en data/06_models/
#   - Las métricas de evaluación como .json en data/09_tracking/
#   - Un resumen de todos los trials como .csv en data/09_tracking/
#
# Para visualizar los resultados en MLflow UI:
#   1. Abre una terminal en la raíz del proyecto
#   2. Ejecuta: mlflow ui --backend-store-uri ./mlruns
#   3. Abre http://127.0.0.1:5000 en tu navegador
# ============================================================

import optuna
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ─── Configuración ───────────────────────────────────────────
N_TRIALS = 30                # Número de experimentos de Optuna
THRESHOLD = 0.5              # Umbral de clasificación
TEST_SIZE = 0.2              # Proporción de datos de prueba
RANDOM_STATE = 42
EXPERIMENT_NAME = "HIS10_NoShow_LightGBM"

# Rutas de salida (relativas a notebooks/)
PROJECT_ROOT = os.path.abspath("..")
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "data", "06_models", "trained_model_his10.pkl"
)
METRICS_PATH = os.path.join(
    PROJECT_ROOT, "data", "09_tracking", "evaluation_metrics_his10.json"
)
TRIALS_CSV_PATH = os.path.join(
    PROJECT_ROOT, "data", "09_tracking", "optuna_trials_his10.csv"
)


# ─── Configurar MLflow ───────────────────────────────────────
tracking_uri = "file:///" + MLRUNS_DIR.replace("\\", "/")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"MLflow tracking URI: {tracking_uri}")
print(f"Experimento: {EXPERIMENT_NAME}")


# ─── Preparar datos ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

negatives = int((y_train == 0).sum())
positives = int((y_train == 1).sum())
scale_weight = negatives / positives

print(f"\nDistribución de entrenamiento: 0s={negatives}, 1s={positives}")
print(f"scale_pos_weight calculado: {scale_weight:.4f}")
print(f"Trials a ejecutar: {N_TRIALS}\n")


# ─── Función de evaluación completa ──────────────────────────
def evaluate_model(model, X_eval, y_eval, threshold=0.5):
    """Evalúa el modelo y retorna un diccionario con todas las métricas."""
    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_eval, y_pred)

    return {
        "roc_auc": float(roc_auc_score(y_eval, y_proba)),
        "pr_auc": float(average_precision_score(y_eval, y_proba)),
        "precision": float(precision_score(y_eval, y_pred)),
        "recall": float(recall_score(y_eval, y_pred)),
        "f1_score": float(f1_score(y_eval, y_pred)),
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "threshold": threshold,
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
    }


# ─── Función objetivo de Optuna ──────────────────────────────
def objective(trial):
    """Función objetivo que Optuna intentará maximizar (ROC-AUC)."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2, log=True
        ),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-8, 10.0, log=True
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-8, 10.0, log=True
        ),
    }

    # Parámetros fijos (no los busca Optuna)
    params["scale_pos_weight"] = scale_weight
    params["random_state"] = RANDOM_STATE
    params["n_jobs"] = -1
    params["verbose"] = -1

    with mlflow.start_run(
        nested=True, run_name=f"trial_{trial.number:03d}"
    ):
        # Entrenar
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        # Evaluar con todas las métricas
        metrics = evaluate_model(model, X_test, y_test, THRESHOLD)

        # Registrar hiperparámetros en MLflow
        mlflow.log_params(
            {k: v for k, v in params.items() if k != "verbose"}
        )

        # Registrar TODAS las métricas en MLflow
        mlflow.log_metrics(
            {
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "accuracy": metrics["accuracy"],
            }
        )

        print(
            f"  Trial {trial.number:3d} │ "
            f"ROC-AUC: {metrics['roc_auc']:.4f} │ "
            f"F1: {metrics['f1_score']:.4f} │ "
            f"Recall: {metrics['recall']:.4f} │ "
            f"Precision: {metrics['precision']:.4f}"
        )

    return metrics["roc_auc"]


# ─── Ejecutar optimización ────────────────────────────────────
print("=" * 72)
print(" Iniciando optimización de hiperparámetros con Optuna + MLflow")
print("=" * 72)

study = optuna.create_study(
    direction="maximize", study_name=EXPERIMENT_NAME
)

run_name = f"Optuna_HPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name) as parent_run:

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    # ─── Mejor configuración encontrada ───────────────────
    print("\n" + "=" * 72)
    print(" MEJORES HIPERPARÁMETROS ENCONTRADOS")
    print("=" * 72)
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print(f"\n  Mejor ROC-AUC en búsqueda: {study.best_value:.4f}")

    # ─── Entrenar modelo final con los mejores parámetros ─
    print("\n" + "=" * 72)
    print(" Entrenando MODELO FINAL con los mejores hiperparámetros...")
    print("=" * 72)

    best_params = study.best_params.copy()
    best_params["scale_pos_weight"] = scale_weight
    best_params["random_state"] = RANDOM_STATE
    best_params["n_jobs"] = -1
    best_params["verbose"] = -1

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # Evaluación final completa
    final_metrics = evaluate_model(final_model, X_test, y_test, THRESHOLD)

    # Registrar parámetros y métricas del mejor modelo en el Run padre
    mlflow.log_params(
        {
            "best_" + k: v
            for k, v in best_params.items()
            if k != "verbose"
        }
    )
    mlflow.log_metrics(
        {
            "best_" + k: v
            for k, v in final_metrics.items()
            if isinstance(v, float)
        }
    )
    mlflow.log_metric("n_trials", N_TRIALS)

    # Registrar el modelo en MLflow (permite cargarlo después)
    mlflow.lightgbm.log_model(final_model, "best_lightgbm_model")

    # ─── Guardar .pkl del mejor modelo ────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    # También registrar el .pkl como artefacto en MLflow
    mlflow.log_artifact(MODEL_PATH, artifact_path="exported_model")

    # ─── Guardar métricas como JSON ───────────────────────
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    evaluation_output = {
        "experiment_name": EXPERIMENT_NAME,
        "mlflow_run_id": parent_run.info.run_id,
        "timestamp": datetime.now().isoformat(),
        "n_trials": N_TRIALS,
        "best_params": {
            k: (
                float(v)
                if isinstance(v, (int, float, np.integer, np.floating))
                else str(v)
            )
            for k, v in best_params.items()
            if k != "verbose"
        },
        "metrics": {
            "roc_auc": final_metrics["roc_auc"],
            "pr_auc": final_metrics["pr_auc"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1_score": final_metrics["f1_score"],
            "accuracy": final_metrics["accuracy"],
        },
        "confusion_matrix": {
            "tn": final_metrics["tn"],
            "fp": final_metrics["fp"],
            "fn": final_metrics["fn"],
            "tp": final_metrics["tp"],
        },
        "training_info": {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "n_features": int(X_train.shape[1]),
            "scale_pos_weight": float(scale_weight),
            "threshold": THRESHOLD,
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_output, f, indent=2, ensure_ascii=False)
    mlflow.log_artifact(METRICS_PATH, artifact_path="evaluation")

    # ─── Guardar resumen de todos los trials como CSV ─────
    trials_df = study.trials_dataframe()
    os.makedirs(os.path.dirname(TRIALS_CSV_PATH), exist_ok=True)
    trials_df.to_csv(TRIALS_CSV_PATH, index=False)
    mlflow.log_artifact(TRIALS_CSV_PATH, artifact_path="optuna_results")

    # ─── Resumen final ────────────────────────────────────
    print("\n" + "=" * 72)
    print(" RESULTADOS DEL MEJOR MODELO")
    print("=" * 72)
    print(f"  ROC-AUC:   {final_metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:    {final_metrics['pr_auc']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1-Score:  {final_metrics['f1_score']:.4f}")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"\n  Matriz de Confusión:")
    print(f"    TN={final_metrics['tn']:5d}  FP={final_metrics['fp']:5d}")
    print(f"    FN={final_metrics['fn']:5d}  TP={final_metrics['tp']:5d}")

    print("\n" + "=" * 72)
    print(" ARCHIVOS GUARDADOS")
    print("=" * 72)
    print(f"  Modelo (.pkl):     {MODEL_PATH}")
    print(f"  Métricas (.json):  {METRICS_PATH}")
    print(f"  Trials (.csv):     {TRIALS_CSV_PATH}")
    print(f"  MLflow tracking:   {MLRUNS_DIR}")

    print("\n" + "=" * 72)
    print(" PARA VISUALIZAR LOS EXPERIMENTOS EN MLFlow:")
    print("=" * 72)
    print(f"  1. Abre una terminal en la raíz del proyecto")
    print(f"  2. Ejecuta:")
    print(f"     mlflow ui --backend-store-uri {tracking_uri}")
    print(f"  3. Abre http://127.0.0.1:5000 en tu navegador")
    print("=" * 72)

print("\n¡Búsqueda de hiperparámetros y registro finalizados!")
