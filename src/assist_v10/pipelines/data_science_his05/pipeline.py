"""
HIS-05 · Monitor de Saturación — Pipeline de Data Science
==========================================================
Uso:
    kedro run --pipeline data_science_his05
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    build_features_node,
    evaluate_model_node,
    hyperparameter_tuning_node,
    train_model_node,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # ── Nodo 1: Feature engineering ──────────────────────────────────
            node(
                func=build_features_node,
                inputs=["master_table_his05", "params:his05"],
                outputs=["his05_features", "his05_feature_cols"],
                name="his05_build_features",
                tags=["his05", "features"],
            ),

            # ── Nodo 2: Búsqueda de hiperparámetros ──────────────────────────
            node(
                func=hyperparameter_tuning_node,
                inputs=["his05_features", "his05_feature_cols", "params:his05"],
                outputs="his05_best_params",
                name="his05_hyperparameter_tuning",
                tags=["his05", "tuning"],
            ),

            # ── Nodo 3: Entrenamiento del modelo final ────────────────────────
            node(
                func=train_model_node,
                inputs=[
                    "his05_features",
                    "his05_feature_cols",
                    "his05_best_params",
                    "params:his05",
                ],
                outputs=[
                    "trained_model_his05",   # nombre del catálogo existente
                    "his05_cv_results",
                    "his05_oof_preds",
                    "his05_y_array",
                ],
                name="his05_train_model",
                tags=["his05", "training"],
            ),

            # ── Nodo 4: Evaluación y métricas ────────────────────────────────
            node(
                func=evaluate_model_node,
                inputs=[
                    "trained_model_his05",
                    "his05_cv_results",
                    "his05_oof_preds",
                    "his05_y_array",
                    "his05_features",
                    "his05_feature_cols",
                    "his05_best_params",
                    "params:his05",
                ],
                outputs="evaluation_metrics_his05",
                name="his05_evaluate_model",
                tags=["his05", "evaluation"],
            ),
        ]
    )