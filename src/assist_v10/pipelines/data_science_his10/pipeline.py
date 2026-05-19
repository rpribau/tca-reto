"""Kedro pipeline definition for HIS-10 Data Science layer.

This pipeline takes the preprocessed HIS-10 dataset produced by the
Data Engineering pipeline, applies feature engineering, trains a LightGBM
model with Optuna hyperparameter optimisation, and evaluates the best
model.  All experiments are tracked with MLflow.

Usage
-----
    kedro run --pipeline data_science_his10
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    evaluate_model,
    preprocess_features,
    split_data,
    train_model_with_optuna,
)


def create_pipeline(**kwargs) -> Pipeline:  # noqa: ARG001
    return pipeline(
        [
            node(
                func=preprocess_features,
                inputs="preprocessed_data_his10",
                outputs="his10_model_input",
                name="feature_engineering_node",
            ),
            node(
                func=split_data,
                inputs=["his10_model_input", "params:his10"],
                outputs="his10_split_data",
                name="split_data_node",
            ),
            node(
                func=train_model_with_optuna,
                inputs=["his10_split_data", "params:his10"],
                outputs=["trained_model_his10", "his10_test_data"],
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=[
                    "trained_model_his10",
                    "his10_test_data",
                    "params:his10",
                ],
                outputs="evaluation_metrics_his10",
                name="evaluate_model_node",
            ),
        ]
    )
