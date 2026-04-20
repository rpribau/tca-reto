from kedro.pipeline import Pipeline, pipeline, node
from .nodes import split_random_data, train_classification_model, evaluate_classification_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_random_data,
                inputs=["preprocessed_data_his10", "params:his10_split_params"],
                outputs=["X_train_rnd", "X_test_rnd", "y_train_rnd", "y_test_rnd"],
                name="split_random_node_his10",
            ),
            node(
                func=train_classification_model,
                inputs=["X_train_rnd", "y_train_rnd", "params:his10_model_params"],
                outputs="trained_model_his10",
                name="train_catboost_classifier_node_his10",
            ),
            node(
                func=evaluate_classification_model,
                inputs=["trained_model_his10", "X_test_rnd", "y_test_rnd"],
                outputs="evaluation_metrics_his10",
                name="evaluate_catboost_node_his10",
            ),
        ]
    )
