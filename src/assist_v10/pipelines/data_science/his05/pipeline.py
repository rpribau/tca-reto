from kedro.pipeline import Pipeline, pipeline, node
from .nodes import split_time_series_data, train_forecasting_model, evaluate_forecasting_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_time_series_data,
                inputs=["preprocessed_data_his05", "params:his05_split_params"],
                outputs=["X_train_ts", "X_test_ts", "y_train_ts", "y_test_ts"],
                name="split_time_series_node_his05",
            ),
            node(
                func=train_forecasting_model,
                inputs=["X_train_ts", "y_train_ts", "params:his05_model_params"],
                outputs="trained_model_his05",
                name="train_xgboost_regressor_node_his05",
            ),
            node(
                func=evaluate_forecasting_model,
                inputs=["trained_model_his05", "X_test_ts", "y_test_ts"],
                outputs="evaluation_metrics_his05",
                name="evaluate_xgboost_node_his05",
            ),
        ]
    )
