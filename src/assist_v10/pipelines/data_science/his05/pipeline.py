# src/assist_v10/pipelines/data_science/his05/pipeline.py
from kedro.pipeline import Pipeline, node, pipeline

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_transformer_model,
            inputs=["master_table_his05", "params:model_options"],
            outputs="trained_model_his05",
            name="train_transformer_node",
        ),
        node(
            func=evaluate_transformer_model,
            inputs=["trained_model_his05", "test_data_his05"],
            outputs="evaluation_metrics_his05",
            name="evaluate_transformer_node",
        ),
    ])