"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from assist_v10.pipelines.data_engineering import pipeline as de

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_engineering_pipeline = de.create_pipeline()

    return {
        "__default__": data_engineering_pipeline,
        "data_engineering": data_engineering_pipeline,
    }