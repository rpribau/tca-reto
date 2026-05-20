"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

from assist_v10.pipelines.data_engineering import pipeline as de
from assist_v10.pipelines.data_science_his10 import pipeline as ds_his10
from assist_v10.pipelines.data_science_his05 import pipeline as ds_his05


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """
    data_engineering_pipeline = de.create_pipeline()
    data_science_his10_pipeline = ds_his10.create_pipeline()
    data_science_his05_pipeline = ds_his05.create_pipeline()

    return {
        "__default__": data_engineering_pipeline + data_science_his10_pipeline,
        "data_engineering": data_engineering_pipeline,
        "data_science_his10": data_science_his10_pipeline,
        "data_science_his05": data_science_his05_pipeline,
    }