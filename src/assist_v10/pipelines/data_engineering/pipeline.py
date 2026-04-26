from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    clean_hospac,
    clean_hosagd,
    clean_hosmpi,
    clean_triage,
    clean_notamedicaurg,
    create_his10_base,
    create_his05_master_table,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_hospac,
                inputs="hospac",
                outputs="processed_hospac",
                name="clean_hospac_node",
            ),
            node(
                func=clean_hosagd,
                inputs="hosagd",
                outputs="processed_hosagd",
                name="clean_hosagd_node",
            ),
            node(
                func=clean_hosmpi,
                inputs="hosmpi",
                outputs="processed_hosmpi",
                name="clean_hosmpi_node",
            ),
            node(
                func=clean_triage,
                inputs="triage",
                outputs="processed_triage",
                name="clean_triage_node",
            ),
            node(
                func=clean_notamedicaurg,
                inputs="notamedicaurg",
                outputs="processed_notamedicaurg",
                name="clean_notamedicaurg_node",
            ),
            node(
                func=create_his10_base,
                inputs=[
                    "processed_hosagd",
                    "processed_hospac",
                    "processed_hosmpi",
                ],
                outputs="preprocessed_data_his10",
                name="create_his10_base_node",
            ),
            node(
                func=create_his05_master_table,
                inputs=[
                    "processed_notamedicaurg",
                    "processed_triage",
                ],
                outputs="master_table_his05",
                name="create_his05_master_table_node",
            ),
        ]
    )