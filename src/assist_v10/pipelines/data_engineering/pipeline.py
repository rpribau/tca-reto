"""
Kedro pipeline definition for the Data Engineering layer.

This pipeline receives raw hospital tables from the Kedro catalog, cleans them,
and creates the base datasets required by the HIS-05 and HIS-10 modeling pipelines.
"""

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
    """
    Create the Data Engineering pipeline.

    Pipeline flow:
    1. Clean raw input tables individually.
    2. Create HIS-10 base dataset for no-show classification.
    3. Create HIS-05 master table for saturation / wait-time forecasting.

    Outputs:
    - processed_* tables in data/02_intermediate.
    - preprocessed_data_his10 in data/03_primary.
    - master_table_his05 in data/03_primary.
    """

    return pipeline(
        [
            # Clean patient / encounter table.
            node(
                func=clean_hospac,
                inputs="hospac",
                outputs="processed_hospac",
                name="clean_hospac_node",
            ),

            # Clean appointment schedule table.
            # This is the main source for HIS-10.
            node(
                func=clean_hosagd,
                inputs="hosagd",
                outputs="processed_hosagd",
                name="clean_hosagd_node",
            ),

            # Clean master patient index table.
            # Used to enrich HIS-10 with demographic variables.
            node(
                func=clean_hosmpi,
                inputs="hosmpi",
                outputs="processed_hosmpi",
                name="clean_hosmpi_node",
            ),

            # Clean emergency triage records.
            # Used as enrichment for HIS-05.
            node(
                func=clean_triage,
                inputs="triage",
                outputs="processed_triage",
                name="clean_triage_node",
            ),

            # Clean emergency medical notes.
            # Used to build hourly demand and wait-time proxy for HIS-05.
            node(
                func=clean_notamedicaurg,
                inputs="notamedicaurg",
                outputs="processed_notamedicaurg",
                name="clean_notamedicaurg_node",
            ),

            # Create modeling base table for HIS-10.
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

            # Create modeling base table for HIS-05.
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
