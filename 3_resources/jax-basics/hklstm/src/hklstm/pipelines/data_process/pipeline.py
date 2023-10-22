"""
This is a boilerplate pipeline 'data_process'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import data_process, generate_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                generate_data,
                ["params:SEQ_LEN", "params:TRAIN_SIZE", "params:VALID_SIZE"],
                ["train", "valid"],
            ),
            node(
                data_process,
                ["train", "valid", "params:BATCH_SIZE"],
                ["train_ds", "valid_ds"],
            ),
        ]
    )
