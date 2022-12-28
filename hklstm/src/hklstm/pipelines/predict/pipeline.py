"""
This is a boilerplate pipeline 'predict'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import plot_predictions


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                plot_predictions,
                ["trained_params", "valid_ds", "params:SEQ_LEN"],
                "output_plot",
            )
        ]
    )
