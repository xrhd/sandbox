"""
This is a boilerplate pipeline 'train'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([node(train_model, ["train_ds", "valid_ds"], "trained_params")])
