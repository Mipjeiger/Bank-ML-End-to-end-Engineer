from __future__ import annotations
import numpy as np
import pandas as pd
from evidently import DataDefinition


def get_column_mapping(
    df: pd.DataFrame,
    target_col: str | None = None,
    prediction_col: str | None = "prediction",
) -> DataDefinition:
    """
    Build Evidently DataDefinition from dataframe.
    Compatible with Evidently >= 0.7.x
    """

    exclude = set(filter(None, [target_col, prediction_col]))

    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if c not in exclude
    ]

    return DataDefinition(
        numerical_columns=num_cols,
        categorical_columns=cat_cols,
        target=target_col,
        prediction=prediction_col,
    )


def get_text_column_mapping(numerical_features: list[str]) -> DataDefinition:
    """
    For LLM evaluation where descriptors are numeric
    """
    return DataDefinition(
        numerical_columns=numerical_features,
        categorical_columns=[],
        target=None,
        prediction=None,
    )