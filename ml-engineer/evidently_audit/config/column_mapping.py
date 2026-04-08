import numpy as np
import pandas as pd
from evidently import ColumnMapping

def get_column_mapping(df: pd.DataFrame, target_col: str, prediction_col: str = "prediction") -> ColumnMapping:
    """Get a column mapping for evidently based on the dataframe and specified target and prediction columns."""
    exclude = {target_col, prediction_col}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c not in exclude]
    return ColumnMapping(
        target=target_col,
        prediction=prediction_col,
        numerical_features=num_cols,
        categorical_features=cat_cols
    )

def get_text_column_mapping(numerical_features: list[str]) -> ColumnMapping:
    """Get a column mapping for evidently when working with text data, specifying only numerical features."""
    return ColumnMapping(
        target=None,
        prediction=None,
        numerical_features=numerical_features
    )