import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df: pd.DataFrame):
    # Load dataset
    df = df.copy()

    # fill missing
    df = df.fillna(0)

    # Encode categorical variables
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df