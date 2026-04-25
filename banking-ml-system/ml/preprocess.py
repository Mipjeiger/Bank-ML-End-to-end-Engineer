import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df: pd.DataFrame):
    # Load dataset
    df = df.copy()
    df = df.drop(columns=["customer_id"], errors="ignore")  # Drop rows with missing customer_id
    df = df.fillna(0) # Fill missing values with 0 value

    # Encode categorical variables
    for col in df.select_dtypes(include=["object", "category"]).columns:
        df[col] = df[col].astype("category").cat.codes

    return df