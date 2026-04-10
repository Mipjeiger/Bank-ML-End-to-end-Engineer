from kfp import dsl

@dsl.component(base_image="python:3.11")
def preprocess(input_data: str) -> str:
    import pandas as pd
    df = pd.read_csv(input_data)
    # Drop target column
    X = df.drop(columns=['Exited'], errors='ignore')
    output = '/tmp/preprocessed_data.csv.gz'
    X.to_csv(output, index=False, compression='gzip')
    return output