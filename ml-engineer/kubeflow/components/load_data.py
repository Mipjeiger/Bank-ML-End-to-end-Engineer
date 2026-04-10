from kfp import dsl
import pandas as pd

@dsl.component(base_image="python:3.11")
def load_data(data_path: str) -> str:
    df = pd.read_csv(data_path)
    output = '/tmp/data.csv.gz'
    df.to_csv(output, index=False, compression='gzip')
    return output