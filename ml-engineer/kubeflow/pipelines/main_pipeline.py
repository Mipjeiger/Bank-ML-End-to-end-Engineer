import sys
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.load_data import load_data
from components.preprocess import preprocess
from components.train_model import train_model
from components.evaluate_model import evaluate_model

from kfp import dsl, compiler

@dsl.pipeline(name='banking-ml-pipeline')
def banking_pipeline(
    data_path: str = 'Database/data/eda_banking.csv.gz',
    models_dir: str = 'ml-engineer/models/banking_models/models'
):
    load_op = load_data(data_path=data_path)
    preprocess_op = preprocess(input_data=load_op.output)
    train_op = train_model(input_data=preprocess_op.output, models_dir=models_dir)
    eval_op = evaluate_model(model_path=train_op.output, test_data_path=preprocess_op.output)

if __name__ == '__main__':
    compiler.Compiler().compile(banking_pipeline, 'banking_pipeline.yaml')
    print("✓ Pipeline compiled")