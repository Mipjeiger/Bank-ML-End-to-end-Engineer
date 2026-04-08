import pickle, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from auditors.model_auditor import ModelAuditor
from config import load_config

# Create @pytest.fixture -> to define functions that provide a reliable and consistent context for tests. 
@pytest.fixture
def tmp_model(tmp_path):
    np.random.seed(0)
    n = 200
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n),
        "balance": np.random.randn(n) * 1000,
        "fraud_label": np.random.randint(0, 2, size=n)
    })
    path = tmp_path / "test.parquet"
    df.to_parquet(path)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(df[["age", "balance"]], df["fraud_label"])
    model_path = tmp_path / "Decision_Tree_Fraud_Audit.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    return model_path, path

def test_model_auditor_runs(tmp_path, tmp_model):
    model_path, data_path = tmp_model
    auditor = ModelAuditor(model_path=model_path, target_column="fraud_label", output_path=tmp_path / "out", config=load_config())
    result = auditor.audit(data_path=data_path)
    assert "passed" in result
    assert (tmp_path / "out" / "Decision_Tree_Fraud_Audit_report.html").exists()