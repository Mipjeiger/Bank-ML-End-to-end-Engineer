import numpy as np
import pandas as pd
import pytest

from auditors.data_auditor import DataAuditor
from config import load_config

@pytest.fixture
def tiny_parquet(tmp_path):
    np.random.seed(1)
    n = 300
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n),
        "balance": np.random.randn(n) * 1000,
        "fraud_label": np.random.randint(0, 2, size=n)
    })
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    return path

def test_data_auditor_runs(tmp_path, tiny_parquet):
    auditor = DataAuditor(tmp_path / "out", load_config(), target_col="fraud_label")
    result = auditor.audit(tiny_parquet)
    assert "passed" in result
    assert (tmp_path / "out" / "data_audit_report.html").exists()