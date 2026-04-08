from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from evidently.metrics import (ClassificationQualityMetric,
    ClassificationClassBalance,
    ClassificationConfusionMatrix,
    ClassificationQualityByClass,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,)
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestAccuracyScore,
    TestF1Score,
    TestNumberOfDriftedColumns,
    TestPrecisionScore,
    TestRecallScore,
    TestRocAuc,
    TestShareOfMissingValues,
)

from auditors.base_auditor import BaseAuditor
from config.column_mapping import get_column_mapping

class ModelAuditor(BaseAuditor):
    def __init__(self, model_path: str | Path, target_col: str, output_dir: str | Path, config: dict):
        super().__init__(output_dir)
        self.model_path = Path(model_path)
        self.target_col = target_col
        self.config = config["model_thresholds"]

    def _load_and_split(self, data_path) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_parquet(data_path)
        le = LabelEncoder()
        for col in df.select_dtypes(include="object").columns:
            df[col] = le.fit_transform(df[col].astype(str))
        if self.target_col not in df.columns:
            np.random.seed(42)
            df[self.target_col] = np.random.randint(0, 2, size=len(df))
        split = int(len(df) * 0.6)
        return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
    
    def _load_model(self):
        with open(self.model_path, "rb") as f:
            return pickle.load(f)
        
    def _predict(self, df: pd.DataFrame, model) -> pd.DataFrame:
        features = [c for c in df.columns if c != self.target_col]
        df = df.copy()
        df["prediction"] = model.predict(df[features])
        return df
    
    def build_report(self) -> Report:
        return Report(metrics=[
            ClassificationQualityMetric(),
            ClassificationClassBalance(),
            ClassificationConfusionMatrix(),
            ClassificationQualityByClass(),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
        ])
    
    def build_test_suite(self) -> TestSuite:
        c = self.cfg
        return TestSuite(tests=[
            TestAccuracyScore(gte=c["accuracy_threshold"]),
            TestF1Score(gte=c["f1_threshold"]),
            TestPrecisionScore(gte=c["precision_threshold"]),
            TestRecallScore(gte=c["recall_threshold"]),
            TestRocAuc(gte=c["roc_auc_threshold"]),
            TestNumberOfDriftedColumns(lte=c["drifted_columns_threshold"]),
            TestShareOfMissingValues(lte=c["missing_values_threshold"]),
        ])
    
    def audit(self, data_path) -> dict:
        ref, cur = self._load_and_split(data_path)
        model = self._load_model()
        cur_preds = self._predict(cur, model)
        col_map = get_column_mapping(cur_preds, self.target_col)
        return self.run(ref, cur_preds, col_map, name=self.model_path.stem)
    
    def audit_all_models(config: dict) -> list[dict]:
        results = []
        data_path = config["paths"]["data"]
        models_dir = Path(config["paths"]["models_dir"])
        output_dir = Path(config["paths"]["reports"]) / "model_evaluation"

        for model_file, meta in config["model_registry"].items():
            print(f"\n▶ {model_file} [{meta['domain']}]")
            try:
                auditor = ModelAuditor(
                    model_path=models_dir / model_file,
                    target_col=meta["target"],
                    output_dir=output_dir,
                    config=config,
                )
                result = auditor.audit(data_path)
                results["domain"] = meta["domain"]
                status = "✅ PASSED" if result["passed"] else "❌ FAILED"
                print(f" {status} ok={result['success_tests']} failed={result['failed_tests']}")
            except Exception as e:
                print(f" ❌ ERROR: {e}")
                result = {"name": model_file, "domain": meta["domain"], "passed": False ,"error": str(e)}
            results.append(result)
        return results