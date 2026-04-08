from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from evidently import Report
from evidently.metric_preset import ClassificationPreset

from auditors.base_auditor import BaseAuditor
from config.column_mapping import get_column_mapping


class ModelAuditor(BaseAuditor):
    def __init__(self, model_path: str | Path, target_col: str, output_dir: str | Path, config: dict):
        super().__init__(output_dir, config)
        self.model_path = Path(model_path)
        self.target_col = target_col
        self.cfg = config["model_thresholds"]

    def _load_and_split(self, data_path):
        df = pd.read_parquet(data_path)

        le = LabelEncoder()
        for col in df.select_dtypes(include="object").columns:
            df[col] = le.fit_transform(df[col].astype(str))

        if self.target_col not in df.columns:
            np.random.seed(42)
            df[self.target_col] = np.random.randint(0, 2, size=len(df))

        split = int(len(df) * 0.6)
        return df.iloc[:split], df.iloc[split:]

    def _load_model(self):
        with open(self.model_path, "rb") as f:
            return pickle.load(f)

    def _predict(self, df, model):
        features = [c for c in df.columns if c != self.target_col]
        df = df.copy()
        df["prediction"] = model.predict(df[features])
        return df

    def build_report(self) -> Report:
        return Report(metrics=[ClassificationPreset()])

    def evaluate_thresholds(self, report_dict: dict) -> dict:
        cfg = self.cfg
        passed = True

        accuracy = f1 = precision = recall = roc_auc = 0

        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {}).get("current", {})

            accuracy = result.get("accuracy", accuracy)
            f1 = result.get("f1", f1)
            precision = result.get("precision", precision)
            recall = result.get("recall", recall)
            roc_auc = result.get("roc_auc", roc_auc)

        if accuracy < cfg["accuracy"]: passed = False
        if f1 < cfg["f1"]: passed = False
        if precision < cfg["precision"]: passed = False
        if recall < cfg["recall"]: passed = False
        if roc_auc < cfg["roc_auc"]: passed = False

        return {
            "passed": passed,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "roc_auc": roc_auc,
        }

    def audit(self, data_path):
        ref, cur = self._load_and_split(data_path)
        model = self._load_model()
        cur = self._predict(cur, model)
        col_map = get_column_mapping(cur, self.target_col)
        return self.run(ref, cur, col_map, name=self.model_path.stem)


def audit_all_models(config: dict):
    results = []
    data_path = config["paths"]["data"]
    models_dir = Path(config["paths"]["models_dir"])
    output_dir = Path(config["paths"]["reports"]) / "model_evaluation"

    for model_file, meta in config["model_registry"].items():
        try:
            auditor = ModelAuditor(
                model_path=models_dir / model_file,
                target_col=meta["target"],
                output_dir=output_dir,
                config=config,
            )
            result = auditor.audit(data_path)
            result["domain"] = meta["domain"]
        except Exception as e:
            result = {"name": model_file, "domain": meta["domain"], "passed": False, "error": str(e)}

        results.append(result)

    return results