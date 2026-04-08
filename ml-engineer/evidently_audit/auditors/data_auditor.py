from __future__ import annotations
from pathlib import Path
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataQualityPreset, DataDriftPreset

from auditors.base_auditor import BaseAuditor
from config.column_mapping import get_column_mapping


class DataAuditor(BaseAuditor):
    def __init__(self, output_dir: str | Path, config: dict, target_col: str | None = None):
        super().__init__(output_dir, config)
        self.cfg = config["data_thresholds"]
        self.target_col = target_col

    def _load_and_split(self, data_path, split_ratio=0.6):
        df = pd.read_parquet(data_path)
        split = int(len(df) * split_ratio)
        return (
            df.iloc[:split].reset_index(drop=True),
            df.iloc[split:].reset_index(drop=True),
        )

    def build_report(self) -> Report:
        return Report(metrics=[
            DataQualityPreset(),
            DataDriftPreset(),
        ])

    def evaluate_thresholds(self, report_dict: dict) -> dict:
        cfg = self.cfg
        passed = True

        drifted = 0
        missing = 0

        for metric in report_dict.get("metrics", []):
            result = metric.get("result", {})

            if "number_of_drifted_columns" in result:
                drifted = result.get("number_of_drifted_columns", 0)

            if "current" in result:
                missing = result.get("current", {}).get("share_of_missing_values", missing)

        if drifted > cfg["max_drifted_columns"]:
            passed = False

        if missing > cfg["max_missing_share"]:
            passed = False

        return {
            "passed": passed,
            "drifted_columns": drifted,
            "missing_share": missing,
        }

    def audit(self, data_path, split_ratio=0.6) -> dict:
        ref, cur = self._load_and_split(data_path, split_ratio)

        col_map = get_column_mapping(
            ref,
            target_col=self.target_col,
        )

        return self.run(ref, cur, col_map, name="data_audit")