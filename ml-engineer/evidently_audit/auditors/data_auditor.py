from __future__ import annotations
from pathlib import Path

import pandas as pd

from evidently.metrics import (
    ColumnCorrelationsMetric,
    ColumnDistributionMetric,
    ColumnDriftMetric,
    ColumnMissingValuesMetric,
    ColumnQuantileMetric,
    ColumnSummaryMetric,
    DataDriftTable,
    DatasetDriftMetric,
    DatasetDuplicatedRowsMetric,
    DatasetMissingValuesMetric,
    DatasetSummaryMetric,
)

from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestColumnsType,
    TestHighlyCorrelatedColumns,
    TestNumberOfColumns,
    TestNumberOfDriftedColumns,
    TestNumberOfDuplicatedRows,
    TestNumberOfRows,
    TestShareOfMissingValues,
)

from auditors.base_auditor import BaseAuditor
from config.column_mapping import get_column_mapping

# Create a class for data auditing using Evidently
class DataAuditor(BaseAuditor):
    def __init__(self, output_dir: str | Path, config: dict, target_col: str | None = None):
        super().__init__(output_dir)
        self.cfg = config["data_thresholds"]
        self.target_col = target_col

    def _load_and_split(self, data_path, split_ratio=0.6):
        df = pd.read_parquet(data_path)
        split = int(len(df) * split_ratio)
        print(f" {df.shape[0]:,} rows x {df.shape[1]} columns | reference={split:,} current={len(df)-split:,}")
        return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)
    
    def build_report(self) -> Report:
        return Report(metrics=[
            DatasetSummaryMetric(),
            DatasetMissingValuesMetric(),
            DatasetDuplicatedRowsMetric(),
            DatasetDriftMetric(stattest_threshold=self.cfg["drift_stattest_threshold"]),
            DataDriftTable(),
        ])
    
    def build_test_suite(self) -> TestSuite:
        c = self.cfg
        return TestSuite(tests=[
            TestNumberOfColumns(),
            TestColumnsType(),
            TestNumberOfRows(gte=c["min_rows"]),
            TestNumberOfDuplicatedRows(lte=c["max_duplicate_rows"]),
            TestShareOfMissingValues(lte=c["max_missing_share"]),
            TestNumberOfDriftedColumns(lte=c["max_drifted_columns"]),
            TestHighlyCorrelatedColumns(max_number=10),
        ])
    
    def _column_detail_report(self, ref, cur, col_map):
        num_cols = col_map.numerical_features or []
        metrics = []
        for col in num_cols[:20]:
            metrics += [
                ColumnSummaryMetric(column_name=col),
                ColumnMissingValuesMetric(column_name=col),
                ColumnDriftMetric(column_name=col),
                ColumnDistributionMetric(column_name=col),
                ColumnQuantileMetric(column_name=col, quantiles=[0.25, 0.5, 0.75]),
            ]
        if not metrics:
            return
        r = Report(metrics=metrics)
        r.run(reference_data=ref, current_data=cur, column_mapping=col_map)
        r.save_html(self.output_dir / "column_details_report.html")

    def _correlation_report(self, ref, cur, col_map):
        num_cols = col_map.numerical_features or []
        if len(num_cols) < 2:
            return
        r = Report(metrics=[ColumnCorrelationsMetric(column_name=c) for c in num_cols[:15]])
        r.run(reference_data=ref, current_data=cur, column_mapping=col_map)
        r.save_html(self.output_dir / "correlation_report.html")

    def audit(self, data_path, split_ratio=0.6) -> dict:
        ref, cur = self._load_and_split(data_path, split_ratio)
        col_map = get_column_mapping(ref, target_col=self.target_col or "NA", prediction_col="NA")
        col_map.target = self.target_col
        col_map.prediction = None

        self._column_detail_report(ref, cur, col_map)
        self._correlation_report(ref, cur, col_map)
        return self.run(ref, cur, col_map, name="data_audit")