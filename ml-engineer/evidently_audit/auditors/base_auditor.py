from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from evidently import Report


class BaseAuditor(ABC):
    def __init__(self, output_dir: str | Path, config: dict):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

    @abstractmethod
    def build_report(self) -> Report: ...

    @abstractmethod
    def evaluate_thresholds(self, report_dict: dict) -> dict: ...

    def run(self, reference, current, column_mapping, name: str = "audit") -> dict:
        report = self.build_report()

        report.run(
            reference_data=reference,
            current_data=current,
            data_definition=column_mapping
        )

        report_path = self.output_dir / f"{name}_report.html"
        report.save_html(str(report_path))

        result_dict = report.as_dict()

        threshold_result = self.evaluate_thresholds(result_dict)

        return {
            "name": name,
            "passed": threshold_result["passed"],
            "details": threshold_result,
            "report_path": str(report_path),
        }