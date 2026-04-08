from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path

import evidently
from evidently.report import Report
from evidently.test_suite import TestSuite
    
class BaseAuditor(ABC):
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def build_report(self) -> Report: ...

    @abstractmethod
    def build_test_suite(self) -> TestSuite: ...

    def run(self, reference, current, column_mapping, name: str = "audit") -> dict:
        report = self.build_report()
        test_suite = self.build_test_suite()

        report.run(reference_data=reference, current_data=current, column_mapping=column_mapping)
        test_suite.run(reference_data=reference, current_data=current, column_mapping=column_mapping)

        report.save_html(str(self.output_dir / f"{name}_report.html"))
        test_suite.save_html(str(self.output_dir / f"{name}_test_suite.html"))
        test_suite.save_json(str(self.output_dir / f"{name}_test_suite.json"))

        summary = test_suite.as_dict().get("summary", {})
        return {
            "name": name,
            "passed": summary.get("all_passed", False),
            "success_tests": summary.get("success_tests", 0),
            "failed_tests": summary.get("failed_tests", 0),
            "report_path": str(self.output_dir / f"{name}_report.html"),
            "suite_path": str(self.output_dir / f"{name}_test_suite.html"),
        }