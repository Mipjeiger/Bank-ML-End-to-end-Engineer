from __future__ import annotations
from pathlib import Path
import pandas as pd

from evidently import Report
from evidently.metrics import ColumnSummaryMetric

from auditors.base_auditor import BaseAuditor
from config.column_mapping import get_text_column_mapping


DESCRIPTOR_COLS = [
    "response_length",
    "response_word_count",
    "prompt_length",
    "references_dataset",
    "is_refusal",
    "mentions_model",
    "answer_overlap_score",
]


def compute_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["response_length"] = df["llm_response"].str.len()
    df["response_word_count"] = df["llm_response"].str.split().str.len()
    df["prompt_length"] = df["prompt"].str.len()

    df["references_dataset"] = df["llm_response"].str.contains(
        "dataset|metric|rate|column|mean|std", case=False, regex=True
    ).astype(int)

    df["is_refusal"] = df["llm_response"].str.contains(
        "cannot|don't know|unable|not enough", case=False, regex=True
    ).astype(int)

    df["mentions_model"] = df["llm_response"].str.contains(
        "xgboost|random forest|logistic|knn|tree", case=False, regex=True
    ).astype(int)

    def overlap(a, b):
        a_w, b_w = set(str(a).lower().split()), set(str(b).lower().split())
        return len(a_w & b_w) / len(b_w) if b_w else 0

    df["answer_overlap_score"] = df.apply(
        lambda r: overlap(r["llm_response"], r["reference_answer"]), axis=1
    )

    return df


class LLMAuditor(BaseAuditor):
    def __init__(self, output_dir: str | Path, config: dict):
        super().__init__(output_dir, config)
        self.cfg = config["llm_thresholds"]

    def build_report(self) -> Report:
        return Report(metrics=[
            ColumnSummaryMetric(column_name=c) for c in DESCRIPTOR_COLS
        ])

    def evaluate_thresholds(self, report_dict: dict) -> dict:
        df = self.latest_df
        cfg = self.cfg

        checks = {
            "min_length": df["response_length"].min() >= cfg["min_response_length"],
            "max_length": df["response_length"].max() <= cfg["max_response_length"],
            "refusal_rate": df["is_refusal"].mean() <= cfg["max_refusal_rate"],
            "grounding": df["references_dataset"].mean() >= cfg["min_grounding_rate"],
            "overlap": df["answer_overlap_score"].mean() >= cfg["min_overlap_score"],
        }

        return {"passed": all(checks.values()), **checks}

    def audit(self, eval_csv):
        df = pd.read_csv(eval_csv)
        df = compute_descriptors(df)

        self.latest_df = df

        half = len(df) // 2
        ref = df.iloc[:half]
        cur = df.iloc[half:]

        col_map = get_text_column_mapping(DESCRIPTOR_COLS)

        return self.run(ref, cur, col_map, name="llm_audit")