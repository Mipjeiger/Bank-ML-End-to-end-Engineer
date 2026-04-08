from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from evidently.metrics import ColumnSummaryMetric
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestColumnValueMax,
    TestColumnValueMean,
    TestColumnValueMin,
    TestShareOfOutRangeValues,
)

from auditors.base_auditor import BaseAuditor
from config.column_mapping import get_text_column_mapping

# TODO: add exact prompt templates from dir ml-engineer/notebooks/03_LLM.ipynb
def build_prompt(context: str, question: str, df_context: str) -> str:
    return f"""You are a senior banking data scientist and AI consultant.
    You have access to research literature on AI/ML in banking and a real dataset summary below.
=== DATASET SUMMARY ===
{df_context}
=== END DATASET SUMMARY ===
=== RESEARCH CONTEXT ===
{context}
=== END RESEARCH CONTEXT ===
Question: {question}
Answer (be specific, reference dataset metrics when relevant):"""

def build_df_context(parquet_path) -> str:
    df = pd.read_parquet(parquet_path)
    lines = [
        f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns",
        f"Columns: {', '.join(df.columns.tolist())}",
        "",
        "Numerical summary:",
        df.describe().round(3).to_string(),
        "",
        "Missing values (top 5):",
        df.isnull().sum().sort_values(ascending=False).head(5).to_string(),
    ]
    for col in df.select_dtypes(include="object").columns[:5]:
        lines.append(f" {col}: {df[col].value_counts().head(3).to_string()}")
    return "\n".join(lines)

# Descriptor functions to extract features from LLM responses for testing
DESCRIPTOR_COLS = [
    "response_length", "response_word_count", "prompt_length",
    "references_dataset", "is_refusal", "mentions_model", "answer_overlap_score",
]

def compute_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df["response_length"] = df["llm_response"].str.len()
        df["response_word_count"] = df["llm_response"].str.split().str.len()
        df["prompt_length"] = df["prompt"].str.len()

        dataset_kw = ["dataset", "column", "row", "metric", "rate", "%", "mean", "std", "fraud", "churn"]
        df["references_dataset"] = df["llm_response"].apply(lambda t: int(any(k in t.lower() for k in dataset_kw)))

        refusal_kw = ["cannot determine", "i don't know", "not enough", "unable to", "no information"]
        df["is_refusal"] = df["llm_response"].apply(lambda t: int(any(k in t.lower() for k in refusal_kw)))

        model_kw = ["xgboost", "random forest", "decision tree", "logistic", "knn", "gradient boost"]
        df["mentions_model"] = df["llm_response"].apply(lambda t: int(any(k in t.lower() for k in model_kw)))

        def overlap(a, b):
            a_w, b_w = set(a.lower().split()), set(b.lower().split())
            return len(a_w & b_w) / len(b_w) if b_w else 0.0
        
        df["answer_overlap_score"] = df.apply(lambda r: overlap(r["llm_response"], r["reference_answer"]), axis=1)
        return df
    except Exception as e:
        raise ValueError (f"Error computing descriptors: {str(e)} | maybe the column names are different? Expected: {DESCRIPTOR_COLS} | Check the columns: {df.columns.tolist()} is not relevant to the LLM response evaluation.")

# Create class for audit LLM responses
class LLMAuditor(BaseAuditor):
    def __init__(self, output_dir: str | Path, config: dict):
        super().__init__(output_dir)
        self.cfg = config["llm_thresholds"]

    def build_report(self) -> Report:
        return Report(metrics=[
            ColumnSummaryMetric(column_name=c) for c in DESCRIPTOR_COLS
        ])
    
    def build_test_suite(self) -> TestSuite:
        c = self.cfg
        return TestSuite(tests=[
            TestColumnValueMin(column_name="response_length",      gte=c["min_response_length"]),
            TestColumnValueMax(column_name="response_length",      lte=c["max_response_length"]),
            TestColumnValueMean(column_name="is_refusal",          lte=c["max_refusal_rate"]),
            TestColumnValueMean(column_name="references_dataset",  gte=c["min_grounding_rate"]),
            TestColumnValueMean(column_name="answer_overlap_score", gte=c["min_overlap_score"]),
            TestShareOfOutRangeValues(column_name="response_word_count", left=5, right=300),
        ])
    
    def audit_prompt_structure(self, df_context: str) -> dict:
        sample = build_prompt("Sample research.", "What is fraud rate?", df_context)
        checks = {
            "has_role_instruction": "You are a senior banking data scientist" in sample,
            "has_dataset_section": "DATASET SUMMARY" in sample,
            "has_research_section": "RESEARCH CONTEXT" in sample,
            "has_question_placeholder": "Question:" in sample,
            "has_answer_cue": "Answer" in sample,
            "has_grounding_instruction": "reference dataset metrics" in sample,
            "prompt_char_length": len(sample),
            "prompt_word_count": len(sample.split()),
        }
        out = self.output_dir / "prompt_structure_audit.json"
        with open(out, "w") as f:
            json.dump(checks, f, indent=2)
        for k, v in checks.items():
            icon = "✅" if v is True else ("ℹ" if isinstance(v, int) else "❌")
            print(f" {icon} {k}: {v}")
        return checks
    
    def audit(self, eval_df: pd.DataFrame | None = None, eval_csv=None, data_path=None) -> dict:
        if eval_df is not None:
            df = eval_df.copy()
        elif eval_csv and Path(eval_csv).exists():
            df = pd.read_csv(eval_csv)
        else:
            raise ValueError("Provide eval_df or a valid eval_csv path.")
        
        df = compute_descriptors(df)
        half = len(df) // 2
        ref = df.iloc[:half].reset_index(drop=True)
        cur = df.iloc[half:].reset_index(drop=True)

        if data_path and Path(data_path).exists():
            self.audit_prompt_structure(build_df_context(data_path))

        return self.run(ref, cur, get_text_column_mapping(DESCRIPTOR_COLS), name="llm_audit")
    
    def build_eval_dataset(df_context: str, n: int = 30) -> pd.DataFrame:
        """Builds a synthetic evaluationd dataset.
        Replace the simulated responses with real model.generate() calls in production."""
        qa_pairs = [
            ("What is the fraud detection rate in the dataset?",
            "Fraud detection in banking uses ML classifiers. Precision-recall tradeoffs are critical.",
            "The dataset shows a fraud label distribution. High precision minimises false positives."),
            ("Which ML model performs best for credit default prediction?",
            "XGBoost and Random Forest outperform logistic regression on imbalanced datasets.",
            "XGBoost typically achieves the highest ROC-AUC for credit default tasks."),
            ("How should missing values be handled in operational banking data?",
            "Imputation must preserve regulatory compliance and not introduce synthetic bias.",
            "Use median for numerical and mode for categorical. Flag missingness as a binary indicator."),
            ("What is the churn rate and how can it be reduced?",
            "Customer churn in retail banking averages 15-20% annually.",
            "Churn column shows the distribution. Targeted retention can reduce churn by 10-15%."),
            ("Explain the class imbalance issue in fraud detection.",
            "Fraud datasets are heavily imbalanced with fraud rates below 1-2%.",
            "Requires resampling or class-weight adjustment. Use F1 and PR-AUC, not accuracy."),
        ] * (n // 5 + 1)

        options = [
            lambda r, _: f"Based on the dataset, {r}",
            lambda r, _: "The answer is unclear from the provided context.",
            lambda r, _: f"According to the research context, {r} The dataset metrics confirm this.",
            lambda r, _: "I cannot determine this from the provided context.",
            lambda r, _: r,
        ]

        rows = []
        for i, (q, ctx, ref_ans) in enumerate(qa_pairs[:n]):
            rows.append({
                "id": i,
                "question": q,
                "context": ctx,
                "prompt": build_prompt(ctx, q, df_context),
                "llm_response": options[i % len(options)](ref_ans, ctx),
                "reference_answer": ref_ans,
            })
        return pd.DataFrame(rows)