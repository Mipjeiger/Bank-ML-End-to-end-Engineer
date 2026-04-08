import pandas as pd
from auditors.llm_auditor import LLMAuditor, build_prompt, build_eval_dataset, compute_descriptors
from config import load_config

def test_build_prompt():
    p = build_prompt("Some research", "What is fraud rate?", "Shape: 1000 rows")
    assert "DATASET SUMMARY" in p
    assert "RESEARCH QUESTION" in p
    assert "Question:" in p

def test_compute_descriptors():
    df = pd.DataFrame({
        "question": ["What is fraud rate?"],
        "context": ["Research"],
        "prompt": ["Full promp text."],
        "llm_response": ["Based on the dataset, fraud rate is 2%."],
        "reference_answer": ["The fraud rate is appromimately 2% in the dataset."],
    })
    df = compute_descriptors(df)
    assert "response_length" in df.columns
    assert 0.0 <= df["answer_overlap_score"].iloc[0] <= 1.0

def test_llm_auditor_runs(tmp_path):
    df_ctx = "Shape: 1000 rows x 10 columns"
    eval_df = build_eval_dataset(df_ctx, n=20)
    auditor = LLMAuditor(tmp_path / "out", load_config())
    result = auditor.audit(eval_df=eval_df)
    assert "passed" in result
    assert (tmp_path / "out" / "llm_audit_report.html").exists()