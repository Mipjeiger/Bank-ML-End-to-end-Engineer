"""
python3 run_audit.py
python3 run_audit.py --audit model
python3 run_audit.py --audit data
python3 run_audit.py --audit LLM
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from config import load_config
from auditors.model_auditor import audit_all_models
from auditors.data_auditor import DataAuditor
from auditors.llm_auditor import LLMAuditor, build_df_context, build_eval_dataset

def run_model_audit(config):
    print("\n" + "="* 55)
    print(" AUDIT 1/3 - MODEL EVALUTATION")
    print("="* 55)
    return audit_all_models(config)

def run_data_audit(config):
    print("\n" + "="* 55)
    print(" AUDIT 2/3 - DATA QUALITY")
    print("="* 55)
    auditor = DataAuditor(
        output_dir=Path(config["paths"]["reports"]) / "data_evaluation",
        config=config,
        target_col="fraud_label"
    )
    result = auditor.audit(config["paths"]["data"], config["split_ratio"])
    print(f" {'✅ PASSED' if result['passed'] else '❌ FAILED'}"
          f"ok={result['success_tests']} fail={result['failed_tests']}")
    return result

def run_llm_audit(config):
    print("\n" + "═" * 55)
    print("  AUDIT 3/3 — LLM EVALUATION")
    print("═" * 55)
    data_path = config["paths"]["data"]
    eval_csv = Path(config["paths"]["llm_eval"])

    if not eval_csv.exists():
        print("Generating evaluation dataset...")
        df_context = build_df_context(data_path)
        eval_df = build_eval_dataset(df_context, n=30)
        eval_csv.parent.mkdir(parents=True, exist_ok=True)
        eval_df.to_csv(eval_csv, index=False)
    else:
        eval_df = pd.read_csv(eval_csv)
        df_context = build_df_context(data_path)

    auditor = LLMAuditor(
        output_dir=Path(config["paths"]["reports"]) / "llm_evaluation",
        config=config
    )
    result = auditor.audit(eval_df=eval_df, data_path=data_path)
    print(f" {'✅ PASSED' if result['passed'] else '❌ FAILED'}"
          f"ok={result['success_tests']} fail={result['failed_tests']}")
    return result

def save_summary(model_results, data_results, llm_result, config):
    rows = [
        {"audit": "model", "name": r.get("name"), "domain": r.get("domain"),
         "passed": r.get("passed"), "ok": r.get("success_tests"), "fail": r.get("failed_tests")}
         for r in model_results
    ]
    for label, r in [("data", data_results), ("llm", llm_result)]:
        rows.append({"audit": label, "name": label, "domain": label,
                     "passed": r.get("passed"), "ok": r.get("success_tests"), "fail": r.get("failed_tests")})
    df = pd.DataFrame(rows)
    out = Path(config["paths"]["reports"]) / "audit_summary.csv"
    df.to_csv(out, index=False)
    print("\n" + "═" * 55)
    print("  AUDIT SUMMARY")
    print("═" * 55)
    print(df.to_string(index=False))
    print(f"\n  Saved → {out}")

# Function for main execution
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", choices=["model", "data", "llm", "all"], default="all")
    args = parser.parse_args()
    config = load_config()
    start = time.time()

    model_results, data_results, llm_result = [], {}, {}

    if args.audit in ("model", "all"):
        model_results = run_model_audit(config)
    if args.audit in ("data", "all"):
        data_result = run_data_audit(config)
    if args.audit in ("llm", "all"):
        llm_result = run_llm_audit(config)
    if args.audit == "all":
        save_summary(model_results, data_result, llm_result, config)

    print(f"\n Done in {time.time() - start:.1f} seconds.")

# Usage to run
if __name__ == "__main__":
    main()