import re
from pathlib import Path
from src.config import REPORT_MD

def parse_report(md_path: Path = REPORT_MD) -> dict:
    """
    Parse the Banking_llm_insights_report.md into a structured dictionary format.
    Returns: { "metadata": {...}, "insights": [ {title, content}, ... ] }"""
    text = md_path.read_text(encoding="utf-8")

    # Extract metadata from header lines
    metadata = {}
    dataset_match = re.search(r"\*\*Dataset:\*\*\s*(.+)", text)
    churn_match = re.search(r"\*\*Churn Rate:\*\*\s*(.+)", text)
    fraud_match = re.search(r"\*\*Fraud Rate:\*\*\s*(.+)", text)

    if dataset_match:
        metadata["dataset"] = dataset_match.group(1).strip()
    if churn_match:
        metadata["churn_rate"] = churn_match.group(1).strip()
    if fraud_match:
        metadata["fraud_rate"] = fraud_match.group(1).strip()

    # Extract insights sections by ## heading
    pattern = r"##\s+Insight\s+(\d+):\s+(.+?)\n(.*?)(?=\n##|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    insights = []
    for number, title, content in matches:
        insights.append({
            "id": int(number),
            "title": title.strip(),
            "content": content.strip()
        })

    return {
        "metadata": metadata,
        "insights": insights
    }

# Singleton - parsed once at startup
REPORT_DATA = parse_report()