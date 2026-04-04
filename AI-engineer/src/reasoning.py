import numpy as np
from huggingface_hub import InferenceClient
from src.config import HF_TOKEN, MODEL_ID, MAX_TOKENS

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# -- Banking context per problem --

FRAUD_RATE = 

PROBLEM_CONTEXT = {
    "fraud": {
        "business_goal": "Minimize financial losses from fraudulent transactions while reducing postives that hurt legitimate customers.",
        "key_features": ["RiskScore", "OperationalRiskScore", "Balance", "NumOfProducts", "IsActiveMember"],
        "business_kpis": ["fraud rate 3.32%", "churn rate 20.38%", "complain rate 20.44%"],
        "stakeholders"   : "Risk Management team, Compliance officers, Core Banking Operations",
        "regulation"     : "Basel III operational risk capital requirements, AML/KYC obligations",
    },
    "marketing": {
        "business_goal"  : "Maximize customer lifetime value through targeted campaigns using MarketingScore to prioritize high-value segments.",
        "key_features"   : ["HighValueCustomer", "MarketingScore", "Point Earned", "Satisfaction Score", "Card Type"],
        "business_kpis"  : ["high value customer 47.99%", "low satisfaction 39.46%", "churn rate 20.38%"],
        "stakeholders"   : "Marketing department, Customer Retention team, Product Managers",
        "regulation"     : "GDPR data usage for marketing, fair lending and anti-discrimination policies",
    },
    "operational": {
        "business_goal"  : "Assess and mitigate operational risk exposure across customer segments to maintain capital adequacy ratios.",
        "key_features"   : ["OperationalRiskScore", "AgeRisk", "BalancePerProduct", "LowCreditRisk", "Tenure"],
        "business_kpis"  : ["low credit risk segment", "operational risk score mean 1.23", "balance per product mean 33,603"],
        "stakeholders"   : "Chief Risk Officer, Internal Audit, Board Risk Committee",
        "regulation"     : "SR 11-7 Model Risk Management, EU AI Act Article 14, GDPR Article 22",
    },
}

def generate_reasoning(
        problem: str,
        best_model: str,
        ranked_models: list,
        df_summary: str,
        retriever
) -> dict:
    """Generate LLM-Powered banking reasoning for why a model is best,
    with retrieved research context from PDFs"""

    ctx = PROBLEM_CONTEXT[problem]
    top3 = ranked_models[:3]

    # Build ranked model performance text
    perf_lines = []
    for rank, m in enumerate(top3, start=1):
        score_str = ", ".join(f"{k}={v}" for k, v in m["metrics"].items() if v is not None)
        perf_lines.append(f" #{rank} {m['model']}: {score_str}")
    performance_text = "\n".join(perf_lines)

    # Retrieve relevant PDF chunks
    retrieval_query = (
        f"best machine learning model for {problem} detection in banking"
        f"with {best_model} performance explainability regulatory compliance"
    )
    source_docs = retriever.invoke(retrieval_query)