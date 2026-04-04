import numpy as np
from huggingface_hub import InferenceClient
from src.config import HF_TOKEN, MODEL_ID, MAX_TOKENS

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# -- Banking context per problem --
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

def reasoning_engine(
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
    research_context = "\n\n".join(d.page_content for d in source_docs)
    sources = list({d.metadata.get("source", "unknown") for d in source_docs})

    # Prompt
    prompt = f"""You are a Chief AI Officer at a banking institution writing an internal model selection report.
    
=== DATASET CONTEXT ===
{df_summary}
=== PROBLEM CONTEXT ===
    

=== RESEARCH LITERATURE ===
{research_context}
=== END RESEARCH LITERATURE ===

=== MODEL EVALUATION RESULTS ===
Problem: {problem.upper()} - {ctx['business_goal']}
Key Features: {', '.join(ctx['key_features'])}
Business KPIs: {', '.join(ctx['business_kpis'])}
Stakeholders: {ctx['stakeholders']}
Regulatory: {ctx['regulation']}

Top 3 Models by Performance:
{performance_text}

Selected Best: {best_model}
=== END EVALUATION RESULTS ===

Write a structured internal reasoning report with thexe exact sections:

1. EXECUTIVE SUMMARY
    Why {best_model} is selected for {problem} in our bank (2-3 sentences, cite actual metrics).

2. PERFORMANCE JUSTIFICATION
    Explain why {best_model} outperforms the other models specifically for our {problem} problem,
    referencing actual metric values and dataset KPIs.

3. BUSINESS IMPACT
    Quantify the expected business impact for our bank:
    - Revenue protection or cost savings from improved {problem} detection
    - Customer experience improvement
    - Risk exposure reduction
    Use our actual dataset stats (fraud rate, churn rate, etc.).

4. REGULATORY & COMPLIANCE FIT
    How does {best_model} align with {ctx['regulation']}?
    Address explainability requirements for {ctx['stakeholders']}.

5. LIMITATIONS & MITIGATIONS
    What are the known limitations of {best_model} for {problem}?
    What mitigation strategies should the bank implement?

6. DEPLOYMENT RECOMMENDATION
   Concrete next steps for deploying {best_model} in production at our bank.
   Necessary monitoring, retraining {best_model} for data drift if applicable.

Be specific, use numbers from the dataset, and reference research findings where relevant.
Write for a banking executive audience — avoid generic statements."""
   
    # Call LLM
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.3
    )
    reasoning_text = response.choices[0].message.content.strip()

    # Parse sections
    sections = _parse_sections(reasoning_text)

    return {
        "reasoning_text": reasoning_text,
        "sections": sections,
        "sources": sources,
        "retrieval_query": retrieval_query
    }

def _parse_sections(text: str) -> dict:
    """Extract numbered sections from LLM output into a clean dict."""
    import re

    section_keys = {
        "1": "executive_summary",
        "2": "performance_justification",
        "3": "business_impact",
        "4": "regulatory_compliance_fit",
        "5": "limitations_mitigations",
        "6": "deployment_recommendation",
    }

    sections = {}
    pattern = r"(\d+)\.\s+[A-Z &]+\n(.*?)(?=\n\d+\.\s+[A-Z]|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    for num, content in matches:
        key = section_keys.get(num.strip())
        if key:
            sections[key] = content.strip()

    # Fallback - if regex fails, return raw text split by newlines
    if not sections:
        sections["full_reasoning"] = text

    return sections