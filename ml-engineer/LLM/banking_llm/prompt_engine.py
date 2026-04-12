def build_prompt(question, context):
    return f"""
You are a banking data scientist & AI consultant assistant.

Context:
{context}

Question:
{question}

Answer clearly with business insight and actionable recommendations:
"""