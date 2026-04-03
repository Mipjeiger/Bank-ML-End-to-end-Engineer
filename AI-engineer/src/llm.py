from huggingface_hub import InferenceClient
from src.config import HF_TOKEN, MODEL_ID, MAX_TOKENS, TEMPERATURE

# Define a client for Hugging Face Inference API
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

def build_prompt(context: str, question: str, df_context: str) -> str:
    return f"""You are a senior banking data scientist and AI consulant.
    
=== DATASET SUMMARY ===
{df_context}
=== END DATASET SUMMARY ===

=== RESEARCH CONTEXT ===
{context}
=== END RESEARCH CONTEXT ===

Question: {question}
Answer (be specific, reference dataset metrics when relevant, and provide actionable insights for a banking executive):"""

def ask(question: str, retriever, df_summary: str) -> dict:
    # Retrieve chunks
    source_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in source_docs])

    # Build prompt & call LLM
    prompt = build_prompt(context, question.strip(), df_summary)
    response = client.chat_completion(
        messages = [{"role": "user", "content": prompt}],
        max_tokens = MAX_TOKENS,
        temperature = TEMPERATURE
    )
    answer = response.choices[0].message.content.strip()
    sources = list({d.metadata.get("source", "unknown") for d in source_docs})

    return {"answer": answer, "sources": sources}