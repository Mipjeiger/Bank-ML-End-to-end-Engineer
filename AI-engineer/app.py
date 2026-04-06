import gradio as gr
import requests

API_URL = "http://localhost:8000"


def fetch_report_insights():
    resp = requests.get(f"{API_URL}/report/insights")
    if resp.ok:
        return "\n".join([f"  {i['id']}. {i['title']}" for i in resp.json()])
    return "❌ Could not load insights."


def fetch_insight(insight_id: int):
    try:
        resp = requests.get(f"{API_URL}/report/insights/{insight_id}", timeout=10)
        if resp.ok:
            data = resp.json()
            return f"### {data['title']}\n\n{data['content']}"
        return f"❌ Error {resp.status_code}: {resp.text}"
    except Exception as e:
        return f"❌ Cannot reach API: {e}"


def ask_question(question: str):
    if not question.strip():
        return "⚠️ Please enter a question.", []
    try:
        resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60, headers={"Content-Type": "application/json"})
        if resp.ok:
            data    = resp.json()
            sources = "\n".join([f"📄 {s}" for s in data["sources"]])
            return data["answer"], sources
        return f"❌ Error: {resp.text}", ""
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to API. Is uvicorn running on port 8000?", ""
    except requests.exceptions.Timeout:
        return "❌ API request timed out. LLM is taking too long.", ""
    except Exception as e:
        return f"❌ An error occurred: {e}", ""

def fetch_dataset_summary():
    try:
        resp = requests.get(f"{API_URL}/dataset/summary", timeout=10)
        if resp.ok:
            d = resp.json()
            return (f"**Rows:** {d['rows']:,} | **Columns:** {d['columns']}\n\n"
                    f"**Churn Rate:** {d['churn_rate']} | "
                    f"**Fraud Rate:** {d['fraud_rate']} | "
                    f"**Complain Rate:** {d['complain_rate']}")
        return f"⚠️ API returned {resp.status_code}"
    except Exception as e:
        return f"❌ Cannot reach API: {e}"

# ── UI ────────────────────────────────────────────────────────
with gr.Blocks(title="🏦 Banking LLM Insights", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🏦 Banking LLM Insights Dashboard")
    gr.Markdown("RAG pipeline over banking research PDFs · Powered by Llama-3 8B")

    # Dataset summary banner
    with gr.Row():
        summary_box = gr.Markdown(value=fetch_dataset_summary())

    gr.Markdown("---")

    with gr.Tabs():

        # Tab 1: Pre-generated insights
        with gr.Tab("📋 Report Insights"):
            gr.Markdown("### Pre-generated insights from the LLM pipeline")
            insight_id = gr.Slider(minimum=1, maximum=5, step=1,
                                   value=1, label="Select Insight (1–5)")
            load_btn   = gr.Button("Load Insight", variant="primary")
            insight_out = gr.Markdown()
            load_btn.click(fn=fetch_insight, inputs=insight_id, outputs=insight_out)

        # Tab 2: Live Q&A
        with gr.Tab("💬 Ask a Question"):
            gr.Markdown("### Ask anything — runs live RAG pipeline")
            question_input = gr.Textbox(
                lines       = 3,
                placeholder = "e.g. What features most predict customer churn?",
                label       = "Your Question",
            )
            ask_btn    = gr.Button("Ask", variant="primary")
            answer_out = gr.Markdown(label="Answer")
            source_out = gr.Textbox(label="📚 Sources", interactive=False)
            
            ask_btn.click(
                fn      = ask_question,
                inputs  = question_input,
                outputs = [answer_out, source_out],
            )

            # Quick question examples
            gr.Examples(
                examples = [
                    ["What are the main drivers of customer churn?"],
                    ["How can LLMs improve fraud detection in banking?"],
                    ["What explainability techniques are recommended for risk scores?"],
                    ["Which customer segments should be prioritized for retention?"],
                ],
                inputs = question_input,
            )

if __name__ == "__main__":
    demo.launch(server_port=7860)