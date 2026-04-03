from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.integration import load_dataframe, build_retriever
from src.llm import ask
from src.report_parser import REPORT_DATA

from src.model_loader import evaluate_model, evaluate_all_problems, MODEL_REGISTRY
# Shared state
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources once at startup."""
    print("🚀 Loading resources...")
    df, df_summary = load_dataframe()
    state["df"] = df
    state["df_summary"] = df_summary
    state["retriever"] = build_retriever()

    print("Evaluating models...")
    state["model_results"] = evaluate_all_problems()
    print("✅ Models evaluated.")
    yield
    state.clear()

app = FastAPI(
    title="🏦 Banking LLM Insights API",
    description="RAG pipeline over banking research PDFs report + parquet dataset",
    version="1.0.0",
    lifespan=lifespan
)

# Schemas
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

class ProblemRequest(BaseModel):
    problem: str # "fraud" | "marketing" | "operational"

# Routes
@app.get("/")
def root():
    return {"status": "ok",
            "message": "Banking LLM Insights API is running"}

@app.get("/report")
def get_report():
    """Get the parsed insights report."""
    return REPORT_DATA

@app.get("/report/insights")
def get_all_insights():
    """Return all insight titles."""
    return [{"id": i["id"], "title": i["title"]} for i in REPORT_DATA["insights"]]

@app.get("/report/insights/{insight_id}")
def get_insight(insight_id: int):
    """Return a specific insight by ID."""
    for insight in REPORT_DATA["insights"]:
        if insight["id"] == insight_id:
            return insight
    raise HTTPException(status_code=404, detail=f"Insight {insight_id} not found")

@app.get("/dataset/summary")
def dataset_summary():
    """Return live dataset statistics."""
    df = state["df"]
    return {
        "rows"          : len(df),
        "columns"       : df.shape[1],
        "churn_rate"    : f"{df['Exited'].mean()*100:.2f}%",
        "fraud_rate"    : f"{df['Fraud'].mean()*100:.2f}%",
        "complain_rate" : f"{df['Complain'].mean()*100:.2f}%",
    }

@app.get("/models")
def list_models():
    """List all available models in the registry."""
    return {problem: list(models.keys()) for problem, models in MODEL_REGISTRY.items()}

@app.get("/models/{problem}")
def get_best_model(problem: str):
    """Return ranked model performance + best mdel recommendation
    for a specific problem: fraud | marketing | operational."""
    if problem not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Invalid problem '{problem}'. Problem must be one of {list(MODEL_REGISTRY.keys())}")
    # return cached result from startup evaluation
    return state["model_results"][problem]

@app.post("/ask", response_model=AnswerResponse)
def ask_question(body: QuestionRequest):
    """Ask any custom question - runs live RAG pipeline."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = ask(body.question, state["retriever"], state["df_summary"])
    return AnswerResponse(**result)

@app.post("/ask/insight", response_model=AnswerResponse)
def ask_insight(insight_id: int, body: QuestionRequest):
    """Ask a question in the context of a specific insight."""
    insight = next((i for i in REPORT_DATA["insights"] if i["id"] == insight_id), None)
    if not insight:
        raise HTTPException(status_code=404, detail=f"Insight {insight_id} not found")
    context = f"Insight {insight_id}: {insight['title']}\n\n{insight['content']}"
    result = ask(body.question, state["retriever"], state["df_summary"])
    return AnswerResponse(**result)

@app.post("/summary_report", response_model=AnswerResponse)
def summary_report(body: QuestionRequest):
    """Retrieve a summary report based on the question and dataset context."""
    context = f"Summary Report Request:\n{body.question}\n\nDataset Summary:\n{state['df_summary']}"
    result = ask(body.question, state["retriever"], state["df_summary"])
    return AnswerResponse(**result)

# Post endpoint -> 
