import pandas as pd
import torch
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sources.config import DATA_PATH, PDF_PATHS

# Create a function to load the dataset
def load_dataframe() -> tuple[pd.DataFrame, str]:
    df = pd.read_parquet(DATA_PATH)

    lines = [f"Dataset: {len(df)} rows, {df.shape[1]} columns."]
    for col, label in [
        ("Exited", "Churn Rate"),
        ("Fraud", "Fraud Rate"),
        ("Complain", "Complain Rate"),
        ("HighValueCustomer", "High Value Customer %"),
        ("LowSatisfaction", "Low Satisfaction %"),
        ("CreditScore", "Average Credit Score"),
        ("Balance", "Average Balance")
    ]:
        if col in df.columns:
            lines.append(f"{label}: {df[col].mean() * 100:.2f}%")
    
    num_cols = [c for c in df.columns if df[c].dtype in ['int64', 'float64']]
    existing = [c for c in num_cols if c in df.columns]
    lines.append(f"\nNumerical Summary:\n{df[existing].describe().round(2).to_string()}")

    return df, "\n".join(lines)

# Create a function to build the retriever
def build_retriever():
    # Load PDFs
    all_docs = []
    for path in tqdm(PDF_PATHS, desc="📂 Loading PDFs"):
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = path.name
        all_docs.extend(docs)

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = []
    for doc in tqdm(all_docs, desc="✂️ Chunking Documents"):
        chunks.extend(splitter.split_documents([doc]))

    # Embed + FAISS for Ollama
    embeddings = OllamaEmbeddings(model="gemma3:4b", 
                                  encode_kwargs={"normalize_embeddings": True},
                                  model_kwargs={"device": "mps" if torch.backends.mps.is_available() else "auto"})
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 4})