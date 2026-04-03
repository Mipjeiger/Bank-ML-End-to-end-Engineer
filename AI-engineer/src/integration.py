import pandas as pd
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import DATA_PATH, PDF_PATHS, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

def load_dataframe() -> tuple[pd.DataFrame, str]:
    df = pd.read_parquet(DATA_PATH)

    lines = [f"Dataset: {len(df)} rows, {df.shape[1]} columns."]
    for col, label in [
        ("Exited", "Chrun Rate"),
        ("Fraud", "Fraud Rate"),
        ("Complain", "Complaint Rate"),
        ("HighValueCustomer", "High Value Customer %"),
        ("LowSatisfaction", "Low Satisfaction %")
    ]:
        if col in df.columns:
            lines.append(f"{label}: {df[col].mean() * 100:.2f}%")

    num_cols = ["CreditScore","Age","Balance","EstimatedSalary",
                "RiskScore","BalancePerProduct","MarketingScore",
                "OperationalRiskScore","Point Earned","Satisfaction Score"]
    existing = [c for c in num_cols if c in df.columns]
    lines.append(f"\nNumerical Summary:\n{df[existing].describe().round(2).to_string()}")

    return df, "\n".join(lines)

def build_retriever():
    # Load PDFs
    all_docs = []
    for path in tqdm(PDF_PATHS, desc="📄 Loading PDFs"):
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = path.name
        all_docs.extend(docs)
    
    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = []
    for doc in tqdm(all_docs, desc="✂️ Chunking Documents"):
        chunks.extend(splitter.split_documents([doc]))

    # Embed + FAISS