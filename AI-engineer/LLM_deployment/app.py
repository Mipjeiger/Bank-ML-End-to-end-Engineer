import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_community.llms import ollama
from langchain_core.callbacks import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="LLM Deployment with Ollama")
MODEL_NAME = "tinyllama"

def get_llm():
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler])
    return ollama(model=MODEL_NAME, callback_manager=callback_manager)

class Question(BaseModel):
    text: str

# Create endpoint
@app.get("/")
def read_root():
    return {"Hello": f"Welcome to the {MODEL_NAME} LLM deployment!"}

@app.on_event("startup")
async def startup_event():
    logger.info(f"Starting up with model: {MODEL_NAME}")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the application.")