from fastapi import FastAPI
from app.schemas import CustomerData
from app.inference import ModelInference
from monitoring.drift import 