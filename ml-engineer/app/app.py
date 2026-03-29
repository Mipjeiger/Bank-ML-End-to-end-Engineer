import gradio as gr
import pandas as pd
import joblib
import os
from fastapi import FastAPI, HTTPException
from pathlib import Path

"""Using Gradio to create a User Interface (UI) web for production implementation of the model."""