import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports" / "deepchecks"
LOGS_DIR = BASE_DIR / "reports" / "logs"

# Setup streamlit
st.set_page_config(page_title="📊 Deepchecks Dashboard", layout="wide")
st.title("🏦 Banking ML Monitoring Dashboard")

# Ensure dirs exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

tab1, tab2 = st.tabs(["📊 Deepchecks Reports", "📈 Live Monitoring"])

# Tab 1: Deepchecks Reports
with tab1:
    st.header("📊 Deepchecks Reports")
    files = list(REPORTS_DIR.glob("*.json"))

    if not files:
        st.warning("No reports found.")
    else:
        selected = st.selectbox("Select Report", files, format_func=lambda x: x.name)

        with open(selected) as f:
            data = json.load(f)

        st.subheader("Summary")
        st.json(data.get("summary", {}))

        st.subheader("Checks")
        for check in data.get("checks", []):
            st.markdown(f"### {check['name']}")
            st.write(check.get("summary", ""))

# Tab 2: Live Monitoring
with tab2:
    st.header("📈 Live Predictions Monitoring")
    log_files = list(LOGS_DIR.glob("*.csv"))

    if not log_files:
        st.warning("No logs found.")
    else:
        latest_log = max(log_files, key=os.path.getctime)

        df = pd.read_csv(latest_log) # Load dataset from the latest log file

        # Create streamlit app to display the dataset
        st.subheader("Recent Predictions Log")
        st.dataframe(df.tail(20)) # Get the last 20 rows of the dataset

        # Statistics
        st.subheader("Prediction Distribution")
        if "prediction" in df.columns:
            st.bar_chart(df["prediction"].value_counts())
        
        st.subheader("Feature Distributions")
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns

        if len(numeric_cols) > 0:
            col = st.selectbox("Select Feature", numeric_cols)
            st.line_chart(df[col])