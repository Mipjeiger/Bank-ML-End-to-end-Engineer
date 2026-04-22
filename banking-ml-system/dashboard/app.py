import streamlit as st
import os
import json
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports" / "deepchecks"

# Setup streamlit
st.set_page_config(page_title="📊 Deepchecks Dashboard", layout="wide")
st.title("🏦 Banking ML Monitoring Dashboard")

# Debug information
st.caption(f"Looking for reports in: {REPORTS_DIR}")

# Ensure the dir is exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# List reports
files = list(REPORTS_DIR.glob("*.json"))

if not files:
    st.warning("No reports found.")
    st.stop()

selected = st.selectbox("Select Report", files, format_func=lambda x: x.name)

# Load JSON
with open(selected) as f:
    data = json.load(f)

# UI Display
st.subheader("📈 Summary")
st.json(data.get("summary", {}))
st.subheader("🔍 Checks")

checks = data.get("checks", [])

if not checks:
    st.info("No checks found in the report.")
else:
    for check in data.get("checks", []):
        st.markdown(f"### {check['name']}")
        st.write(check.get("summary", "No summary available."))

        # Optional: show full row data
        with st.expander("Show Full Check Data"):
            st.json(check)