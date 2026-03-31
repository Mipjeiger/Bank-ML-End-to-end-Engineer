import gradio as gr
import requests

# ================================
# CONFIGURATION
# ================================
API_URL = "http://127.0.0.1:8000" # API URL for the FastAPI backend

MODEL_OPTIONS = [
    "Logistic Regression",
    "Random Forest",
    "Decision Tree",
    "XGBoost",
    "KNN"
]

GEOGRAPHY_OPTIONS = ["France", "Spain", "Germany"]
GENDER_OPTIONS = ["Male", "Female"]
CARD_OPTIONS = ["DIAMOND", "GOLD", "PLATINUM", "SILVER"]

# ================================
# CORE FUNCTIONS
# ================================

def call_api(endpoint, payload, model_name):
    try:
        response = requests.post(
            f"{API_URL}{endpoint}",
            json=payload,
            params={"model_name": model_name}
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        result = response.json()

        return (
            f"📊 Prediction: {result['prediction']}\n"
            f"📈 Probability: {result['probability']:.4f}\n"
            f"🤖 Model: {result['model_used']}\n"
            f"💡 Reasoning: {result.get('reasoning', 'N/A')}"
        )
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
# ================================
# MARKETING FUNCTION
# ================================

def predict_marketing(*inputs):
    payload = build_payload(inputs)
    return call_api("/predict/marketing/Marketing", payload, inputs[-1])

# ================================
# OPERATIONAL FUNCTION
# ================================

def predict_operational(*inputs):
    payload = build_payload(inputs)
    return call_api("/predict/operational-risk/Operational", payload, inputs[-1])

# ================================
# PAYLOAD BUILDER
# ================================

def build_payload(inputs):
    return {
        "CreditScore": inputs[0],
        "Geography": inputs[1],
        "Gender": inputs[2],
        "Age": inputs[3],
        "Tenure": inputs[4],
        "Balance": inputs[5],
        "NumOfProducts": inputs[6],
        "HasCrCard": inputs[7],
        "IsActiveMember": inputs[8],
        "EstimatedSalary": inputs[9],
        "Exited": inputs[10],
        "Complain": inputs[11],
        "SatisfactionScore": inputs[12],
        "CardType": inputs[13],
        "PointEarned": inputs[14],
        "RiskScore": inputs[15],
        "BalancePerProduct": inputs[16],
        "AgeRisk": inputs[17],
        "HighValueCustomer": inputs[18],
        "LowCreditRisk": inputs[19],
        "ComplainFlag": inputs[20],
        "LowSatisfaction": inputs[21]
    }

# ================================
# UI INPUT COMPONENTS
# ================================

inputs = [
    gr.Number(label="Credit Score"),
    gr.Dropdown(GEOGRAPHY_OPTIONS, label="Geography"),
    gr.Dropdown(GENDER_OPTIONS, label="Gender"),
    gr.Number(label="Age"),
    gr.Number(label="Tenure"),
    gr.Number(label="Balance"),
    gr.Number(label="Number of Products"),
    gr.Checkbox(label="Has Credit Card"),
    gr.Number(label="Is Active Member (0/1)"),
    gr.Number(label="Estimated Salary"),
    gr.Checkbox(label="Exited"),
    gr.Number(label="Complain (0/1)"),
    gr.Number(label="Satisfaction Score (1-5)"),
    gr.Dropdown(CARD_OPTIONS, label="Card Type"),
    gr.Number(label="Points Earned"),
    gr.Number(label="Risk Score"),
    gr.Number(label="Balance Per Product"),
    gr.Number(label="Age Risk"),
    gr.Number(label="High Value Customer (0/1)"),
    gr.Number(label="Low Credit Risk (0/1)"),
    gr.Number(label="Complain Flag (0/1)"),
    gr.Number(label="Low Satisfaction (0/1)"),
    gr.Dropdown(MODEL_OPTIONS, label="Model Selection")
]

# ================================
# BUILD UI (USER INTERFACE)
# ================================

with gr.Blocks(title="🏦 Fintech Intelligence System") as demo:

    gr.Markdown("# 🏦 Fintech Intelligence System")
    gr.Markdown("### Marketing & Operational Risk Prediction")

    with gr.Tab("📈 Marketing Prediction"):
        gr.Interface(
            fn=predict_marketing,
            inputs=inputs,
            outputs=gr.Textbox(label="Result")
        )

    with gr.Tab("⚙️ Operational Risk Prediction"):
        gr.Interface(
            fn=predict_operational,
            inputs=inputs,
            outputs=gr.Textbox(label="Result")
        )

# ================================
# LAUNCH THE APP
# ================================

if __name__ == "__main__":
    demo.launch(server_port=7860)