from feast import FeatureStore
import joblib

# Define feature store
store = FeatureStore(repo_path="../feature_repo")

# Config model paths
fraud_model = joblib.load("../models/fraud_model.pkl")
marketing_model = joblib.load("../models/marketing_model.pkl")
operational_model = joblib.load("../models/operational_model.pkl")

# Create function for fraud prediction
def predict_fraud(customer_id: int):
    response = store.get_online_features(
        features=[
            "customer_profile_features:CreditScore",
            "customer_profile_features:Age",
            "customer_profile_features:Balance",
            "fraud_features:RiskScore",
            "fraud_features:ComplainFlag",
            "fraud_features:LowSatisfaction"
        ],
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()

    FEATURE_ORDER = [
        "CreditScore", "Age", "Balance",
        "RiskScore", "ComplainFlag", "LowSatisfaction"
    ]

    values = [response[f][0] for f in FEATURE_ORDER]
    prediction = fraud_model.predict([values])

    return int(prediction[0])

# Create function for marketing prediction
def predict_marketing(customer_id: int):
    feature_vector = store.get_online_features(
        features=[
            "customer_profile_features:CreditScore",
            "customer_profile_features:Balance",
            "customer_profile_features:NumOfProducts",
            "marketing_features:IsActiveMember",
            "marketing_features:Tenure",
            "marketing_features:HighValueCustomer"
        ],
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()

    FEATURE_ORDER = [
        "CreditScore", "Balance", "NumOfProducts",
        "IsActiveMember", "Tenure", "HighValueCustomer"
    ]

    values = [feature_vector[f][0] for f in FEATURE_ORDER]
    prediction = marketing_model.predict([values])

    return int(prediction[0])

# Create function for operational prediction
def predict_operational(customer_id: int):
    feature_vector = store.get_online_features(
        features=[
            "risk_features:Satisfaction Score",
            "risk_features:AgeRisk",
            "risk_features:OperationalRiskScore",
            "risk_features:LowCreditRisk"
        ],
        entity_rows=[{"customer_id": customer_id}],
    ).to_dict()

    FEATURE_ORDER = [
        "Satisfaction Score", "AgeRisk",
        "OperationalRiskScore", "LowCreditRisk"
    ]

    values = [feature_vector[f][0] for f in FEATURE_ORDER]
    prediction = operational_model.predict([values])

    return int(prediction[0])