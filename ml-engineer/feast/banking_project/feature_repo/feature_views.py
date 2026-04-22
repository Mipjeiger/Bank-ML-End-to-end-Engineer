from feast import FeatureView, Field
from feast.types import Float32, Int64, Bool
from datetime import timedelta

from data_sources import data_source
from entities import customer

customer_profile_features = FeatureView(
    name="customer_profile_features",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="CreditScore", dtype=Int64),
        Field(name="Age", dtype=Int64),
        Field(name="Balance", dtype=Float32),
        Field(name="EstimatedSalary", dtype=Float32),
        Field(name="NumOfProducts", dtype=Int64),
    ],
    source=data_source,
)

risk_features = FeatureView(
    name="risk_features",
    entities=[customer],
    ttl=timedelta(days=7),
    schema=[
        Field(name="Satisfaction Score", dtype=Int64),
        Field(name="AgeRisk", dtype=Int64),
        Field(name="OperationalRiskScore", dtype=Int64),
        Field(name="LowCreditRisk", dtype=Int64),
    ],
    source=data_source
)

marketing_features = FeatureView(
    name="marketing_features",
    entities=[customer],
    ttl=timedelta(days=30),
    schema=[
        Field(name="IsActiveMember", dtype=Bool),
        Field(name="HasCrCard", dtype=Bool),
        Field(name="Tenure", dtype=Int64),
        Field(name="HighValueCustomer", dtype=Bool)
    ],
    source=data_source
)

fraud_features = FeatureView(
    name="fraud_features",
    entities=[customer],
    ttl=timedelta(days=7),
    schema=[
        Field(name="RiskScore", dtype=Int64),
        Field(name="ComplainFlag", dtype=Bool),
        Field(name="LowSatisfaction", dtype=Bool)
    ],
    source=data_source
)