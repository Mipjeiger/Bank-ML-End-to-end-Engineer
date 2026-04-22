import streamlit as st
from feast import FeatureStore

store = FeatureStore(repo_path="../feature_repo")
st.title("🏦 Banking Feature Store Dashboard")

# List feature views
st.header("Feature Store Views")
feature_views = store.list_feature_views()

for fv in feature_views:
    st.subheader(fv.name)
    st.write("Entities:", fv.entities)
    st.write("Features:", [f.name for f in fv.schema])

# Test online features
st.header("Test Online Features")
customer_id = st.text_input("Customer ID", "ADX_1")

if st.button("Get Features"):
    response = store.get_online_features(
        features=[
            "customer_profile_features:CreditScore",
            "risk_features:OperationalRiskScore",
            "marketing_features:Tenure"
        ],
        entity_rows=[{"customer_id": customer_id}]
    ).to_dict()

    st.write(response)