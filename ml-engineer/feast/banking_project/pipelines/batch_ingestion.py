from feast import FeatureStore
from datetime import datetime, timedelta

# Initialize the feature store
store = FeatureStore(repo_path="../feature_repo")

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

# materialize features for the past year
store.materialize(start_date=start_date, end_date=end_date)
print(f"Features materialized from {start_date} to {end_date}")