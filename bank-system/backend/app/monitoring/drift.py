from prometheus_client import Gauge
from backend.config.config import DATA_PATH
from backend.config.config import DATA_PATH
from .prometheus import REGISTRY, DATA_DRIFT_SCORE
from scipy.stats import ks_2samp
import pandas as pd

"""Data drift monitoring using Prometheus metrics."""
class DataDriftMonitor:
    def __init__(self):
        self.data_path = DATA_PATH
        self.drift_gauge = DATA_DRIFT_SCORE

    def calculate_drift(self, reference_data, new_data):
        """Calculate data drift using statistical test (e.g., Kolmogorov-Smirnov test) and return a drift score."""
        try:
            drift_scores = []
            for col in reference_data.columns:
                if reference_data[col].dtype in ['float64', 'int64']:
                    stat, _ = ks_2samp(reference_data[col], new_data[col])
                    drift_scores.append(stat)

            # Average drift score across all features
            average_drift_score = sum(drift_scores) / len(drift_scores) if drift_scores else 0
            return average_drift_score
        
        except Exception as e:
            print(f"Error calculating drift: {e}")
            return 0
    
    def monitor_drift(self):
        """Monitor data drift by comparing new data with reference data and updating prometheus Gauge."""
        try:
            reference_data = pd.read_parquet(self.data_path)
            new_data = pd.read_parquet(self.data_path) # In real implementation, this would be new incoming dataset

            drift_score = self.calculate_drift(reference_data, new_data)
            self.drift_gauge.set(drift_score)
            print(f"Data drift score updated: {drift_score}")
        except Exception as e:
            print(f"Error monitoring drift: {e}")
            