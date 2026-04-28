import pandas as pd
import numpy as np
import logging
from config.config import DATA_PATH
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Create logger
logger = logging.getLogger(__name__)

"""Feature engineering for dataframe has been loaded"""
class FeatureEngineering:    
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path

        # Must be exactly 24 featyres - All keys must matcg CustomerData.dict()
        self.feature_order = [
            "customer_id",        # label-encoded
            "CreditScore",
            "Geography",          # label-encoded
            "Gender",             # label-encoded
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
            "Exited",
            "Complain",
            "SatisfactionScore",  # matches CustomerData field exactly
            "CardType",           # label-encoded
            "PointEarned",        # matches CustomerData field exactly
            "RiskScore",
            "BalancePerProduct",
            "AgeRisk",
            "HighValueCustomer",
            "LowCreditRisk",
            "ComplainFlag",
            "LowSatisfaction",
            "Fraud",              # not in schema — always defaults to 0
        ] 
        self.categorical_cols = ["Geography", "Gender", "CardType", "customer_id"]
        self.label_encoders = {col: LabelEncoder() for col in self.categorical_cols}

        # Defaults for optional/missing fields
        self.defaults = {
            "Fraud": 0.0,
            "BalancePerProduct": 0.0,
            "AgeRisk": 0.0,
            "HighValueCustomer": 0.0,
            "LowCreditRisk": 0.0,
            "ComplainFlag": 0.0,
            "LowSatisfaction": 0.0,
        }

    def _safe_float(self, value, default=0.0):
        """Safely convert any value to float"""
        try:
            if isinstance(value, bool):
                return float(int(value))
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value)
            return float(default)
        except:
            logger.warning(f"Could not convert {value!r} to float, using default {default}")
            return float(default)

    # ====== Feature Engineering Methods ======
    def load_data(self):
        df = pd.read_parquet(self.data_path)
        return df

    def feature_engineering(self, df):
        """Feature engineering to ensure the data is in the right format before being fed into the model for inference"""
        df = df.copy().dropna()

        # Encode categorical variables
        for col in self.categorical_cols:
            if col in df.columns:
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))

        return df
    
    def split_data(self, df, target_col, test_size=0.2, random_state=42):
        """Split the data into training and testing sets"""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def smote_resampling(self, X_train, y_train, random_state=42):
        """SMOTE resampling to handle class imbalance in the training data"""
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        return X_train_resampled, y_train_resampled
    
    def scaled_features(self, X_train, X_test):
        """Scale features using StandardScaler to prevent features with larger ranges from dominating the model in training process."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    # ===== Inference Feature Engineering =====
    def transform(self, data: dict) -> np.ndarray:
        """Transform customer data dict to feature array for inference"""
        if not isinstance(data, dict):
            raise ValueError(f"Input data should be a dictionary, got {type(data).__name__}. Ensure calling data.dict() on Pydantic models")
        
        features = []

        for col in self.feature_order:
            value = data.get(col, None)
            default = self.defaults.get(col, 0.0)

            if col in self.categorical_cols:
                str_value = str(value) if value is not None else "Unknown"
                try:
                    encoded = self.label_encoders[col].transform([str(value)])[0]
                    features.append(float(encoded))
                except Exception as e:
                    logger.warning(f"Could not encode '{col}' with value '{str_value}': {e}. Fitting on the fly - fit encoder on training data for production.")
                    self.label_encoders[col].fit([str_value]) # Fit on the fly to handle unseen categories during inference, but ideally should be fit on training data and saved for production use.
                    features.append(0.0)
            else:
                features.append(self._safe_float(value if value is not None else default, default))

        if len(features) != 24:
            raise ValueError(
                f"Feature count mismatch: expected 24 features, but got {len(features)}."
                "Check the feature engineering pipeline and ensure all features are in the correct order which are trained in the model."
            )
        
        logger.debug(f"Transformed features: ({len(features)}): {features}")
        return np.array([features], dtype=np.float64)

    def transform_batch(self, data_list: list[dict]) -> np.ndarray:
        """Transform batch of customer data for inference"""
        return np.vstack([self.transform(d) for d in data_list])

# Usage
feature_engineering = FeatureEngineering()