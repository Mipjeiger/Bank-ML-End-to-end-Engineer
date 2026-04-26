import pandas as pd
from config.config import DATA_PATH
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler

"""Feature engineering for dataframe has been loaded"""
class FeatureEngineering:    
    def __init__(self):
        self.data_path = DATA_PATH

    def load_data(self):
        df = pd.read_parquet(self.data_path)
        return df

    def feature_engineering(self, df):
        """Feature engineering to ensure the data is in the right format before being fed into the model for inference"""
        df = df.copy()
        df = df.dropna()

        # Encode categorical variables
        le = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = le.fit_transform(df[col])

        return df
    
    def split_data(self, df, target_col, test_size=0.2, random_state=42):
        """Split the data into training and testing sets"""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test
    
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
    
# Usage
feature_engineering = FeatureEngineering()