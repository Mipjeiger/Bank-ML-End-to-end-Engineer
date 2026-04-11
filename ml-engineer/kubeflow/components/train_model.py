from kfp import dsl

@dsl.component(base_image="python:3.11", 
               packages_to_install=["scikit-learn", "xgboost"])
def train_model(
    preprocessed_data_path: str,
    model_output_path: str
):
    import pandas as pd
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV

    # Load preprocessed data
    df = pd.read_csv(preprocessed_data_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train all models
    model = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    "Decision Tree":       DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "Random Forest":       RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    "XGBoost":             XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, scale_pos_weight=3),
    "KNN":                 KNeighborsClassifier()
}
    
    # Hyperparameters for GridSearchCV
    param_grid = {
    "Logistic Regression": {"C": [0.001, 0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]},
    "Decision Tree":       {"max_depth": [5, 10, 15, 20], "min_samples_split": [5, 10, 20], "min_samples_leaf": [2, 4, 8]},
    "Random Forest":       {"n_estimators": [100, 200, 300], "max_depth": [5, 10, 15], "min_samples_split": [5, 10], "min_samples_leaf": [2, 4]},
    "XGBoost":             {"n_estimators": [100, 200], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.7, 0.8, 1.0]},
    "KNN":                 {"n_neighbors": [3, 5, 7, 9], "weights": ["uniform", "distance"]}
}
    
    # Train all models with GridSearchCV
    best_models = {}
    for name, clf in model.items():
        print(f"Training {name}...")
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid[name], cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")

    # Save all models to the specified output path
    with open(model_output_path, 'wb') as f:
        pickle.dump(best_models, f)
    print(f"Models saved to {model_output_path}")