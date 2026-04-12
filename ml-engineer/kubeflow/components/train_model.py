from kfp import dsl
from kfp.dsl import Output, Model, Metrics

@dsl.component(
    base_image="python:3.11",
    packages_to_install=["pandas", "scikit-learn", "xgboost"]
)
def train_model(
    preprocessed_data_path: str,
    model: Output[Model],
    metrics: Output[Metrics]
):
    import pandas as pd
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import f1_score

    df = pd.read_csv(preprocessed_data_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(class_weight='balanced', n_jobs=-1),
        "XGBoost": XGBClassifier(eval_metric='logloss', verbosity=0, scale_pos_weight=3),
        "KNN": KNeighborsClassifier()
    }

    param_grid = {
        "Logistic Regression": {"C": [0.1, 1]},
        "Decision Tree": {"max_depth": [5, 10]},
        "Random Forest": {"n_estimators": [100, 200]},
        "XGBoost": {"n_estimators": [100], "max_depth": [3, 5]},
        "KNN": {"n_neighbors": [3, 5]}
    }

    best_model = None
    best_score = -1
    best_name = ""

    for name, clf in models.items():
        print(f"Training {name}...")

        grid = GridSearchCV(
            clf,
            param_grid[name],
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        preds = grid.best_estimator_.predict(X_val)
        score = f1_score(y_val, preds, average='weighted')

        print(f"{name} F1: {score:.4f}")

        # 🔥 log to Kubeflow UI
        metrics.log_metric(name, float(score))

        if score > best_score:
            best_score = score
            best_model = grid.best_estimator_
            best_name = name

    print(f"Best model: {best_name} ({best_score:.4f})")

    # 🔥 log best model info
    metrics.log_metric("best_f1_score", float(best_score))

    # save ONLY best model (important)
    with open(model.path, "wb") as f:
        pickle.dump(best_model, f)