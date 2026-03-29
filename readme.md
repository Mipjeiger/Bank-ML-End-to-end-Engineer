## 🏦 Bank financial enterprise engineer project

## 🎯 Goals project
- Credit Risk analysis -> High consumer impact
    - Modeling toward to target data variable
    - Credit scoring for preventing customer churn
    - Audit credit risk score analyst to improve credit database
- Fraud Detection -> High financial fraud prevention
    - Transaction monitoring
    - Real-time detection with Fraud label
    - Data scalability on monitoring insights
- Market Risk -> Systematic implication
    - Volatility forecasting to improve Bank advantage in each sides
    - Optioning on pricing product to get stabilize market
- Operation Risk -> Internal focus to improve company system
    - Predictive maintenance if exists in Production
    - Root causes analysis

## 🤖 Conftusion matrix on metrics machine learning models result -> (Fraud detection, Market Risk, Operational Risk)
- Decision Tree Classifier - Confusion Matrix
    ![alt text](Database/images/confusion_matrix_Decision_Tree.png)
- KNeighbor Classifier - Confusion Matrix
    ![alt text](Database/images/confusion_matrix_KNN.png)
- Random Forest Classifier - Confusion Matrix
    ![alt text](Database/images/confusion_matrix_Random_Forest.png)
- XGBoost Classifier - Confusion Matrix
    ![alt text](Database/images/confusion_matrix_XGBoost.png)

## 🏦 API Production -> Models ingested to embed in Production with API Integration
- Machine learning models Information -> Churn and Fraud (Involving Market analysis, Operational risk, Fraud detection, Credit risk analysis)
    - Machine learning models information
    ![alt text](EEDE4B98-12BD-47EE-B2C8-5DB30E675E46.png)
    - Customer churn prediction with probability
    ![alt text](Database/images/48CAB9E8-A072-49AC-AD63-7C585A75E6CD.png)
    - Fraud detection with models choice by users or best model chosen
    ![alt text](Database/images/70A1F20A-C137-44E6-BC86-A434EFCB66D7.png)
- Gradio -> User Interface (UI) for website implementation on Fintech real-time project condition

- 


## 🧠 Metrics output on Machine Learning models -> Churn prediction
Training model Decision Tree...
Classification report for Decision Tree:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1592
           1       1.00      0.99      0.99       408

    accuracy                           1.00      2000
   macro avg       1.00      0.99      1.00      2000
weighted avg       1.00      1.00      1.00      2000

Precision: 1.0000
Recall: 0.9853
F1 Score: 0.9926
Accuracy: 0.9970
Training model Random Forest...
Classification report for Random Forest:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1592
           1       1.00      1.00      1.00       408

    accuracy                           1.00      2000
   macro avg       1.00      1.00      1.00      2000
weighted avg       1.00      1.00      1.00      2000

Precision: 0.9975
Recall: 0.9951
F1 Score: 0.9963
Accuracy: 0.9985
Training model Logistic Regression...
Classification report for Logistic Regression:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1592
           1       1.00      1.00      1.00       408

    accuracy                           1.00      2000
   macro avg       1.00      1.00      1.00      2000
weighted avg       1.00      1.00      1.00      2000

Precision: 0.9975
Recall: 0.9951
F1 Score: 0.9963
Accuracy: 0.9985
Training model XGBoost...
Classification report for XGBoost:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      1592
           1       1.00      0.99      1.00       408

    accuracy                           1.00      2000
   macro avg       1.00      1.00      1.00      2000
weighted avg       1.00      1.00      1.00      2000

Precision: 0.9975
Recall: 0.9926
F1 Score: 0.9951
Accuracy: 0.9980