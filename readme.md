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
    ![alt text](Database/images/18E46D30-9ED8-4F50-80EE-1877CFCA8B35.png)
    - Customer churn prediction with probability
    ![alt text](Database/images/48CAB9E8-A072-49AC-AD63-7C585A75E6CD.png)
    - Fraud detection with models choice by users or best model chosen
    ![alt text](Database/images/70A1F20A-C137-44E6-BC86-A434EFCB66D7.png)
    - Marketing prediction with reasoning engine -> Reasoning: based on data features inputed the Decision Tree model represents prediction true label about 2 values.
    Probability = 1.0 represent perfect of proba this means in marketing prediction has strong premis to give a reasoning.
    ![alt text](Database/images/B7C4B085-9F0D-4989-B3D1-98FF66059463.png)
    - Operational risk prediction with reasoning engine -> Reasoning: based on data features inputed the Decision Tree model represents prediction true label out 0 values.
    Probability = 1.0 represent about the operational will getting elevated risk.
    ![alt text](Database/images/BA790BFA-B121-4147-B13C-5F2295B7113F.png)
- Gradio -> User Interface (UI) for website implementation on Fintech real-time project condition
    - User Interface (UI) for Marketing prediction
    ![alt text](Database/images/DEA74967-91B8-4D82-B317-B5E0D093F5A7.png)
    - User Interface (UI) for Operational Risk prediction
    ![alt text](Database/images/38B55107-C295-43E1-A6A4-0BCE299B5122.png)


- MLFlow -> Tracking machine learning models & improve metrics result on based models
    - Machine learning models metrics -> visualization on coordinates parallel plot
    ![alt text](068A3389-DA3C-47EE-BD30-07D42E7B361E.png)
    - Machine learning models metrics comparison
    ![alt text](Database/images/51D6281C-6FCF-4D07-83ED-12C473B67223.png)
    - Metrics result comparison with scores
    ![alt text](Database/images/BD581B4A-3357-4392-AFFD-A831725A861B.png)
    ![alt text](Database/images/DF61392C-575F-4325-941D-C1020F66B6A7.png)
    - Registered models
    ![alt text](Database/images/FB4286F8-0CE9-4C06-A4CC-A8C82B35F8D7.png)
    - List models on API
    ![alt text](Database/images/30271C53-789C-4001-B190-F191FEE4296E.png)

- LLM -> Large Language Models for Banking solution for simplify complex problem
    - Banking LLM AI insights
    ![alt text](Database/images/4BF5C5CD-D301-478C-81FB-DBF6446AF146.png)

- Grafana & Prometheus


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