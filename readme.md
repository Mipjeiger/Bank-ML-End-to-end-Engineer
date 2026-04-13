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
    - Experiment visualization: Using Parallel Coordinates Plots, we can instantly see the relationship between hyperparameters and performance. The "red" paths represent our champion models, predominantly XGBoost, which achieved the highest stability across all metrics.
    ![alt text](<Database/images/068A3389-DA3C-47EE-BD30-07D42E7B361E copy.png>)
    - Multi-Metric Evaluation: Beyond simple accuracy, we tracked Precision, Recall, and F1-Score. This is critical for our Fraud Detection task, where balancing the detection of fraudulent transactions against user friction is a top priority.
    ![alt text](Database/images/51D6281C-6FCF-4D07-83ED-12C473B67223.png)
    - Optimal Thresholding: MLflow helped identify that a higher classification threshold (approximately. 0.85) significantly improved model reliability for operational tasks.
    ![alt text](Database/images/BD581B4A-3357-4392-AFFD-A831725A861B.png)
    ![alt text](Database/images/DF61392C-575F-4325-941D-C1020F66B6A7.png)
    - Box plot with 5 algorithms comparison -> XGBoost is a stable model in the top cluster (followed by K-Nearest Neighbors (KNN)) based on the performance values (max: 1, q3: 0.999874, median: 0.999497, q1: 0.5930945, min: 0.4576271)
    ![alt text](Database/images/7F92B2F3-736C-498C-8D03-E352E106145E.png)
    - Model Registry & Deployment: Every high-performing run is versioned in the MLflow Model Registry, allowing for seamless transition from training to our FastAPI-based production environment.
    ![alt text](Database/images/FB4286F8-0CE9-4C06-A4CC-A8C82B35F8D7.png)
    ![alt text](Database/images/30271C53-789C-4001-B190-F191FEE4296E.png)

- LLM -> Large Language Models for Banking solution for simplify complex problem
- LLM thoughts -> as LLMs disclaimer isn't defending this choice on the claim LLMs are accurate models from human cognition. In banking sector environment depend on case LLM can provide a useful structural priority to answer the banking problems through key of comparative statistics LLMs model embed.
    - Banking LLM AI insights (Credit: https://www.getdynamiq.ai/post/generative-ai-and-llms-in-banking-examples-use-cases-limitations-and-solutions)
    ![alt text](Database/images/4BF5C5CD-D301-478C-81FB-DBF6446AF146.png)
    - Banking LLM User Interface (UI) for banking solution in Report Insights
    ![alt text](Database/images/FA8FF2CE-1181-4D30-88F9-56FA62DC0BC3.png)
    - Banking LLM UI for question as problem decision making helper
    ![alt text](Database/images/DE6EC6A9-A9BF-4589-9FC6-E272EADAE0D6.png)
    - Banking LLM in answering question to get insights
    ![alt text](Database/images/9ABD266C-6350-4783-88C7-636C568D2F84.png)
    - Banking LLM in answering question with simple characters
    ![alt text](Database/images/103980E7-3631-4749-AAFC-016380C514AE.png)
    - Banking LLM in Summary report explanation
    ![alt text](Database/images/A4D2E73E-B032-46E4-B659-A57FCCE3D2D0.png)
    - Banking LLM in gradio as User Interface (UI) inference to answer the questions
    ![alt text](Database/images/8E018A7C-9306-4E35-8AF2-B7240828BFFA.png)
    - Banking LLM insights for answering about factors of fraud from PDFs sources
    ![alt text](Database/images/30957B22-8DA3-4423-AC91-CA6FB117F97B.png)

- Grafana & Prometheus for models monitoring
- Prometheus
    - Hierarcy models request total based on request calls on API's integrations by instances in host.docker.internal with job ml_models
        - 1. models request total -> Model_name: XGBoost_Operational, Category: Operational
        ![alt text](Database/images/DB42418B-FA1A-41D3-8A12-53263675EE4F.png)
        - 2. models request total -> Model_name: Decision_Tree_Operational, Category: Operational
        ![alt text](Database/images/0A9E1C40-C6C6-42D7-8D53-7C380F4CF030.png)
        - 3. models request total -> Model_name: Decision_Tree_Marketing, Category: Marketing
        ![alt text](Database/images/50D8486C-8207-4FB0-B8F1-A63E8C472769.png)
        - 4. models request total -> Model_name: Random_Forest_Fraud, Category: Fraud
        ![alt text](Database/images/D6C1105A-5FE5-41F8-BD9F-FAC7923435C7.png)
        - 5. models request total -> Model_name: Decision_Tree_Fraud, Category: Fraud
        ![alt text](Database/images/D6C1105A-5FE5-41F8-BD9F-FAC7923435C7.png)
    - Comparison metric models with query request "model_request_total"
        - model_name= "Decision_Tree_Operational"
        ![alt text](Database/images/522B3381-6890-48D3-85C0-5586F7344B86.png)
        - model_name= "Random_Forest_Fraud"
        ![alt text](Database/images/96C214CE-2FDD-4456-B3AC-379AC6E6D921.png)

- Grafana
    - Models metric monitoring for ensure models tracking, health, performance, and behavior of a deployed machine learning model in MLOps (Machine Learning Operations) model-driven environment.
        - Models metrics 1st pic
        ![alt text](Database/images/7FB05999-4448-4087-8587-3F60BC746BAD.png)
        - Models metrics 2nd pic
        ![alt text](Database/images/F392AA28-6602-46D5-8805-5D57219D68A2.png)
        - Models metrics 3rd pic
        ![alt text](Database/images/054EDFC7-CFAD-43E7-B6B8-A01ED6638C1C.png)

- Evidently AI for evaluation data drifting -> - Deploy on EvidentlyAI Cloud
    - Dataset summary of LLMs Evaluation
    ![alt text](Database/images/F644BE69-A0C4-4106-9EA6-4A18925138BC.png) 
    - Answer length and word count of LLMs evaluation
    ![alt text](Database/images/2B87ADD8-E294-40AF-B35B-1A8D00022A7C.png)
    - Contains number & Uncertain sentence to prevent the hallucinates text
    ![alt text](Database/images/08172A16-B2BD-46E9-8DD0-9198E7254B89.png)
    - Contains percentage & fraud mentions on LLMs sentences
    ![alt text](Database/images/95B0D563-60AE-42A9-9007-8D86B4C3ACDF.png)
    - Mentions churn & numeric density based on data-driven (statistically assumption)
    ![alt text](Database/images/872478B3-1515-47CE-8F38-00F558FB119F.png)
    - Sentiment for answer & text length on sentences traffic
    ![alt text](Database/images/074C6CB9-8E61-42B4-A717-650AAE983B37.png)

- PostgreSQL for data analysis integration with bank system
    - SQL code cheatseet for data analyzing
    ![alt text](Database/images/5805ABB6-F3BB-43AE-9E7E-74E95AA13BC3.png)
    - Fraud analysis which are integrating with RiskScore, Balance column name count by labels (0: non fraud, 1: fraud)
    ![alt text](Database/images/ACEE3FCC-688E-4A82-A68B-3CB9CF348AB4.png)
    - Amount of banking products based on NumOfProducts column name -> Counted toward to avg of balance columnname, Sum of exited columnname, avg of HasCrCard columnname 
    ![alt text](Database/images/56CE1C2B-CFD4-40AB-BEE2-37D46372BB6B.png)

- Kubeflow pipelines deployment
    - Kubeflow architecture pipeline design (sources: https://www.kubeflow.org/docs/started/architecture/)
    ![alt text](Database/images/A24DE502-D999-4B8B-BCE6-37AF29888914.png)
    - Minikube cluster info
    ![alt text](Database/images/69AD4D4D-C994-4006-9B57-A4DF15D2EB4F.png)

    - Models deploy on kubeflow

    - Data pipelines merged

    - LLMs finetuning

- Seldon core for real-time monitoring, analysis, and performance tracking of ML systems, models, and deployment environments, with goals to get decision making to ensure teams have the key metrics for maintenance and decision-making
    - Deploy models on dockerhub repository based on seldon core progress
    ![alt text](Database/images/B00514A5-54CB-40E1-8D6D-AD999050C296.png)

    - Create seldon-system core by kubernetes on docker dekstop
    ![alt text](Database/images/FCBB7E04-1D09-43E1-85DD-4701CAC4F814.png)

    - 1.  Kubernetes pods deployment seldon.yaml for LLM, marketing, fraud, operational models
    ![alt text](Database/images/01C56821-6803-43CF-BD5B-187DD3DEAFE5.png)
      2. Debugging on one of the kubernetes pods on seldon-system
    ![alt text](Database/images/9B2E8BA1-A001-4B8D-AE77-94FB63B1F702.png) 

    - Real-time ML models or LLM deployments

    - Data science monitoring

- CI/CD pipelines for automation ML banking system

- Slack (for event in advantage & disadvantage actions)
    - Slack getting advantage notifications

    - Slack getting disadvantage notifications

- Deploy models on cloud (Optional: Try deploy on railway)


- Deploy LLMs gradio on Huggingface

## 🏦 LLM Reasoning Engine by retrieval PDFs document - Banking AI Insights Report
**Dataset:** 10,000 rows, 25 columns
**Churn Rate:** 20.38% | **Fraud Rate:** 3.32%
**LLMs Directory: ml-engineer/notebooks/Banking_llm_insights_report.md**

---

## Insight 1: Customer Churn Drivers & Prediction

Based on the provided dataset and research context, the main drivers of customer churn can be identified as follows:

1. **RiskScore**: The mean RiskScore is 1.23, indicating that most customers have a relatively low risk profile. However, the standard deviation is 0.92, suggesting that there is a significant variation in risk scores among customers. A higher RiskScore might indicate a higher likelihood of churn, as customers with higher risk profiles may be more likely to switch banks due to dissatisfaction or financial difficulties.

2. **Satisfaction Score**: The mean Satisfaction Score is 3.01, indicating that customers are generally satisfied with their banking services. However, the standard deviation is 1.41, suggesting that there is a significant variation in satisfaction levels among customers. A lower Satisfaction Score might indicate a higher likelihood of churn, as dissatisfied customers are more likely to switch banks.

3. **Complain**: The Complain Rate is 20.44%, indicating that a significant proportion of customers have complained about their banking services. This might suggest that customers who complain are more likely to churn, as they may be dissatisfied with the service quality or resolution of their issues.

4. **IsActiveMember**: This feature is not explicitly mentioned in the dataset summary, but it is likely related to the customer's membership status or activity level. Customers who are inactive or have terminated their membership may be more likely to churn.

5. **BalancePerProduct**: The mean BalancePerProduct is 33603.67, indicating that customers have a significant amount of money invested in their banking products. However, the standard deviation is 28665.63, suggesting that there is a significant variation in balance levels among customers. Customers with lower balance levels may be more likely to churn, as they may be more financially vulnerable or have fewer assets tied to the bank.

As for the recommended ML models for churn prediction with imbalanced data, the papers suggest the following:

1. **Random Forest**: This model achieved an 86% F1-score after applying SMOTE-Tomek Links, demonstrating strong predictive capability. Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy and handle imbalanced data.

2. **Support Vector Machine (SVM)**: This model was used in a voting ensemble model comprising K-Nearest Neighbors (KNN), SVM, Decision Trees (DT), Random Forest (RF), and XGBoost, augmented by SMOTE-based class balancing, resulting in an improvement of F1-score.

3.


---

## Insight 2: Fraud Detection with ML & LLMs

Based on the research context and the provided dataset, the following ML and LLM approaches are recommended for fraud detection:

1. **Gradient Boosting Machines (XGBoost and LightGBM)**: These ensemble methods are particularly effective in detecting anomalies and are recommended for models of probability of default, loss given default, and exposure at default (Robisco & Martínez, 2022; Alamsyah et al., 2025).
2. **Graph Neural Networks**: These can be used to analyze transaction networks and uncover money laundering rings (Dichev et al., 2020).
3. **Pattern Recognition**: This is central to ML and can be efficient in fraud detection and anti-money laundering (AML) (Dichev et al., 2020).
4. **Anomaly Detection Algorithms**: These can detect anomalous activities of transactions, which is crucial for real-time fraud flagging pipelines (Dichev et al., 2020).

To improve real-time fraud flagging pipelines in banking using GenAI, the following approaches can be considered:

1. **Real-time Transaction Analysis**: GenAI can be used to analyze transactions in real-time, identifying unusual patterns indicative of suspicious activity.
2. **Predictive Modeling**: GenAI can be used to develop predictive models that can identify high-risk transactions and flag them for further investigation.
3. **Adaptive Learning**: GenAI can be used to develop adaptive models that can learn from new data and update their predictions in real-time, allowing for more accurate and timely fraud detection.
4. **Integration with Existing Systems**: GenAI can be integrated with existing banking systems, such as customer relationship management (CRM) systems, to provide a more comprehensive view of customer behavior and identify potential fraud risks.

In the context of the provided dataset, GenAI can be used to analyze the following metrics:

* **Fraud Rate**: 3.32% (this can be used as a target variable for fraud detection models)
* **RiskScore**: This can be used as a feature for fraud detection models, as it is likely to be correlated with the likelihood of fraud
* **OperationalRiskScore**: This can also be used as a feature for fraud detection models, as it is likely to be correlated with the likelihood of fraud
* **Transaction Volume**: This can be used as a feature for fraud detection models, as high transaction volumes may indicate suspicious activity

By leveraging these metrics and using the recommended ML and LLM approaches, GenAI can improve real-time fraud flagging pipelines in

---

## Insight 3: GenAI Use Cases for Banking

Based on the provided research context and dataset summary, I would recommend the following GenAI and LLM use cases that are most applicable and deliver the highest ROI for banks:

1. **Fraud Detection and Prevention**: With a Fraud Rate of 3.32%, banks can leverage GenAI and LLMs to improve the accuracy and efficiency of fraud detection. The use of natural language processing (NLP) and machine learning algorithms can help identify patterns and anomalies in customer behavior, reducing the risk of fraudulent activities. According to KPMG, 76% of banking executives in the US plan to use generative AI for fraud detection and prevention, indicating a high potential ROI.

2. **Customer Service**: With a Low Satisfaction Rate of 39.46%, banks can use GenAI and LLMs to improve customer service and experience. By analyzing customer feedback and sentiment, banks can identify areas for improvement and develop personalized responses to customer inquiries. This can lead to increased customer satisfaction and loyalty, resulting in higher revenue and reduced churn rates.

3. **Regulatory Compliance and Risk Avoidance**: With a High Value Customer Rate of 47.99%, banks can use GenAI and LLMs to ensure regulatory compliance and mitigate risks associated with high-value customers. By analyzing customer data and behavior, banks can identify potential risks and develop strategies to mitigate them, reducing the risk of non-compliance and reputational damage.

4. **Personalized Marketing and Product Recommendations**: With a Marketing Score of 1.04 and a Point Earned of 606.52, banks can use GenAI and LLMs to develop personalized marketing campaigns and product recommendations. By analyzing customer behavior and preferences, banks can identify opportunities to cross-sell and upsell products, increasing revenue and customer satisfaction.

5. **Sentiment Analysis and Customer Feedback Analysis**: With a Low Satisfaction Rate of 39.46%, banks can use GenAI and LLMs to analyze customer feedback and sentiment, identifying areas for improvement and developing strategies to increase customer satisfaction.

In terms of specific features, the following are most relevant to these use cases:

* **Marketing Score**: This feature is relevant to personalized marketing and product recommendations, as well as customer service and sentiment analysis.
* **High Value Customer**: This feature is relevant to regulatory compliance and risk avoidance, as well as customer service and sentiment analysis.
* **Low Satisfaction**: This feature is relevant to customer service and sentiment analysis, as well as personalized marketing and product recommendations.
* **Card Type**: This feature is relevant to customer


---

## Insight 4: Risk Score Validation & Explainability

Based on the research context and the provided dataset, we can infer the following recommendations for validating the RiskScore, LowCreditRisk, AgeRisk, and OperationalRiskScore in regulated banking environments.

**Validation of RiskScore:**

The ML model risk management papers recommend the following validation methods for RiskScore:

1. **Drift monitoring**: Regularly monitor the RiskScore for changes in its distribution over time to ensure it remains accurate and reliable (Bank of England, 2023).
2. **Bias detection**: Use techniques such as bias noise detection formulations to identify and mitigate biases in the RiskScore (Misheva et al., 2021).
3. **Fairness testing**: Evaluate the RiskScore for fairness and non-discrimination, ensuring it does not unfairly penalize or benefit certain groups (de Lange et al., 2022).
4. **Model interpretability**: Use explainability techniques such as SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to understand how the RiskScore is generated and identify potential biases (Lundberg & Lee, 2017).

**Validation of LowCreditRisk:**

For LowCreditRisk, the papers recommend:

1. **Model interpretability**: Use SHAP or LIME to understand how the LowCreditRisk model is generated and identify potential biases (Lundberg & Lee, 2017).
2. **Bias detection**: Use bias noise detection formulations to identify and mitigate biases in the LowCreditRisk model (Misheva et al., 2021).
3. **Fairness testing**: Evaluate the LowCreditRisk model for fairness and non-discrimination, ensuring it does not unfairly penalize or benefit certain groups (de Lange et al., 2022).

**Validation of AgeRisk:**

For AgeRisk, the papers recommend:

1. **Model interpretability**: Use SHAP or LIME to understand how the AgeRisk model is generated and identify potential biases (Lundberg & Lee, 2017).
2. **Bias detection**: Use bias noise detection formulations to identify and mitigate biases in the AgeRisk model (Misheva et al., 2021).
3. **Fairness testing**: Evaluate the AgeRisk model for fairness and non-discrimination, ensuring it does not unfairly penalize or benefit certain groups (de Lange et al., 2022).

**Validation of OperationalRiskScore:**

For OperationalRiskScore, the papers recommend:

1. **Model interpretability**: Use SHAP


---

## Insight 5: Customer Segmentation Strategy

Based on the provided research context and dataset summary, I will outline customer segmentation strategies for AI-powered retention and personalized banking services.

1. **Segmentation by Customer Lifetime Value (CLV)**: The papers emphasize the significance of CLV as a foundational construct for designing effective churn reduction strategies [10]. To calculate CLV, we can use features like EstimatedSalary, Balance, and Point Earned. We can create a CLV score by multiplying the average transaction value by the customer's lifetime, which can be estimated using the customer's age and average transaction frequency.

   **Recommendation**: Use a combination of EstimatedSalary, Balance, and Point Earned to create a CLV score, and segment customers into high-value, medium-value, and low-value categories.

2. **Segmentation by Churn Risk**: The papers highlight the importance of identifying customers at risk of churning [6, 7, 8, 9]. We can use features like MarketingScore, Card Type, and Point Earned to identify customers who are likely to churn.

   **Recommendation**: Use a combination of MarketingScore, Card Type, and Point Earned to create a churn risk score, and segment customers into high-risk, medium-risk, and low-risk categories.

3. **Segmentation by Demographic Characteristics**: The papers mention the importance of demographic characteristics like age, gender, and education level in predicting customer behavior [15, 16]. We can use features like Age, Gender, and Education_Level to segment customers into different demographic groups.

   **Recommendation**: Use a combination of Age, Gender, and Education_Level to create demographic segments, and tailor marketing campaigns and services to each segment's preferences and needs.

4. **Segmentation by Behavioral Characteristics**: The papers highlight the importance of behavioral characteristics like transaction frequency, balance, and point earned in predicting customer behavior [11, 12]. We can use features like BalancePerProduct, Point Earned, and OperationalRiskScore to segment customers into different behavioral groups.

   **Recommendation**: Use a combination of BalancePerProduct, Point Earned, and OperationalRiskScore to create behavioral segments, and offer personalized services and rewards to each segment based on their behavior.

5. **Segmentation by Geographic Location**: The papers mention the importance of geographic location in predicting customer behavior [13, 14]. We can use features like Geography (France, Germany, Spain) to segment customers into different geographic groups.

   **Recommendation**: Use a combination of Geography to create geographic segments, and tailor marketing campaigns


---

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