"use Client";

import { useState } from "react";

type CustomerData = {
    customer_id: string;
    CreditScore: number;
    Geography: string;
    Gender: string;
    Age: number;
    Tenure: number;
    Balance: number;
    NumOfProducts: number;
    HasCrCard: boolean;
    IsActiveMember: boolean;
    EstimatedSalary: number;
    Complain: boolean;
    SatisfactionScore: number;
    CardType: string;
    PointEarned: number;
    RiskScore: number;
    BalancePerProduct: number;
    AgeRisk: boolean;
    HighValueCustomer: boolean;
    LowCreditRisk: boolean;
    ComplainFlag: boolean;
    LowSatisfaction: boolean;
};

const MODELS = [
    { value: "LogisticRegression_fraud", label: "Logistic Regression - Fraud" },
    { value: "DecisionTreeClassifier_fraud", label: "Decision Tree — Fraud" },
    { value: "RandomForestClassifier_fraud", label: "Random Forest — Fraud" },
    { value: "KNeighborsClassifier_fraud",   label: "KNN — Fraud" },
    { value: "XGBClassifier_fraud",          label: "XGBoost — Fraud" },

    { value: "LogisticRegression_marketing",     label: "Logistic Regression — Marketing" },
    { value: "DecisionTreeClassifier_marketing", label: "Decision Tree — Marketing" },
    { value: "RandomForestClassifier_marketing", label: "Random Forest — Marketing" },
    { value: "KNeighborsClassifier_marketing",   label: "KNN — Marketing" },
    { value: "XGBClassifier_marketing",          label: "XGBoost — Marketing" },

    { value: "LogisticRegression_operational",     label: "Logistic Regression — Operational" },
    { value: "DecisionTreeClassifier_operational", label: "Decision Tree — Operational" },
    { value: "RandomForestClassifier_operational", label: "Random Forest — Operational" },
    { value: "KNeighborsClassifier_operational",   label: "KNN — Operational" },
    { value: "XGBClassifier_operational",          label: "XGBoost — Operational" },
] as const;

type ModelValue = typeof MODELS[number]["value"];

const DEFAULT_CUSTOMER: CustomerData = {
    customer_id: "ADX_10",
    CreditScore: 528,
    Geography: "France",
    Gender: "Male",
    Age: 31,
    Tenure: 6,
    Balance: 10201672.0		,
    NumOfProducts: 2,
    HasCrCard: false,
    IsActiveMember: false,
    EstimatedSalary: 8018112.0,
    Complain: false,
    SatisfactionScore: 3,
    CardType: "GOLD",
    PointEarned: 264,
    RiskScore: 1,
    BalancePerProduct: 0.000000,
    AgeRisk: false,
    HighValueCustomer: true,
    LowCreditRisk: false,
    ComplainFlag: false,
    LowSatisfaction: false
};

export default function Page() {
    const [model, setModel] = useState<ModelValue>("LogisticRegression_fraud");
    const [customer, setCustomer] = useState<CustomerData>(DEFAULT_CUSTOMER);
    const [result, setResults] = useState<unknown>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
}