"use client";

type Model = {
    value: string;
    label: string;
};

type ModelGroup = {
    group: string;
    models: Model[];
};

const MODEL_GROUPS: ModelGroup[] = [
    {
        group: "Fraud",
        models: [
            { value: "LogisticRegression_fraud",     label: "Logistic Regression" },
            { value: "DecisionTreeClassifier_fraud", label: "Decision Tree" },
            { value: "RandomForestClassifier_fraud", label: "Random Forest" },
            { value: "KNeighborsClassifier_fraud",   label: "KNN" },
            { value: "XGBClassifier_fraud",          label: "XGBoost" },
        ],
    },
    {
        group: "Marketing",
        models: [
            { value: "LogisticRegression_marketing",     label: "Logistic Regression" },
            { value: "DecisionTreeClassifier_marketing", label: "Decision Tree" },
            { value: "RandomForestClassifier_marketing", label: "Random Forest" },
            { value: "KNeighborsClassifier_marketing",   label: "KNN" },
            { value: "XGBClassifier_marketing",          label: "XGBoost" },
        ],
    },
    {
        group: "Operational",
        models: [
        { value: "LogisticRegression_operational",     label: "Logistic Regression" },
        { value: "DecisionTreeClassifier_operational", label: "Decision Tree" },
        { value: "RandomForestClassifier_operational", label: "Random Forest" },
        { value: "KNeighborsClassifier_operational",   label: "KNN" },
        { value: "XGBClassifier_operational",          label: "XGBoost" },
        ],
    },
];

// Export model values as a union type for type safety
export type ModelValue =
    | "LogisticRegression_fraud"
    | "DecisionTreeClassifier_fraud"
    | "RandomForestClassifier_fraud"
    | "KNeighborsClassifier_fraud"
    | "XGBClassifier_fraud"
    | "LogisticRegression_marketing"
    | "DecisionTreeClassifier_marketing"
    | "RandomForestClassifier_marketing"
    | "KNeighborsClassifier_marketing"
    | "XGBClassifier_marketing"
    | "LogisticRegression_operational"
    | "DecisionTreeClassifier_operational"
    | "RandomForestClassifier_operational"
    | "KNeighborsClassifier_operational"
    | "XGBClassifier_operational";

export const DEFAULT_MODEL: ModelValue = "LogisticRegression_fraud";

type Props = {
    value: ModelValue;
    onChange: (value: ModelValue) => void;
};

export default function ModelSelector({ value, onChange }: Props) {
    return (
    <div>
      <label htmlFor="model-select">Select Model</label>
      <select
        id="model-select"
        value={value}
        onChange={(e) => onChange(e.target.value as ModelValue)}
        style={{ width: "100%", padding: 8 }}
      >
        {MODEL_GROUPS.map((group) => (
          <optgroup key={group.group} label={group.group}>
            {group.models.map((model) => (
              <option key={model.value} value={model.value}>
                {model.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </div>
  );
}