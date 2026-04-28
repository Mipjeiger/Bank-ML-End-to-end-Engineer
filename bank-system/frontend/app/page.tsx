"use client";
import { useState } from "react";
import ModelSelector, { ModelValue, DEFAULT_MODEL } from "../components/ModelSelector";

type RawCustomer = {
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
  Exited: boolean;
  Complain: boolean;
  SatisfactionScore: number;
  CardType: string;
  PointEarned: number;
  RiskScore: number;
  HighValueCustomer: boolean;
};

type CustomerData = RawCustomer & {
  BalancePerProduct: number;
  AgeRisk: boolean;
  HighValueCustomer: boolean;
  LowCreditRisk: boolean;
  ComplainFlag: boolean;
  LowSatisfaction: boolean;
};

function deriveFeatures(c: RawCustomer): CustomerData {
  return {
    ...c,
    BalancePerProduct: c.NumOfProducts > 0 ? c.Balance / c.NumOfProducts : 0,
    AgeRisk: c.Age > 50,
    HighValueCustomer: c.Balance > 100000 && c.EstimatedSalary > 80000,
    LowCreditRisk: c.CreditScore > 700,
    ComplainFlag: c.Complain,
    LowSatisfaction: c.SatisfactionScore < 5,
  };
}

const DEFAULT_RAW: RawCustomer = {
  customer_id: "",
  CreditScore: 0,
  Geography: "France",
  Gender: "Male",
  Age: 0,
  Tenure: 0,
  Balance: 0,
  NumOfProducts: 1,
  HasCrCard: false,
  IsActiveMember: false,
  EstimatedSalary: 0,
  Exited: false,
  Complain: false,
  SatisfactionScore: 1,
  CardType: "Silver",
  PointEarned: 0,
  RiskScore: 0,
  HighValueCustomer: false
};

interface PredictionResult {
  prediction: number;
  probability?: number;
  [key: string]: unknown;
}

export default function Page() {
  const [model, setModel] = useState<ModelValue>(DEFAULT_MODEL);
  const [raw, setRaw] = useState<RawCustomer>(DEFAULT_RAW);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function setField<K extends keyof RawCustomer>(key: K, value: RawCustomer[K]) {
    setRaw((prev) => ({ ...prev, [key]: value }));
  }

  async function predict() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = deriveFeatures(raw);
      const res = await fetch(`http://localhost:8000/predict/${model}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      setResult(await res.json());
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  function resetCustomer() {
    setRaw(DEFAULT_RAW);
    setResult(null);
    setError(null);
  }

  const derived = deriveFeatures(raw);

  return (
    <div className="min-h-screen" style={{ backgroundColor: "#f0f4f9" }}>
      {/* Header */}
      <header style={{ backgroundColor: "#1e3a8a", boxShadow: "0 2px 8px rgba(30, 58, 138, 0.15)" }}>
        <div className="max-w-6xl mx-auto px-6 py-6">
          <h1 className="text-4xl font-bold text-white">
            🏦 Banking ML Prediction System
          </h1>
          <p className="text-blue-100 text-sm mt-1">Advanced Customer Risk Assessment</p>
        </div>
      </header>

      <main className="max-w-6xl mx-auto py-8 px-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Form Section */}
          <div className="lg:col-span-2 space-y-6">
            {/* Model Selector */}
            <div style={{ backgroundColor: "white", borderRadius: "12px", padding: "24px", boxShadow: "0 1px 3px rgba(30, 58, 138, 0.1)", borderLeft: "4px solid #3b82f6" }}>
              <h2 className="text-xl font-bold mb-4" style={{ color: "#1e3a8a" }}>📋 Model Selection</h2>
              <ModelSelector value={model} onChange={setModel} />
            </div>

            {/* Customer Information */}
            <div style={{ backgroundColor: "white", borderRadius: "12px", padding: "24px", boxShadow: "0 1px 3px rgba(30, 58, 138, 0.1)", borderLeft: "4px solid #3b82f6" }}>
              <h2 className="text-xl font-bold mb-4" style={{ color: "#1e3a8a" }}>👤 Customer Information</h2>
              <div className="grid grid-cols-2 gap-4">
                <InputField
                  label="Customer ID"
                  type="text"
                  value={raw.customer_id}
                  onChange={(e) => setField("customer_id", e.target.value)}
                />
                <InputField
                  label="Credit Score"
                  type="number"
                  value={raw.CreditScore}
                  onChange={(e) => setField("CreditScore", Number(e.target.value))}
                />
                <SelectField
                  label="Geography"
                  value={raw.Geography}
                  options={["France", "Germany", "Spain"]}
                  onChange={(e) => setField("Geography", e.target.value)}
                />
                <SelectField
                  label="Gender"
                  value={raw.Gender}
                  options={["Male", "Female"]}
                  onChange={(e) => setField("Gender", e.target.value)}
                />
                <InputField
                  label="Age"
                  type="number"
                  value={raw.Age}
                  onChange={(e) => setField("Age", Number(e.target.value))}
                />
                <InputField
                  label="Tenure (years)"
                  type="number"
                  value={raw.Tenure}
                  onChange={(e) => setField("Tenure", Number(e.target.value))}
                />
                <InputField
                  label="Balance"
                  type="number"
                  value={raw.Balance}
                  onChange={(e) => setField("Balance", Number(e.target.value))}
                />
                <InputField
                  label="Num Of Products"
                  type="number"
                  value={raw.NumOfProducts}
                  min={1}
                  max={4}
                  onChange={(e) => setField("NumOfProducts", Number(e.target.value))}
                />
                <InputField
                  label="Estimated Salary"
                  type="number"
                  value={raw.EstimatedSalary}
                  onChange={(e) => setField("EstimatedSalary", Number(e.target.value))}
                />
                <InputField
                  label="Risk Score"
                  type="number"
                  value={raw.RiskScore}
                  onChange={(e) => setField("RiskScore", Number(e.target.value))}
                />
              </div>
            </div>

            {/* Card Information */}
            <div style={{ backgroundColor: "white", borderRadius: "12px", padding: "24px", boxShadow: "0 1px 3px rgba(30, 58, 138, 0.1)", borderLeft: "4px solid #3b82f6" }}>
              <h2 className="text-xl font-bold mb-4" style={{ color: "#1e3a8a" }}>💳 Card Information</h2>
              <div className="grid grid-cols-3 gap-4">
                <SelectField
                  label="Card Type"
                  value={raw.CardType}
                  options={["Silver", "Gold", "Platinum", "Diamond"]}
                  onChange={(e) => setField("CardType", e.target.value)}
                />
                <InputField
                  label="Points Earned"
                  type="number"
                  value={raw.PointEarned}
                  onChange={(e) => setField("PointEarned", Number(e.target.value))}
                />
                <InputField
                  label="Satisfaction Score"
                  type="number"
                  value={raw.SatisfactionScore}
                  min={1}
                  max={10}
                  onChange={(e) => setField("SatisfactionScore", Number(e.target.value))}
                />
              </div>
            </div>

            {/* Flags */}
            <div style={{ backgroundColor: "white", borderRadius: "12px", padding: "24px", boxShadow: "0 1px 3px rgba(30, 58, 138, 0.1)", borderLeft: "4px solid #3b82f6" }}>
              <h2 className="text-xl font-bold mb-4" style={{ color: "#1e3a8a" }}>🚩 Account Flags</h2>
              <div className="grid grid-cols-2 gap-4">
                <CheckboxField
                  label="Has Credit Card"
                  checked={raw.HasCrCard}
                  onChange={(e) => setField("HasCrCard", e.target.checked)}
                />
                <CheckboxField
                  label="Is Active Member"
                  checked={raw.IsActiveMember}
                  onChange={(e) => setField("IsActiveMember", e.target.checked)}
                />
                <CheckboxField
                  label="Exited"
                  checked={raw.Exited}
                  onChange={(e) => setField("Exited", e.target.checked)}
                />
                <CheckboxField
                  label="Has Complaint"
                  checked={raw.Complain}
                  onChange={(e) => setField("Complain", e.target.checked)}
                />
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4">
              <button
                onClick={predict}
                disabled={loading}
                style={{
                  flex: 1,
                  backgroundColor: "#3b82f6",
                  color: "white",
                  padding: "12px 24px",
                  borderRadius: "8px",
                  fontWeight: "600",
                  border: "none",
                  cursor: "pointer",
                  fontSize: "16px",
                  transition: "all 0.3s ease",
                  opacity: loading ? 0.6 : 1,
                }}
                onMouseEnter={(e) => !loading && (e.currentTarget.style.backgroundColor = "#2563eb")}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#3b82f6")}
              >
                {loading ? "🔄 Predicting..." : "✨ Get Prediction"}
              </button>
              <button
                onClick={resetCustomer}
                disabled={loading}
                style={{
                  flex: 1,
                  backgroundColor: "#9ca3af",
                  color: "white",
                  padding: "12px 24px",
                  borderRadius: "8px",
                  fontWeight: "600",
                  border: "none",
                  cursor: "pointer",
                  fontSize: "16px",
                  transition: "all 0.3s ease",
                  opacity: loading ? 0.6 : 1,
                }}
                onMouseEnter={(e) => !loading && (e.currentTarget.style.backgroundColor = "#6b7280")}
                onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "#9ca3af")}
              >
                Reset
              </button>
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-1">
            {error && (
              <div style={{ backgroundColor: "#fee2e2", borderLeft: "4px solid #ef4444", borderRadius: "8px", padding: "16px" }}>
                <h3 style={{ color: "#991b1b", fontWeight: "bold" }}>❌ Error</h3>
                <p style={{ color: "#dc2626", fontSize: "14px", marginTop: "4px" }}>{error}</p>
              </div>
            )}

            {result && (
              <div style={{ backgroundColor: "#ecfdf5", borderLeft: "4px solid #10b981", borderRadius: "8px", padding: "24px" }}>
                <h3 style={{ color: "#065f46", fontWeight: "bold", fontSize: "18px", marginBottom: "16px" }}>
                  ✅ Prediction Result
                </h3>
                <div className="space-y-3">
                  <div>
                    <p style={{ color: "#059669", fontSize: "14px" }}>Prediction</p>
                    <p style={{ fontSize: "28px", fontWeight: "bold", color: "#065f46", marginTop: "4px" }}>
                      {result.prediction === 1 ? "🔴 HIGH RISK" : "🟢 LOW RISK"}
                    </p>
                  </div>
                  {result.probability && (
                    <div>
                      <p style={{ color: "#059669", fontSize: "14px" }}>Confidence</p>
                      <div style={{ width: "100%", backgroundColor: "#d1fae5", borderRadius: "8px", height: "12px", marginTop: "8px" }}>
                        <div
                          style={{
                            backgroundColor: "#10b981",
                            height: "12px",
                            borderRadius: "8px",
                            width: `${(result.probability as number) * 100}%`,
                            transition: "width 0.3s ease",
                          }}
                        ></div>
                      </div>
                      <p style={{ color: "#065f46", fontWeight: "600", fontSize: "14px", marginTop: "8px" }}>
                        {((result.probability as number) * 100).toFixed(2)}%
                      </p>
                    </div>
                  )}
                  <details style={{ marginTop: "16px" }}>
                    <summary style={{ cursor: "pointer", color: "#059669", fontWeight: "600" }}>
                      View Details
                    </summary>
                    <pre style={{ backgroundColor: "white", padding: "8px", borderRadius: "4px", marginTop: "8px", fontSize: "12px", overflow: "auto", maxHeight: "192px" }}>
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </details>
                </div>
              </div>
            )}

            {!result && !error && (
              <div style={{ backgroundColor: "#dbeafe", borderLeft: "4px solid #3b82f6", borderRadius: "8px", padding: "24px" }}>
                <p style={{ color: "#1e40af", lineHeight: "1.6" }}>
                  📝 Fill in the customer information and click "Get Prediction" to see results.
                </p>
              </div>
            )}

            {/* Debug Section */}
            <details style={{ marginTop: "24px" }}>
              <summary style={{ cursor: "pointer", color: "#1e3a8a", fontWeight: "600" }}>
                📊 Debug: Current Payload
              </summary>
              <pre style={{ backgroundColor: "#1e293b", color: "#e2e8f0", padding: "16px", borderRadius: "8px", marginTop: "8px", fontSize: "12px", overflow: "auto", maxHeight: "256px" }}>
                {JSON.stringify({ model, ...derived }, null, 2)}
              </pre>
            </details>
          </div>
        </div>
      </main>
    </div>
  );
}

// Helper Components
interface InputFieldProps {
  label: string;
  type?: string;
  value: any;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  min?: number;
  max?: number;
}

function InputField({ label, type = "text", ...props }: InputFieldProps) {
  return (
    <div>
      <label style={{ display: "block", fontSize: "14px", fontWeight: "500", color: "#1e3a8a", marginBottom: "6px" }}>
        {label}
      </label>
      <input
        type={type}
        style={{
          width: "100%",
          padding: "10px 12px",
          border: "1px solid #cbd5e1",
          borderRadius: "6px",
          fontSize: "14px",
          transition: "all 0.3s ease",
          backgroundColor: "#f9fafb",
        }}
        onFocus={(e) => {
          e.currentTarget.style.borderColor = "#3b82f6";
          e.currentTarget.style.boxShadow = "0 0 0 3px rgba(59, 130, 246, 0.1)";
        }}
        onBlur={(e) => {
          e.currentTarget.style.borderColor = "#cbd5e1";
          e.currentTarget.style.boxShadow = "none";
        }}
        {...props}
      />
    </div>
  );
}

interface SelectFieldProps {
  label: string;
  value: string;
  options: string[];
  onChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
}

function SelectField({ label, value, options, onChange }: SelectFieldProps) {
  return (
    <div>
      <label style={{ display: "block", fontSize: "14px", fontWeight: "500", color: "#1e3a8a", marginBottom: "6px" }}>
        {label}
      </label>
      <select
        value={value}
        onChange={onChange}
        style={{
          width: "100%",
          padding: "10px 12px",
          border: "1px solid #cbd5e1",
          borderRadius: "6px",
          fontSize: "14px",
          backgroundColor: "#f9fafb",
          cursor: "pointer",
          transition: "all 0.3s ease",
        }}
        onFocus={(e) => {
          e.currentTarget.style.borderColor = "#3b82f6";
          e.currentTarget.style.boxShadow = "0 0 0 3px rgba(59, 130, 246, 0.1)";
        }}
        onBlur={(e) => {
          e.currentTarget.style.borderColor = "#cbd5e1";
          e.currentTarget.style.boxShadow = "none";
        }}
      >
        {options.map((opt: string) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}

interface CheckboxFieldProps {
  label: string;
  checked: boolean;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}

function CheckboxField({ label, checked, onChange }: CheckboxFieldProps) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: "8px", cursor: "pointer" }}>
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        style={{
          width: "18px",
          height: "18px",
          cursor: "pointer",
          accentColor: "#3b82f6",
        }}
      />
      <span style={{ fontSize: "14px", color: "#1e3a8a", fontWeight: "500" }}>{label}</span>
    </label>
  );
}