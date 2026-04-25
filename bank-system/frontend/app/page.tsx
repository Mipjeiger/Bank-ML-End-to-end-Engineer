// app/page.tsx
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
    BalancePerProduct:  c.NumOfProducts > 0 ? c.Balance / c.NumOfProducts : 0,
    AgeRisk:            c.Age > 50,
    HighValueCustomer:  c.Balance > 100000 && c.EstimatedSalary > 80000,
    LowCreditRisk:      c.CreditScore > 700,
    ComplainFlag:       c.Complain,
    LowSatisfaction:    c.SatisfactionScore < 5,
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
};

export default function Page() {
  const [model, setModel] = useState<ModelValue>(DEFAULT_MODEL);
  const [raw, setRaw] = useState<RawCustomer>(DEFAULT_RAW);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
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

  // preview derived payload for debug panel
  const derived = deriveFeatures(raw);

  return (
    <div style={{ maxWidth: 700, margin: "0 auto", padding: 24, fontFamily: "monospace" }}>
      <h1>🏦 Banking ML Prediction</h1>

      {/* ── Model Selector ── */}
      <section>
        <h2>Model</h2>
        <ModelSelector value={model} onChange={setModel} />
      </section>

      {/* ── Basic Info ── */}
      <section>
        <h2>📊 Information Dataset</h2>

        <label>Customer ID
          <input type="text" value={raw.customer_id}
            onChange={(e) => setField("customer_id", e.target.value)} />
        </label>

        <label>Credit Score
          <input type="number" value={raw.CreditScore}
            onChange={(e) => setField("CreditScore", Number(e.target.value))} />
        </label>

        <label>Geography
          <select value={raw.Geography}
            onChange={(e) => setField("Geography", e.target.value)}>
            <option value="France">France</option>
            <option value="Germany">Germany</option>
            <option value="Spain">Spain</option>
          </select>
        </label>

        <label>Gender
          <select value={raw.Gender}
            onChange={(e) => setField("Gender", e.target.value)}>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
          </select>
        </label>

        <label>Age
          <input type="number" value={raw.Age}
            onChange={(e) => setField("Age", Number(e.target.value))} />
        </label>

        <label>Tenure
          <input type="number" value={raw.Tenure}
            onChange={(e) => setField("Tenure", Number(e.target.value))} />
        </label>

        <label>Balance
          <input type="number" value={raw.Balance}
            onChange={(e) => setField("Balance", Number(e.target.value))} />
        </label>

        <label>Num Of Products
          <input type="number" value={raw.NumOfProducts} min={1} max={4}
            onChange={(e) => setField("NumOfProducts", Number(e.target.value))} />
        </label>

        <label>Estimated Salary
          <input type="number" value={raw.EstimatedSalary}
            onChange={(e) => setField("EstimatedSalary", Number(e.target.value))} />
        </label>

        <label>Risk Score
          <input type="number" value={raw.RiskScore}
            onChange={(e) => setField("RiskScore", Number(e.target.value))} />
        </label>
      </section>

      {/* ── Card Info ── */}
      <section>
        <h2>Card Info</h2>

        <label>Card Type
          <select value={raw.CardType}
            onChange={(e) => setField("CardType", e.target.value)}>
            <option value="Silver">Silver</option>
            <option value="Gold">Gold</option>
            <option value="Platinum">Platinum</option>
            <option value="Diamond">Diamond</option>
          </select>
        </label>

        <label>Points Earned
          <input type="number" value={raw.PointEarned}
            onChange={(e) => setField("PointEarned", Number(e.target.value))} />
        </label>

        <label>Satisfaction Score
          <input type="number" value={raw.SatisfactionScore} min={1} max={10}
            onChange={(e) => setField("SatisfactionScore", Number(e.target.value))} />
        </label>
      </section>

      {/* ── Boolean Flags ── */}
      <section>
        <h2>Flags</h2>

        <label>
          <input type="checkbox" checked={raw.HasCrCard}
            onChange={(e) => setField("HasCrCard", e.target.checked)} />
          Has Credit Card
        </label>

        <label>
          <input type="checkbox" checked={raw.IsActiveMember}
            onChange={(e) => setField("IsActiveMember", e.target.checked)} />
          Is Active Member
        </label>

        <label>
          <input type="checkbox" checked={raw.Exited}
            onChange={(e) => setField("Exited", e.target.checked)} />
          Exited
        </label>

        <label>
          <input type="checkbox" checked={raw.Complain}
            onChange={(e) => setField("Complain", e.target.checked)} />
          Complain
        </label>
      </section>

      {/* ── Actions ── */}
      <div style={{ display: "flex", gap: 12, marginTop: 24 }}>
        <button onClick={predict} disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
        <button onClick={resetCustomer} disabled={loading}>
          Reset
        </button>
      </div>

      {/* ── Error ── */}
      {error && (
        <div style={{ color: "red", marginTop: 16 }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* ── Result ── */}
      {result && (
        <div style={{ marginTop: 16 }}>
          <h2>Result</h2>
          <pre style={{ background: "#f4f4f4", padding: 16, borderRadius: 6 }}>
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}

      {/* ── Debug: show current payload ── */}
      <details style={{ marginTop: 24 }}>
        <summary style={{ cursor: "pointer" }}>Current Payload (debug)</summary>
        <pre style={{ background: "#f4f4f4", padding: 16, borderRadius: 6, fontSize: 11 }}>
          {JSON.stringify({ model, ...derived }, null, 2)}
        </pre>
      </details>
    </div>
  );
}