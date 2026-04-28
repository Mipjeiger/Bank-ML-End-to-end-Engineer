import React, { useEffect, useState } from 'react';
import { apiClient } from "../services/api";

export default function Dashboard() {
    const [metrics, setMetrics] = useState<any>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const data = await apiClient.getMetrics();
        setMetrics(data);
      } catch (error) {
        console.error("Error fetching metrics:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchMetrics();
  }, []);

  if (loading) return <div className="p-6">Loading...</div>;

  return (
    <div className="p-6 bg-gray-50">
      <h1 className="text-3xl font-bold mb-6">Banking Dashboard</h1>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <MetricCard
          title="Total Customers"
          value={metrics?.total_customers}
          icon="👥"
        />
        <MetricCard
          title="Churn Risk"
          value={`${metrics?.churn_rate}%`}
          icon="⚠️"
        />
        <MetricCard
          title="Avg Credit Score"
          value={metrics?.avg_credit_score}
          icon="📊"
        />
        <MetricCard
          title="Total Balance"
          value={`$${metrics?.total_balance?.toLocaleString()}`}
          icon="💰"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <RecentPredictions />
        <HighRiskCustomers />
      </div>
    </div>
  );
}

function MetricCard({ title, value, icon }: any) {
  return (
    <div className="bg-white p-6 rounded-lg shadow border-l-4 border-blue-500">
      <div className="text-2xl mb-2">{icon}</div>
      <p className="text-gray-600 text-sm">{title}</p>
      <p className="text-2xl font-bold">{value}</p>
    </div>
  );
}

function RecentPredictions() {
  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">Recent Predictions</h2>
      <div className="space-y-2">
        <PredictionRow
          customer="John Doe"
          model="Logistic Regression"
          risk="High"
        />
        <PredictionRow
          customer="Jane Smith"
          model="Random Forest"
          risk="Low"
        />
        <PredictionRow
          customer="Bob Johnson"
          model="XGB Classifier"
          risk="Medium"
        />
      </div>
    </div>
  );
}

function HighRiskCustomers() {
  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <h2 className="text-xl font-semibold mb-4">High Risk Customers</h2>
      <div className="space-y-3">
        <RiskCustomerRow name="Alice Brown" risk={92} score={450} />
        <RiskCustomerRow name="Charlie Davis" risk={87} score={520} />
        <RiskCustomerRow name="Diana Evans" risk={78} score={580} />
      </div>
    </div>
  );
}

function PredictionRow({ customer, model, risk }: any) {
  const riskColor =
    risk === "High" ? "bg-red-100 text-red-800" : "bg-green-100 text-green-800";
  return (
    <div className="flex justify-between items-center p-3 border-b">
      <span className="font-medium">{customer}</span>
      <span className="text-sm text-gray-600">{model}</span>
      <span className={`px-3 py-1 rounded-full text-sm font-medium ${riskColor}`}>
        {risk}
      </span>
    </div>
  );
}

function RiskCustomerRow({ name, risk, score }: any) {
  return (
    <div className="flex justify-between items-center p-3 border-b bg-red-50">
      <span className="font-medium">{name}</span>
      <span className="text-sm text-gray-600">Score: {score}</span>
      <div className="w-20 bg-gray-200 rounded-full h-2">
        <div
          className="bg-red-600 h-2 rounded-full"
          style={{ width: `${risk}%` }}
        ></div>
      </div>
    </div>
  );
}