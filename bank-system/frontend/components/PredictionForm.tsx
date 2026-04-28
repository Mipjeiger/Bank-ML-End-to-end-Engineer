"use client";
import { useState } from "react";
import { apiClient } from "../services/api";

export default function PredictionForm() {
  const [model, setModel] = useState("LogisticRegression");
  const [customerId, setCustomerId] = useState("");
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const result = await apiClient.getPrediction(customerId, model);
      setPrediction(result);
    } catch (error) {
      alert("Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-2xl mx-auto">
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-2xl font-bold mb-6">Customer Churn Prediction</h2>

        <form onSubmit={handlePredict} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Model
            </label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="LogisticRegression">Logistic Regression</option>
              <option value="RandomForest">Random Forest</option>
              <option value="XGBClassifier">XGB Classifier</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Customer ID
            </label>
            <input
              type="text"
              value={customerId}
              onChange={(e) => setCustomerId(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              placeholder="Enter customer ID"
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-2 rounded-lg font-medium hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? "Predicting..." : "Get Prediction"}
          </button>
        </form>

        {prediction && (
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h3 className="font-semibold mb-2">Prediction Result</h3>
            <p className="text-lg">
              Risk Level:{" "}
              <span
                className={
                  prediction.prediction === 1
                    ? "text-red-600 font-bold"
                    : "text-green-600 font-bold"
                }
              >
                {prediction.prediction === 1 ? "HIGH RISK" : "LOW RISK"}
              </span>
            </p>
            <p className="text-sm text-gray-600 mt-2">
              Confidence: {(prediction.confidence * 100).toFixed(2)}%
            </p>
          </div>
        )}
      </div>
    </div>
  );
}