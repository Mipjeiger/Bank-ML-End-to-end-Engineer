const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export const apiClient = {
    // Predictions
    async getPrediction(customerId: string, model: string) {
        const res = await fetch(`${API_BASE_URL}/api/predictions/batch`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ data, model }),
        });
        if (!res.ok) throw new Error("Batch prediction failed");
        return res.json();
    },

    // Customers
    async getCustomers(page = 1, limit = 50) {
        const res = await fetch(`${API_BASE_URL}/api/customers?page=${page}&limit=${limit}`);
        if (!res.ok) throw new Error("Failed to fetch customers");
        return res.json();
    },

    async getCustomer(id: string) {
        const res = await fetch(`${API_BASE_URL}/api/customers/${id}`);
        if (!res.ok) throw new Error("Failed to fetch customer");
        return res.json();
    },

    // Metrics
    async getMetrics() {
        const res = await fetch(`${API_BASE_URL}/api/metrics`);
        if (!res.ok) throw new Error("Failed to fetch metrics");
        return res.json();
    },

    // Reports
    async generateReports(startDate: string, endDate: string) {
        const res = await fetch(`${API_BASE_URL}/api/reports/generate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ start_date: startDate, end_date: endDate }),
        });
        if (!res.ok) throw new Error("Failed to generate reports");
        return res.json();
    }
};