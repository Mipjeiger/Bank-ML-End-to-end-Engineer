"use client";
import { useEffect, useState } from "react";
import { apiClient } from "../services/api";

export default function CustomersList() {
  const [customers, setCustomers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);

  useEffect(() => {
    const fetchCustomers = async () => {
      try {
        const data = await apiClient.getCustomers(page);
        setCustomers(data.customers || []);
      } catch (error) {
        console.error("Error fetching customers:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchCustomers();
  }, [page]);

  if (loading) return <div className="p-6">Loading customers...</div>;

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Customer List</h2>

      <div className="bg-white rounded-lg shadow overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-100 border-b">
            <tr>
              <th className="px-6 py-3 text-left text-sm font-semibold">ID</th>
              <th className="px-6 py-3 text-left text-sm font-semibold">Name</th>
              <th className="px-6 py-3 text-left text-sm font-semibold">Email</th>
              <th className="px-6 py-3 text-left text-sm font-semibold">
                Credit Score
              </th>
              <th className="px-6 py-3 text-left text-sm font-semibold">
                Status
              </th>
            </tr>
          </thead>
          <tbody>
            {customers.map((customer) => (
              <tr key={customer.id} className="border-b hover:bg-gray-50">
                <td className="px-6 py-4 text-sm">{customer.id}</td>
                <td className="px-6 py-4 text-sm font-medium">
                  {customer.name}
                </td>
                <td className="px-6 py-4 text-sm">{customer.email}</td>
                <td className="px-6 py-4 text-sm">{customer.credit_score}</td>
                <td className="px-6 py-4 text-sm">
                  <span
                    className={`px-3 py-1 rounded-full text-xs font-medium ${
                      customer.is_active
                        ? "bg-green-100 text-green-800"
                        : "bg-red-100 text-red-800"
                    }`}
                  >
                    {customer.is_active ? "Active" : "Inactive"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 flex justify-center space-x-2">
        <button
          onClick={() => setPage(Math.max(1, page - 1))}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Previous
        </button>
        <span className="px-4 py-2 text-gray-700">Page {page}</span>
        <button
          onClick={() => setPage(page + 1)}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Next
        </button>
      </div>
    </div>
  );
}