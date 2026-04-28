import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Banking ML Prediction System",
  description: "Advanced Customer Risk Assessment",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
