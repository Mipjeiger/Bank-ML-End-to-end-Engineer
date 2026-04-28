import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#3b82f6",
        "primary-dark": "#1e3a8a",
        "primary-light": "#dbeafe",
      },
    },
  },
  plugins: [],
};

export default config;
