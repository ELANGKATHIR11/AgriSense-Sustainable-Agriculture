/**
 * AGRISENSE Application Environment Configuration
 */

// We can read configuration from import.meta.env
// Extract env elements safely without strict DOM compiler warnings
// @ts-ignore
const metaEnv = typeof import.meta !== "undefined" ? (import.meta.env || {}) : {};

export const VITE_API_URL = 
  (metaEnv.VITE_API_URL as string) || 
  (typeof window !== "undefined" ? (window.location.port === "3000" ? "http://localhost:8000" : window.location.origin) : "http://localhost:8000");

export const appConfig = {
  apiUrl: VITE_API_URL,
  version: "1.0.0",
  appName: "AGRISENSE"
};
