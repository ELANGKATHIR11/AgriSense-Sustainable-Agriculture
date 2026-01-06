/**
 * Frontend Security Configuration for React/TypeScript
 */

// API Base URLs - should be set from environment
export const API_CONFIG = {
  // Use environment variables instead of hardcoded URLs
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
  wsURL: import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws",
  timeout: 30000,
  retryAttempts: 3,
  retryDelay: 1000,
};

// Security headers that should be sent with all requests
export const SECURITY_HEADERS = {
  "Content-Type": "application/json",
  "X-Requested-With": "XMLHttpRequest",
  // CSRF token will be added by axios interceptor
};

// CORS configuration
export const CORS_CONFIG = {
  credentials: "include" as const,
  headers: SECURITY_HEADERS,
};

// Input validation rules
export const VALIDATION_RULES = {
  email: {
    pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
    message: "Invalid email address",
  },
  deviceId: {
    pattern: /^[A-Z0-9_-]{1,32}$/,
    message: "Invalid device ID format",
  },
  password: {
    minLength: 12,
    pattern: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{12,}$/,
    message:
      "Password must be at least 12 characters with uppercase, lowercase, number, and special character",
  },
  username: {
    pattern: /^[a-zA-Z0-9_-]{3,32}$/,
    message: "Username must be 3-32 characters (alphanumeric, underscore, dash only)",
  },
  phone: {
    pattern: /^[\d\s()+-]{7,20}$/,
    message: "Invalid phone number",
  },
};

// Sensitive field patterns to never log
export const SENSITIVE_FIELDS = [
  "password",
  "token",
  "apiKey",
  "secret",
  "authorization",
  "credentials",
  "refreshToken",
  "accessToken",
  "privateKey",
];

// CSP Policy (matches backend CSP)
export const CONTENT_SECURITY_POLICY = {
  "default-src": ["'self'"],
  "script-src": ["'self'", "'unsafe-inline'", "'unsafe-eval'", "https://cdn.jsdelivr.net"],
  "style-src": ["'self'", "'unsafe-inline'", "https://cdn.jsdelivr.net"],
  "img-src": ["'self'", "data:", "https:"],
  "connect-src": ["'self'", "https:", "ws:", "wss:"],
  "font-src": ["'self'", "https://fonts.gstatic.com"],
};

// Session management
export const SESSION_CONFIG = {
  tokenKey: "agrisense_token",
  refreshTokenKey: "agrisense_refresh_token",
  userKey: "agrisense_user",
  expirationKey: "agrisense_exp",
  warningTimeBeforeExpiry: 5 * 60 * 1000, // 5 minutes
};

// Rate limiting (frontend)
export const RATE_LIMITING = {
  maxRequestsPerMinute: 100,
  maxConcurrentRequests: 5,
  requestTimeout: 30000,
};

// File upload restrictions
export const FILE_UPLOAD_CONFIG = {
  maxFileSizeMB: 10,
  allowedImageFormats: ["jpg", "jpeg", "png", "webp"],
  allowedDocumentFormats: ["pdf", "xlsx", "csv"],
};

export default {
  API_CONFIG,
  SECURITY_HEADERS,
  CORS_CONFIG,
  VALIDATION_RULES,
  SENSITIVE_FIELDS,
  CONTENT_SECURITY_POLICY,
  SESSION_CONFIG,
  RATE_LIMITING,
  FILE_UPLOAD_CONFIG,
};
