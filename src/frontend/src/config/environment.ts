/**
 * Frontend Environment Validation
 * Ensures all required environment variables are set and valid
 */

interface EnvironmentConfig {
  apiBaseUrl: string;
  wsUrl: string;
  environment: "development" | "staging" | "production";
  enableLogging: boolean;
  enableSentry: boolean;
  sentryDsn?: string;
  apiTimeout: number;
}

class EnvironmentValidator {
  private config: EnvironmentConfig;
  private errors: string[] = [];

  constructor() {
    this.config = {
      apiBaseUrl: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
      wsUrl: import.meta.env.VITE_WS_URL || "ws://localhost:8000/ws",
      environment: ((import.meta.env.VITE_ENVIRONMENT || "development") as "development" | "staging" | "production"),
      enableLogging: import.meta.env.VITE_ENABLE_LOGGING !== "false",
      enableSentry: !!import.meta.env.VITE_SENTRY_DSN,
      sentryDsn: import.meta.env.VITE_SENTRY_DSN,
      apiTimeout: parseInt(import.meta.env.VITE_API_TIMEOUT || "30000"),
    };

    this.validate();
  }

  private validate(): void {
    // Validate API base URL
    if (!this.isValidUrl(this.config.apiBaseUrl)) {
      this.errors.push(`Invalid VITE_API_BASE_URL: ${this.config.apiBaseUrl}`);
    }

    // Validate WebSocket URL
    if (!this.isValidWebSocketUrl(this.config.wsUrl)) {
      this.errors.push(`Invalid VITE_WS_URL: ${this.config.wsUrl}`);
    }

    // Validate environment
    if (!["development", "staging", "production"].includes(this.config.environment)) {
      this.errors.push(`Invalid VITE_ENVIRONMENT: ${this.config.environment}`);
    }

    // Production-specific validations
    if (this.config.environment === "production") {
      // HTTPS enforcement
      if (!this.config.apiBaseUrl.startsWith("https://")) {
        this.errors.push(
          "Production API must use HTTPS (VITE_API_BASE_URL should start with https://)"
        );
      }

      // WSS enforcement
      if (!this.config.wsUrl.startsWith("wss://")) {
        this.errors.push(
          "Production WebSocket must use WSS (VITE_WS_URL should start with wss://)"
        );
      }

      // Sentry should be enabled in production
      if (!this.config.enableSentry) {
        console.warn(
          "Sentry error tracking not enabled in production. Set VITE_SENTRY_DSN."
        );
      }
    }

    // Validate Sentry DSN if provided
    if (this.config.sentryDsn && !this.isValidUrl(this.config.sentryDsn)) {
      this.errors.push(`Invalid VITE_SENTRY_DSN: ${this.config.sentryDsn}`);
    }

    if (this.errors.length > 0) {
      const errorMessage =
        "Environment configuration validation failed:\n" + this.errors.join("\n");
      if (this.config.environment === "production") {
        throw new Error(errorMessage);
      } else {
        console.warn(errorMessage);
      }
    }
  }

  private isValidUrl(url: string): boolean {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }

  private isValidWebSocketUrl(url: string): boolean {
    return url.startsWith("ws://") || url.startsWith("wss://");
  }

  getConfig(): EnvironmentConfig {
    return this.config;
  }

  isProduction(): boolean {
    return this.config.environment === "production";
  }

  isDevelopment(): boolean {
    return this.config.environment === "development";
  }
}

export const environmentValidator = new EnvironmentValidator();
export const appConfig = environmentValidator.getConfig();

export default appConfig;
